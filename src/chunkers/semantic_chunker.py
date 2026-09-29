# 语义切片文件
#
# 基本原则：
#
# 1. 双换行 = 强边界，用于先识别段落
# 2. 常见缩写的句号不能切
# 3. 1. / 2. / 3. 这样的列表编号不能误判为句末
# 4. semantic breakpoint 由相邻 semantic unit 的 embedding similarity 决定
# 5. chunk 长度约束使用 embedding 模型自己的 token，而不是字符数
# 6. hard max 只作为兜底，不应该成为主要切片方式


import re
import json
from pathlib import Path
from collections import Counter

import numpy as np

from src.embedding import (
    load_embedding_model,
    embed_for_similarity,
)

from src.document_builder import (
    load_pages,
    build_document,
)


# ============================================================
# 可调参数
# ============================================================

BREAKPOINT_PERCENTILE = 15

MIN_TOKENS = 120

# multilingual-e5-small 的 max_seq_length 是 512。
# 这里不顶满，留出前缀 / special tokens / tokenizer 差异的安全空间。
MAX_TOKENS = 450

# hard max 被触发时，只在 chunk 后 40% 的区域寻找候选切点。
# 450 * 0.6 = 270 tokens
HARD_SEARCH_RATIO = 0.6


# ============================================================
# 文本处理相关常量
# ============================================================

# 用单字符占位符保护“不应该作为句末”的句号。
# 必须使用单字符，因为这样不会改变字符位置。
DOT_PLACEHOLDER = "\uE000"


# ============================================================
# Token 工具
# ============================================================

def count_tokens(
    text,
    model,
    add_special_tokens=True,
):
    """
    使用 embedding 模型自己的 tokenizer
    计算文本真实 token 数。

    truncation=False：
    我们要看到真实长度，不能偷偷截断。
    """

    encoded = model.tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )

    return len(encoded["input_ids"])


# ============================================================
# Semantic Unit
# ============================================================

def split_into_semantic_units(text):
    """
    将整份 full_text 拆成 semantic units。

    semantic unit 通常接近一句话，
    但也可能是标题、列表项等独立语义单位。

    每个 unit 保存：
    - text
    - start_char
    - end_char
    - char_count

    start_char / end_char 对应 full_text 中的位置。
    """

    units = []

    # 找出由双换行分隔的段落。
    # match 会保留每个段落在 full_text 中的位置。
    paragraph_matches = re.finditer(
        r"(?s)(.*?)(?:\n\s*\n|$)",
        text,
    )

    abbreviations = [
        "Dra.",
        "Dr.",
        "Mtra.",
        "Mtro.",
        "Lic.",
        "A.C.",
        "D.R.",
    ]

    for match in paragraph_matches:

        paragraph = match.group(1)

        if not paragraph.strip():
            continue

        paragraph_start = match.start(1)

        protected = paragraph

        # ----------------------------------------------------
        # 1. 保护缩写中的句号
        # ----------------------------------------------------

        for abbreviation in abbreviations:

            protected_abbreviation = abbreviation.replace(
                ".",
                DOT_PLACEHOLDER,
            )

            protected = protected.replace(
                abbreviation,
                protected_abbreviation,
            )

        # ----------------------------------------------------
        # 2. 保护 12a. 之类的版本号 / 序号
        # ----------------------------------------------------

        protected = re.sub(
            r"\b(\d+[a-zA-Z])\.",
            rf"\1{DOT_PLACEHOLDER}",
            protected,
        )

        # ----------------------------------------------------
        # 3. 保护 1. / 2. / 3. 之类的列表编号
        # ----------------------------------------------------

        protected = re.sub(
            r"\b(\d+)\.",
            rf"\1{DOT_PLACEHOLDER}",
            protected,
        )

        # PDF 中的单换行很多只是版式换行。
        #
        # 换成一个空格：
        # \n 和空格都只占一个字符，
        # 因此不会破坏字符坐标。
        protected = protected.replace(
            "\n",
            " ",
        )

        # ----------------------------------------------------
        # 4. 根据 . ! ? 找 semantic units
        # ----------------------------------------------------

        unit_matches = re.finditer(
            r".+?(?:[.!?](?=\s+|$)|$)",
            protected,
        )

        for unit_match in unit_matches:

            raw_unit = unit_match.group(0)

            if not raw_unit.strip():
                continue

            # 去掉 unit 左侧空白，同时修正坐标
            left_trim = (
                len(raw_unit)
                - len(raw_unit.lstrip())
            )

            # 去掉 unit 右侧空白，同时修正坐标
            right_trim = (
                len(raw_unit)
                - len(raw_unit.rstrip())
            )

            local_start = (
                unit_match.start()
                + left_trim
            )

            local_end = (
                unit_match.end()
                - right_trim
            )

            actual_start = (
                paragraph_start
                + local_start
            )

            actual_end = (
                paragraph_start
                + local_end
            )

            unit_text = (
                raw_unit
                .strip()
                .replace(
                    DOT_PLACEHOLDER,
                    ".",
                )
            )

            units.append({
                "text": unit_text,
                "start_char": actual_start,
                "end_char": actual_end,
                "char_count": len(unit_text),
            })

    return units


# ============================================================
# Oversized Semantic Unit 处理
# ============================================================

def split_one_oversized_unit(
    unit,
    full_text,
    model,
    max_tokens,
):
    """
    如果单个 semantic unit 自己就超过 max_tokens，
    进一步按照空白边界切成多个较小 unit。

    尽量不从单词中间切断，
    同时保留它们在 full_text 中的真实字符坐标。
    """

    original_start = unit["start_char"]
    original_end = unit["end_char"]

    original_text = full_text[
        original_start:original_end
    ]

    # 找到每一个非空白 token-like piece 的字符位置。
    word_matches = list(
        re.finditer(
            r"\S+",
            original_text,
        )
    )

    if not word_matches:
        return [unit]

    split_units = []

    start_word_index = 0

    while start_word_index < len(word_matches):

        best_end_word_index = None

        end_word_index = start_word_index

        while end_word_index < len(word_matches):

            local_start = (
                word_matches[start_word_index].start()
            )

            local_end = (
                word_matches[end_word_index].end()
            )

            candidate_text = original_text[
                local_start:local_end
            ]

            token_count = count_tokens(
                text=candidate_text,
                model=model,
                add_special_tokens=True,
            )

            if token_count <= max_tokens:

                best_end_word_index = (
                    end_word_index
                )

                end_word_index += 1

            else:

                break

        # 极端情况：
        # 单独一个连续字符串就超过 max_tokens。
        #
        # 这种情况在普通 PDF 文本里几乎不会发生，
        # 但为了避免死循环，按字符逐步寻找可接受长度。
        if best_end_word_index is None:

            local_start = (
                word_matches[start_word_index].start()
            )

            word_end = (
                word_matches[start_word_index].end()
            )

            best_local_end = None

            for candidate_end in range(
                local_start + 1,
                word_end + 1,
            ):

                candidate_text = original_text[
                    local_start:candidate_end
                ]

                token_count = count_tokens(
                    text=candidate_text,
                    model=model,
                    add_special_tokens=True,
                )

                if token_count <= max_tokens:
                    best_local_end = candidate_end
                else:
                    break

            if best_local_end is None:

                raise ValueError(
                    "发现无法按 token 上限切分的文本。"
                )

            actual_start = (
                original_start
                + local_start
            )

            actual_end = (
                original_start
                + best_local_end
            )

            piece_text = full_text[
                actual_start:actual_end
            ].strip()

            split_units.append({
                "text": piece_text,
                "start_char": actual_start,
                "end_char": actual_end,
                "char_count": len(piece_text),
            })

            # 找到下一个尚未覆盖的 word
            while (
                start_word_index
                < len(word_matches)
                and word_matches[start_word_index].end()
                <= best_local_end
            ):
                start_word_index += 1

            continue

        # 正常情况：
        # 使用最大的、不超过 max_tokens 的单词范围
        local_start = (
            word_matches[start_word_index].start()
        )

        local_end = (
            word_matches[
                best_end_word_index
            ].end()
        )

        actual_start = (
            original_start
            + local_start
        )

        actual_end = (
            original_start
            + local_end
        )

        piece_text = full_text[
            actual_start:actual_end
        ].strip()

        split_units.append({
            "text": piece_text,
            "start_char": actual_start,
            "end_char": actual_end,
            "char_count": len(piece_text),
        })

        start_word_index = (
            best_end_word_index + 1
        )

    return split_units


def normalize_oversized_units(
    units,
    full_text,
    model,
    max_tokens,
):
    """
    检查 semantic units。

    如果一个 unit 自己已经超过 max_tokens，
    在计算 embedding similarity 之前先进一步切小。

    返回：
    - 新 units
    - 被进一步拆分的原始 unit 数量
    """

    normalized_units = []

    oversized_count = 0

    for unit in units:

        token_count = count_tokens(
            text=unit["text"],
            model=model,
            add_special_tokens=True,
        )

        if token_count <= max_tokens:

            normalized_units.append(unit)
            continue

        oversized_count += 1

        smaller_units = (
            split_one_oversized_unit(
                unit=unit,
                full_text=full_text,
                model=model,
                max_tokens=max_tokens,
            )
        )

        normalized_units.extend(
            smaller_units
        )

    return (
        normalized_units,
        oversized_count,
    )


# ============================================================
# Embedding Similarity
# ============================================================

def calculate_adjacent_similarities(
    units,
    model,
):
    """
    计算每个 semantic unit
    与下一个 semantic unit 的 cosine similarity。
    """

    texts = [
        unit["text"]
        for unit in units
    ]

    embeddings = embed_for_similarity(
        model=model,
        texts=texts,
    )

    similarities = []

    for index in range(
        len(embeddings) - 1
    ):

        # embedding.py 已经使用 normalize_embeddings=True，
        # 所以点积等于 cosine similarity。
        similarity = np.dot(
            embeddings[index],
            embeddings[index + 1],
        )

        similarities.append(
            float(similarity)
        )

    return similarities


def calculate_breakpoint_threshold(
    similarities,
    percentile=10,
):
    """
    根据整个文档相邻 similarity 的分布，
    计算 semantic breakpoint 候选阈值。

    percentile=10：
    similarity 最低的 10% 属于候选断点。
    """

    return float(
        np.percentile(
            similarities,
            percentile,
        )
    )


# ============================================================
# Page Metadata
# ============================================================

def find_page_range(
    start_char,
    end_char,
    page_spans,
):
    """
    根据 full_text 中的字符范围，
    判断一个 chunk 来自哪些 PDF 页面。
    """

    source_pages = []

    for span in page_spans:

        if (
            span["start"] < end_char
            and span["end"] > start_char
        ):

            source_pages.append(
                span["page"]
            )

    if not source_pages:
        return None, None

    return (
        source_pages[0],
        source_pages[-1],
    )


# ============================================================
# Semantic Chunking
# ============================================================

def create_semantic_chunks(
    full_text,
    units,
    similarities,
    page_spans,
    model,
    threshold,
    min_tokens=120,
    max_tokens=450,
    hard_search_ratio=0.6,
):
    """
    根据 semantic breakpoint + token 长度约束
    生成最终 semantic chunks。

    规则：

    1. chunk < min_tokens
       即使遇到 semantic breakpoint，也不主动切。

    2. chunk >= min_tokens
       且当前边界 similarity <= threshold
       → semantic_breakpoint。

    3. 如果继续加入 unit 会导致 chunk > max_tokens
       → 回看已经积累的候选边界，
       → 在较靠后的范围内寻找 similarity 最低的位置，
       → hard_max。

    4. 单个 semantic unit 在进入这里之前
       已经由 normalize_oversized_units() 保证
       不超过 max_tokens。
    """

    chunks = []

    start_unit = 0

    hard_min_tokens = max(
        min_tokens,
        int(
            max_tokens
            * hard_search_ratio
        ),
    )

    while start_unit < len(units):

        end_unit = None
        break_reason = None

        # hard_candidates 中保存：
        # {
        #   "unit_index": ...,
        #   "token_count": ...
        # }
        hard_candidates = []

        current_unit = start_unit

        while current_unit < len(units):

            candidate_start_char = (
                units[start_unit]["start_char"]
            )

            candidate_end_char = (
                units[current_unit]["end_char"]
            )

            candidate_text = (
                full_text[
                    candidate_start_char:
                    candidate_end_char
                ]
                .strip()
            )

            current_tokens = count_tokens(
                text=candidate_text,
                model=model,
                add_special_tokens=True,
            )

            # ------------------------------------------------
            # 1. 最优先检查 hard max
            # ------------------------------------------------

            if current_tokens > max_tokens:

                if hard_candidates:

                    # 在所有“长度已经比较靠后但还没超限”的
                    # 候选边界中，选择 similarity 最低的位置。
                    best_candidate = min(
                        hard_candidates,
                        key=lambda item:
                            similarities[
                                item["unit_index"]
                            ],
                    )

                    end_unit = (
                        best_candidate[
                            "unit_index"
                        ]
                    )

                    break_reason = "hard_max"

                elif current_unit > start_unit:

                    # 当前 unit 加进来后突然超过上限，
                    # 且此前还没有进入 hard candidate 区域。
                    #
                    # 那就在前一个完整 unit 结束。
                    end_unit = (
                        current_unit - 1
                    )

                    break_reason = "hard_max"

                else:

                    # 理论上不会出现，
                    # 因为前面已经拆过 oversized unit。
                    raise ValueError(
                        "单个 semantic unit "
                        f"超过 {max_tokens} tokens，"
                        "oversized unit 预处理没有生效。"
                    )

                break

            # ------------------------------------------------
            # 2. 如果没有超限，判断是不是文档结尾
            # ------------------------------------------------

            if (
                current_unit
                == len(units) - 1
            ):

                end_unit = current_unit
                break_reason = "document_end"

                break

            # 当前 unit 和下一个 unit 之间
            # 对应一个 similarity。
            boundary_similarity = (
                similarities[current_unit]
            )

            # ------------------------------------------------
            # 3. 收集 hard max 候选边界
            # ------------------------------------------------

            if (
                current_tokens
                >= hard_min_tokens
            ):

                hard_candidates.append({
                    "unit_index": current_unit,
                    "token_count": current_tokens,
                })

            # ------------------------------------------------
            # 4. 正常 semantic breakpoint
            # ------------------------------------------------

            if (
                current_tokens >= min_tokens
                and boundary_similarity
                <= threshold
            ):

                end_unit = current_unit

                break_reason = (
                    "semantic_breakpoint"
                )

                break

            # ------------------------------------------------
            # 5. 继续加入下一个 unit
            # ------------------------------------------------

            current_unit += 1

        # ----------------------------------------------------
        # 生成最终 chunk
        # ----------------------------------------------------

        start_char = (
            units[start_unit]["start_char"]
        )

        end_char = (
            units[end_unit]["end_char"]
        )

        chunk_text = (
            full_text[
                start_char:end_char
            ]
            .strip()
        )

        actual_token_count = count_tokens(
            text=chunk_text,
            model=model,
            add_special_tokens=True,
        )

        start_page, end_page = (
            find_page_range(
                start_char=start_char,
                end_char=end_char,
                page_spans=page_spans,
            )
        )

        chunks.append({
            "chunk_id": len(chunks) + 1,
            "start_page": start_page,
            "end_page": end_page,
            "start_char": start_char,
            "end_char": end_char,
            "char_count": len(chunk_text),
            "token_count": actual_token_count,
            "break_reason": break_reason,
            "text": chunk_text,
        })

        # 下一块从当前结束 unit 的下一项开始
        start_unit = end_unit + 1

    return chunks


# ============================================================
# Validation
# ============================================================

def validate_chunks(
    chunks,
    max_tokens,
):
    """
    最终安全检查。

    确认：
    - 没有空 chunk
    - 没有 chunk 超过 token hard max
    """

    empty_chunks = []

    over_limit_chunks = []

    for chunk in chunks:

        if not chunk["text"].strip():

            empty_chunks.append(
                chunk["chunk_id"]
            )

        if (
            chunk["token_count"]
            > max_tokens
        ):

            over_limit_chunks.append({
                "chunk_id":
                    chunk["chunk_id"],

                "token_count":
                    chunk["token_count"],
            })

    if empty_chunks:

        raise ValueError(
            "发现空 Chunk: "
            f"{empty_chunks}"
        )

    if over_limit_chunks:

        raise ValueError(
            "发现超过 Token 上限的 Chunk: "
            f"{over_limit_chunks}"
        )


# ============================================================
# Save
# ============================================================

def save_chunks(
    chunks,
    output_path,
):
    """
    保存 semantic chunks。
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Statistics
# ============================================================

def show_chunk_statistics(
    chunks,
    threshold,
    max_tokens,
):
    """
    打印最终 semantic chunk 统计。
    """

    char_lengths = np.array([
        chunk["char_count"]
        for chunk in chunks
    ])

    token_lengths = np.array([
        chunk["token_count"]
        for chunk in chunks
    ])

    reasons = Counter(
        chunk["break_reason"]
        for chunk in chunks
    )

    print(
        "\n--- Semantic Chunk 统计 ---"
    )

    print(
        f"Semantic breakpoint 阈值: "
        f"{threshold:.4f}"
    )

    print(
        f"总 Chunk 数量: "
        f"{len(chunks)}"
    )

    print(
        "\n--- Character 长度 ---"
    )

    print(
        f"平均: "
        f"{char_lengths.mean():.1f} chars"
    )

    print(
        f"最短: "
        f"{char_lengths.min()} chars"
    )

    print(
        f"最长: "
        f"{char_lengths.max()} chars"
    )

    print(
        "\n--- Token 长度 ---"
    )

    print(
        f"Hard Max: "
        f"{max_tokens} tokens"
    )

    print(
        f"平均: "
        f"{token_lengths.mean():.1f} tokens"
    )

    print(
        f"最短: "
        f"{token_lengths.min()} tokens"
    )

    print(
        f"最长: "
        f"{token_lengths.max()} tokens"
    )

    print(
        "\n--- Break Reasons ---"
    )

    for reason, count in reasons.items():

        print(
            f"{reason}: {count}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    input_path = (
        "data/processed/"
        "pages_cleaned.json"
    )

    output_path = (
        "data/processed/"
        "chunks_semantic.json"
    )

    # --------------------------------------------------------
    # 1. 读取清洗后的 PDF
    # --------------------------------------------------------

    pages = load_pages(
        input_path
    )

    full_text, page_spans = (
        build_document(pages)
    )

    print(
        f"整份文档字符数: "
        f"{len(full_text)}"
    )

    # --------------------------------------------------------
    # 2. 初步生成 semantic units
    # --------------------------------------------------------

    units = split_into_semantic_units(
        full_text
    )

    print(
        f"初始 Semantic Unit 数量: "
        f"{len(units)}"
    )

    # --------------------------------------------------------
    # 3. 加载 Embedding 模型
    # --------------------------------------------------------

    model = load_embedding_model()

    print(
        f"Embedding 模型最大序列长度: "
        f"{model.max_seq_length} tokens"
    )

    if MAX_TOKENS >= model.max_seq_length:

        raise ValueError(
            f"MAX_TOKENS={MAX_TOKENS} "
            f"不能达到或超过模型上限 "
            f"{model.max_seq_length}。"
        )

    # --------------------------------------------------------
    # 4. 处理单个 unit 自己就过长的情况
    # --------------------------------------------------------

    units, oversized_unit_count = (
        normalize_oversized_units(
            units=units,
            full_text=full_text,
            model=model,
            max_tokens=MAX_TOKENS,
        )
    )

    print(
        f"超过 {MAX_TOKENS} tokens "
        f"并被进一步拆分的原始 Unit 数量: "
        f"{oversized_unit_count}"
    )

    print(
        f"处理后的 Semantic Unit 数量: "
        f"{len(units)}"
    )

    # --------------------------------------------------------
    # 5. 计算相邻 semantic similarity
    # --------------------------------------------------------

    similarities = (
        calculate_adjacent_similarities(
            units=units,
            model=model,
        )
    )

    # --------------------------------------------------------
    # 6. 计算 candidate breakpoint threshold
    # --------------------------------------------------------

    threshold = (
        calculate_breakpoint_threshold(
            similarities=similarities,
            percentile=(
                BREAKPOINT_PERCENTILE
            ),
        )
    )

    # --------------------------------------------------------
    # 7. 真正进行 Semantic Chunking
    # --------------------------------------------------------

    chunks = create_semantic_chunks(
        full_text=full_text,
        units=units,
        similarities=similarities,
        page_spans=page_spans,
        model=model,
        threshold=threshold,
        min_tokens=MIN_TOKENS,
        max_tokens=MAX_TOKENS,
        hard_search_ratio=(
            HARD_SEARCH_RATIO
        ),
    )

    # --------------------------------------------------------
    # 8. 最终安全检查
    # --------------------------------------------------------

    validate_chunks(
        chunks=chunks,
        max_tokens=MAX_TOKENS,
    )

    # --------------------------------------------------------
    # 9. 保存
    # --------------------------------------------------------

    save_chunks(
        chunks=chunks,
        output_path=output_path,
    )

    # --------------------------------------------------------
    # 10. 打印统计
    # --------------------------------------------------------

    show_chunk_statistics(
        chunks=chunks,
        threshold=threshold,
        max_tokens=MAX_TOKENS,
    )

    print(
        f"\n结果已保存到: "
        f"{output_path}"
    )