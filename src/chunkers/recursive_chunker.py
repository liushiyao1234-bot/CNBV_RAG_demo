# recursive调用的是连续的json，并且整份文档连续切，页码只做 metadata，不再作为边界；当然了这只是文本结构切片，离语义还差得远hhh

from pathlib import Path
import json

from src.document_builder import load_pages, build_document


def find_split_position(text, target_end, min_end):
    """
    从目标位置向前寻找比较自然的切分位置。

    优先级：
    1. 空行
    2. 句号
    3. 问号
    4. 感叹号
    5. 换行
    6. 空格
    """

    separators = [
        "\n\n",
        ". ",
        ".\n",
        "? ",
        "?\n",
        "! ",
        "!\n",
        "\n",
        " "
    ]

    for separator in separators:

        position = text.rfind(
            separator,
            min_end,
            target_end
        )

        if position != -1:
            return position + len(separator)

    # 实在找不到自然边界，就硬切
    return target_end


def find_overlap_start(text, ideal_start, search_window=80):
    """
    调整 overlap 的起始位置，避免从单词中间开始。

    会在 ideal_start 前后一定范围内寻找最近的自然边界。
    """

    separators = [
        "\n\n",
        ". ",
        ".\n",
        "? ",
        "?\n",
        "! ",
        "!\n",
        "\n",
        " "
    ]

    search_start = max(0, ideal_start - search_window)
    search_end = min(len(text), ideal_start + search_window)

    candidates = []

    for separator in separators:

        # 找 ideal_start 前面的最近边界
        before = text.rfind(
            separator,
            search_start,
            ideal_start
        )

        if before != -1:
            candidates.append(
                before + len(separator)
            )

        # 找 ideal_start 后面的最近边界
        after = text.find(
            separator,
            ideal_start,
            search_end
        )

        if after != -1:
            candidates.append(
                after + len(separator)
            )

    # 一个自然边界都找不到
    if not candidates:
        return ideal_start

    # 找距离 ideal_start 最近的自然边界
    best_start = min(
        candidates,
        key=lambda position: abs(position - ideal_start)
    )

    return best_start


def split_text_recursive(
    text,
    chunk_size=1000,
    overlap=150,  # 大约保留 150 个字符的上下文，但优先保证文本边界完整。
    min_ratio=0.6
):
    """
    对整份连续文本进行切片。

    返回的不只是 chunk 文本，
    还会记录它在 full_text 中的字符位置。
    """

    chunks = []

    start = 0

    while start < len(text):

        target_end = min(
            start + chunk_size,
            len(text)
        )

        # 如果已经到整份文档末尾，就直接结束
        if target_end == len(text):
            end = target_end

        else:
            min_end = start + int(
                chunk_size * min_ratio
            )

            end = find_split_position(
                text=text,
                target_end=target_end,
                min_end=min_end
            )

        # 原始切片
        raw_chunk = text[start:end]

        # 去掉 chunk 两端多余的空格和换行
        chunk_text = raw_chunk.strip()

        if chunk_text:

            # 因为 strip() 会删除前后空白，
            # 所以这里同步修正真正的字符位置
            left_trim = len(raw_chunk) - len(raw_chunk.lstrip())
            right_trim = len(raw_chunk) - len(raw_chunk.rstrip())

            actual_start = start + left_trim
            actual_end = end - right_trim

            chunks.append({
                "start_char": actual_start,
                "end_char": actual_end,
                "char_count": len(chunk_text),
                "text": chunk_text
            })

        # 已经到全文末尾
        if end == len(text):
            break

        # 下一个 chunk 保留 overlap并且取自然的文本边界
        ideal_start = end - overlap

        new_start = find_overlap_start(
            text=text,
            ideal_start=ideal_start
        )

        start = max(
            new_start,
            start + 1
        )

    return chunks


def find_page_range(
    start_char,
    end_char,
    page_spans
):
    """
    根据 chunk 在 full_text 中的位置，
    找出这个 chunk 涉及哪些 PDF 页面。
    """

    source_pages = []

    for span in page_spans:

        # 判断 chunk 和这一页的字符范围是否有交集
        if (
            span["start"] < end_char
            and span["end"] > start_char
        ):
            source_pages.append(span["page"])

    if not source_pages:
        return None, None

    return source_pages[0], source_pages[-1]


def create_recursive_chunks(
    full_text,
    page_spans,
    chunk_size=1000,
    overlap=150
):
    """
    对整份 PDF 连续切片，
    并给每个 chunk 补充页码 metadata。
    """

    text_chunks = split_text_recursive(
        text=full_text,
        chunk_size=chunk_size,
        overlap=overlap
    )

    all_chunks = []

    for chunk_id, chunk in enumerate(
        text_chunks,
        start=1
    ):

        start_page, end_page = find_page_range(
            start_char=chunk["start_char"],
            end_char=chunk["end_char"],
            page_spans=page_spans
        )

        chunk_data = {
            "chunk_id": chunk_id,
            "start_page": start_page,
            "end_page": end_page,
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
            "char_count": chunk["char_count"],
            "text": chunk["text"]
        }

        all_chunks.append(chunk_data)

    return all_chunks


def save_chunks(chunks, output_path):
    """
    保存切片结果。
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":

    input_path = "data/processed/pages_cleaned.json"
    output_path = "data/processed/chunks_recursive.json"

    # 1. 读取清洗后的页面
    pages = load_pages(input_path)

    # 2. 拼成一整份连续文档
    full_text, page_spans = build_document(pages)

    # 3. 对整份文档进行切片
    chunks = create_recursive_chunks(
        full_text=full_text,
        page_spans=page_spans,
        chunk_size=1000,
        overlap=150
    )

    # 4. 保存结果
    save_chunks(
        chunks,
        output_path
    )

    print(f"整份文档字符数: {len(full_text)}")
    print(f"生成 Chunk 数量: {len(chunks)}")
    print(f"结果已保存到: {output_path}")

    # 找一个真正跨页的 chunk 看看
    cross_page_chunks = [
        chunk
        for chunk in chunks
        if chunk["start_page"] != chunk["end_page"]
    ]

    print(
        f"其中跨页 Chunk 数量: "
        f"{len(cross_page_chunks)}"
    )

    if cross_page_chunks:

        print("\n--- 第一个跨页 Chunk ---")

        print(
            json.dumps(
                cross_page_chunks[0],
                ensure_ascii=False,
                indent=2
            )
        )