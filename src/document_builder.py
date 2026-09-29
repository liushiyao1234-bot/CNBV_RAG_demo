# 一个公共层，把所有PDF text内容拼成一整段连续文本，避免chunk切到PDF一页末尾是自动结束切片，但是理论上来说PDF一页结束是没有语义意义的，同时记录每一段字符来自PDF源文件的哪一页

from pathlib import Path
import json


def load_pages(input_path):
    """
    读取清洗后的 pages_cleaned.json。
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        pages = json.load(file)

    return pages


def build_document(pages):
    """
    将所有有文本的 PDF 页面拼接成一个连续文档。

    返回：
        full_text:
            整份 PDF 的连续文本

        page_spans:
            每一页在 full_text 中对应的字符范围
    """

    text_parts = []
    page_spans = []

    current_position = 0

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"].strip()

        # 纯图片页或空页暂时跳过
        if not text:
            continue

        # 页面之间加入两个换行
        # 这样既允许 chunk 跨页，
        # 又保留一个明显的自然边界
        if text_parts:
            separator = "\n\n"
            text_parts.append(separator)
            current_position += len(separator)

        start = current_position

        text_parts.append(text)

        current_position += len(text)

        end = current_position

        page_spans.append({
            "page": page_number,
            "start": start,
            "end": end
        })

    full_text = "".join(text_parts)

    return full_text, page_spans


if __name__ == "__main__":

    input_path = "data/processed/pages_cleaned.json"

    pages = load_pages(input_path)

    full_text, page_spans = build_document(pages)

    print(f"原始 PDF 页数: {len(pages)}")
    print(f"有文本的页数: {len(page_spans)}")
    print(f"整份文档字符数: {len(full_text)}")

    print("\n--- 前 500 个字符 ---")
    print(full_text[:500])

    print("\n--- 前 5 个页面范围 ---")
    for span in page_spans[:5]:
        print(span)