# 最简单最笨的切片方法，纯按照固定字符长度+overlap切

from pathlib import Path
import json


def load_pages(input_path):
    """
    读取 pdf_parser.py 生成的 pages.json。
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        pages = json.load(file)

    return pages


def split_text_fixed(text, chunk_size=1000, overlap=150):
    """
    使用固定字符长度切分文本。

    参数:
        text: 要切分的文本
        chunk_size: 每个 chunk 最多包含多少字符
        overlap: 相邻 chunk 重叠多少字符

    返回:
        chunks: 切分后的文本列表
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks = []

    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        chunks.append(chunk)

        # 已经切到文本结尾，就结束
        if end == len(text):
            break

        # 下一个 chunk 向前回退 overlap 个字符
        start = end - overlap

    return chunks


def create_fixed_chunks(pages, chunk_size=1000, overlap=150):
    """
    对 PDF 的每一页分别进行固定长度切片。
    """

    all_chunks = []

    chunk_id = 1

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        # 跳过没有文本的纯图片页
        if not text.strip():
            continue

        page_chunks = split_text_fixed(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_in_page, chunk_text in enumerate(page_chunks, start=1):

            chunk_data = {
                "chunk_id": chunk_id, # 整份PDF的第几个chunk
                "page": page_number, # 来自PDF第几页
                "chunk_in_page": chunk_in_page, # 该页的第几个chunk
                "char_count": len(chunk_text),
                "text": chunk_text
            }

            all_chunks.append(chunk_data)

            chunk_id += 1

    return all_chunks


def save_chunks(chunks, output_path):
    """
    将切片结果保存成 JSON。
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":

    # input_path = "data/processed/pages.json"
    input_path = "data/processed/pages_cleaned.json"
    output_path = "data/processed/chunks_fixed.json"

    pages = load_pages(input_path)

    chunks = create_fixed_chunks(
        pages,
        chunk_size=1000,
        overlap=150
    )

    save_chunks(chunks, output_path)

    print(f"原始 PDF 页数: {len(pages)}")
    print(f"生成 Chunk 数量: {len(chunks)}")
    print(f"结果已保存到: {output_path}")

    print("\n--- 第一个 Chunk ---")
    print(chunks[0])