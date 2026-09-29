from pathlib import Path
import json

import pymupdf


def load_pdf(pdf_path):
    """
    读取 PDF，并按页提取文本。

    参数:
        pdf_path: PDF 文件路径

    返回:
        pages: 一个 list，每个元素代表一页 PDF
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到 PDF 文件: {pdf_path}")

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        page_data = {
            "page": page_number,
            "text": text.strip()
        }

        pages.append(page_data)

    document.close()

    return pages


def save_pages(pages, output_path):
    """
    将 PDF 解析结果保存成 JSON 文件。

    参数:
        pages: load_pdf() 返回的页面列表
        output_path: JSON 文件保存路径
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            pages,
            file,
            ensure_ascii=False, # 文档是西语，有特殊字母
            indent=2
        )


if __name__ == "__main__":
    pdf_path = "data/raw/Gui_a_CNBV-PLDFT_12a_ed_2026.pdf"
    output_path = "data/processed/pages.json"

    pages = load_pdf(pdf_path)

    save_pages(pages, output_path)

    print(f"成功读取 PDF，共 {len(pages)} 页。")
    print(f"解析结果已保存到: {output_path}")

    print("\n--- 第 1 页 ---")
    print(pages[0]["text"])

    print("\n--- 第 2 页 ---")
    print(pages[1]["text"])