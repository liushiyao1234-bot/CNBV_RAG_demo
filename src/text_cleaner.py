# PDF文件里有一些类似目录.........这种内容，PDF读取后需要清洗一下先

from pathlib import Path
import json
import re


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


def remove_standalone_page_numbers(text):
    """
    删除单独占一行的页码。

    例如：
        25

    但不会删除正文中的普通数字。
    """

    cleaned_text, replacement_count = re.subn(
        r"(?m)^\s*\d{1,3}\s*$",
        "",
        text
    )

    return cleaned_text, replacement_count


def clean_dot_leaders(text):
    """
    清理目录中的 dot leader。

    例如：
        Presentación .  .  .  .  . 5

    变成：
        Presentación 5

    只处理“连续多个点 + 页码”这种情况，
    避免误删正文里的正常句号。
    """

    pattern = r"[ \t]+(?:\.[ \t]*){3,}(?=\d+\s*$)"

    cleaned_text, replacement_count = re.subn(   # 展示替换次数
        pattern,
        " ",
        text,
        flags=re.MULTILINE
    )

    return cleaned_text, replacement_count


def remove_running_headers(text):
    """
    删除 PDF 中重复出现的独立页眉。

    目前只删除单独占一行的：
        PLD-FT

    不会删除正文中正常出现的 PLD-FT。
    """

    cleaned_text, replacement_count = re.subn(
        r"(?m)^\s*PLD-FT\s*$",
        "",
        text
    )

    return cleaned_text, replacement_count


def normalize_bullets(text):
    """
    PyMuPDF 把原 PDF 的 bullet 符号提取成了字母 w。
    把行首的 w 恢复成统一的 bullet 标记。
    """

    cleaned_text, replacement_count = re.subn(
        r"(?m)^\s*w\s+",
        "• ",
        text
    )

    return cleaned_text, replacement_count


def remove_image_captions(text):
    """
    删除重复的图片说明文字。
    """

    cleaned_text, replacement_count = re.subn(
        r"(?m)^\s*Imagen con fines ilustrativos\s*$",
        "",
        text
    )

    return cleaned_text, replacement_count


def remove_long_running_header(text):
    """
    删除重复页眉：
    PLD-FT | GUÍA PARA EL SUSTENTANTE

    只删除这个前缀。
    如果后面还有真正的章节标题，例如 El Ceneval，
    会把章节标题保留下来。
    """

    cleaned_text, replacement_count = re.subn(
        r"(?mi)^\s*PLD-FT\s*\|\s*GU[IÍí]A PARA EL SUSTENTANTE\s*",
        "",
        text
    )

    return cleaned_text, replacement_count


def clean_pages(pages):
    """
    对所有页面进行文本清洗。
    """

    cleaned_pages = []

    total_replacements = 0

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        cleaned_text, dot_replacements = clean_dot_leaders(text)

        cleaned_text, page_number_replacements = (
            remove_standalone_page_numbers(cleaned_text)
        )

        cleaned_text, header_replacements = (
            remove_running_headers(cleaned_text)
        )

        cleaned_text, bullet_replacements = (
            normalize_bullets(cleaned_text)
        )

        cleaned_text, image_caption_replacements = (
            remove_image_captions(cleaned_text)
        )

        cleaned_text, long_header_replacements = (
            remove_long_running_header(cleaned_text)
        )

        replacement_count = (
            dot_replacements
            + page_number_replacements
            + header_replacements
            + bullet_replacements
            + image_caption_replacements
            + long_header_replacements
        )

        cleaned_pages.append({
            "page": page_number,
            "text": cleaned_text
        })

        total_replacements += replacement_count

    return cleaned_pages, total_replacements


def save_pages(pages, output_path):
    """
    保存清洗后的页面数据。
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            pages,
            file,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":

    input_path = "data/processed/pages.json"
    output_path = "data/processed/pages_cleaned.json"

    pages = load_pages(input_path)

    cleaned_pages, replacement_count = clean_pages(pages)

    save_pages(
        cleaned_pages,
        output_path
    )

    print(f"原始页数: {len(pages)}")
    print(f"清理 dot leader 数量: {replacement_count}")
    print(f"结果已保存到: {output_path}")

    print("\n--- 清洗后的第 3 页 ---")
    print(cleaned_pages[2]["text"])