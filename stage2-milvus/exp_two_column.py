"""
文字版双栏 PDF 解析验证
合成"双栏 / 单栏"两份文字版 PDF → 走完整 parse_pdf → 校验阅读顺序

运行:  python exp_two_column.py
"""
import os
import sys
import tempfile

sys.stdout.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pdfplumber
import pymupdf

from ingest import _detect_columns, parse_pdf

FONT = "china-s"   # PyMuPDF 内置简体中文字体


def build_two_column(path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (50, 70), "双栏版面识别测试文档 —— 验证文字版 PDF 的阅读顺序还原",
        fontsize=15, fontname=FONT,
    )
    for i in range(1, 17):
        page.insert_text((40, 130 + i * 26), f"左栏第{i:02d}行：左栏正文内容，用于验证阅读顺序。",
                         fontsize=11, fontname=FONT)
        page.insert_text((320, 130 + i * 26), f"右栏第{i:02d}行：右栏正文内容，用于验证阅读顺序。",
                         fontsize=11, fontname=FONT)
    doc.save(path)
    doc.close()


def build_single_column(path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i in range(1, 21):
        page.insert_text(
            (45, 60 + i * 30),
            f"单栏第{i:02d}行：这是整页通栏排版的正文内容，行宽接近整页宽度，不应触发双栏检测逻辑。",
            fontsize=11, fontname=FONT,
        )
    doc.save(path)
    doc.close()


def main():
    tmp = tempfile.gettempdir()
    two = os.path.join(tmp, "two_column_test.pdf")
    single = os.path.join(tmp, "single_column_test.pdf")
    build_two_column(two)
    build_single_column(single)
    print(f"测试文件: {two}")
    print(f"          {single}\n")

    # ---- 双栏：应被识别为两栏，阅读顺序 = 标题 → 左栏 → 右栏 ----
    print("===== 双栏 PDF =====")
    with pdfplumber.open(two) as pdf:
        split, top_y = _detect_columns(pdf.pages[0])
    print(f"栏检测: split={split!r}, 正文起始 y={top_y!r}")
    text = parse_pdf(two)
    print("提取文本（前 400 字）:")
    print(text[:400])

    i_title = text.find("双栏版面识别测试文档")
    i_l_end = text.find("左栏第16行")
    i_r_01 = text.find("右栏第01行")
    i_tail = text.find("阅读顺序还原")
    title_intact = (i_title != -1 and text.count("双栏版面识别测试文档") == 1
                    and i_tail != -1 and i_tail < i_l_end)
    left_before_right = i_l_end != -1 and i_r_01 != -1 and i_l_end < i_r_01
    print(f"\n标题完整且先行（未劈开、只出现一次）: {title_intact}")
    print(f"左栏全文先于右栏: {left_before_right}（左栏末行@{i_l_end} < 右栏首行@{i_r_01}）")
    ok_two = title_intact and left_before_right
    print("结果:", "PASS" if ok_two else "FAIL")

    # ---- 单栏：不应误报 ----
    print("\n===== 单栏 PDF =====")
    with pdfplumber.open(single) as pdf:
        split, top_y = _detect_columns(pdf.pages[0])
    print(f"栏检测: {split!r}（应为 None，不触发分栏）")
    text = parse_pdf(single)
    ok_one = split is None and "单栏第01行" in text
    print("结果:", "PASS" if ok_one else "FAIL")

    print("\n===== 总结果:", "PASS" if (ok_two and ok_one) else "FAIL", "=====")


if __name__ == "__main__":
    main()
