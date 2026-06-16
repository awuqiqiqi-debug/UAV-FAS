#!/usr/bin/env python3
"""将 Markdown 论文转换为 PDF（支持中文）"""
import re, sys, os
from fpdf import FPDF

# ── 配置 ──────────────────────────────────────────────
FONT_DIR = "C:/Windows/Fonts"
FONT_REGULAR = os.path.join(FONT_DIR, "msyh.ttc")   # 微软雅黑
FONT_BOLD    = os.path.join(FONT_DIR, "msyhbd.ttc") # 微软雅黑粗体
INPUT_MD     = os.path.join(os.path.dirname(__file__), "强化学习用于保密能源——流体天线辅助无人机保密通信.md")
OUTPUT_PDF   = os.path.join(os.path.dirname(__file__), "强化学习用于保密能源——流体天线辅助无人机保密通信.pdf")

class PaperPDF(FPDF):
    """自定义 PDF 类，支持中文字体和页眉页脚"""

    def __init__(self):
        super().__init__()
        self.add_font("msyh",  "", FONT_REGULAR)
        self.add_font("msyh", "B", FONT_BOLD)
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        if self.page_no() == 1:
            return  # 首页不加页眉
        self.set_font("msyh", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, "强化学习用于保密能源——流体天线辅助无人机保密通信", align="C")
        self.ln(10)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("msyh", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"- {self.page_no()} -", align="C")
        self.set_text_color(0, 0, 0)


def parse_and_render(pdf: PaperPDF, md_text: str):
    """逐行解析 Markdown 并渲染到 PDF"""
    lines = md_text.split("\n")
    i = 0
    in_table = False
    table_rows = []

    while i < len(lines):
        line = lines[i].rstrip()

        # ── 跳过空行 ──
        if not line.strip():
            if in_table:
                _render_table(pdf, table_rows)
                table_rows = []
                in_table = False
            i += 1
            continue

        # ── 水平分割线 ──
        if line.strip() == "---":
            if in_table:
                _render_table(pdf, table_rows)
                table_rows = []
                in_table = False
            pdf.ln(3)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(5)
            i += 1
            continue

        # ── 代码块 ──
        if line.strip().startswith("```"):
            if in_table:
                _render_table(pdf, table_rows)
                table_rows = []
                in_table = False
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            pdf.set_font("msyh", "", 9)
            pdf.set_fill_color(245, 245, 245)
            pdf.set_text_color(40, 40, 40)
            for cl in code_lines:
                pdf.set_x(25)
                pdf.multi_cell(160, 4.5, cl, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            continue

        # ── 表格行 ──
        if line.strip().startswith("|") and line.strip().endswith("|"):
            stripped = line.strip()
            # 跳过分隔行 |---|---|
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                i += 1
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            table_rows.append(cells)
            in_table = True
            i += 1
            continue

        # ── 标题 ──
        if in_table:
            _render_table(pdf, table_rows)
            table_rows = []
            in_table = False

        m_h = re.match(r"^(#{1,4})\s+(.*)", line)
        if m_h:
            level = len(m_h.group(1))
            text = _clean_md(m_h.group(2))
            if level == 1:
                pdf.set_font("msyh", "B", 18)
                pdf.ln(5)
                pdf.multi_cell(0, 10, text, align="C")
                pdf.ln(3)
            elif level == 2:
                pdf.set_font("msyh", "B", 14)
                pdf.ln(4)
                pdf.multi_cell(0, 8, text)
                pdf.ln(2)
            elif level == 3:
                pdf.set_font("msyh", "B", 12)
                pdf.ln(3)
                pdf.multi_cell(0, 7, text)
                pdf.ln(1)
            else:
                pdf.set_font("msyh", "B", 11)
                pdf.ln(2)
                pdf.multi_cell(0, 6, text)
                pdf.ln(1)
            i += 1
            continue

        # ── 公式行（$$...$$）──
        if line.strip().startswith("$$"):
            formula_lines = [line.strip().replace("$$", "").strip()]
            if not line.strip().endswith("$$") or line.strip() == "$$":
                i += 1
                while i < len(lines) and "$$" not in lines[i]:
                    formula_lines.append(lines[i].strip())
                    i += 1
                if i < len(lines):
                    formula_lines.append(lines[i].strip().replace("$$", "").strip())
            formula_text = " ".join(f for f in formula_lines if f)
            pdf.set_font("msyh", "", 10)
            pdf.set_x(25)
            pdf.multi_cell(160, 5.5, formula_text, align="C")
            pdf.ln(3)
            i += 1
            continue

        # ── 图片占位（![...](...)）──
        if line.strip().startswith("!["):
            i += 1
            continue

        # ── 普通段落 ──
        pdf.set_font("msyh", "", 10.5)
        text = _clean_md(line)
        pdf.multi_cell(0, 6, text)
        pdf.ln(1)
        i += 1

    # flush remaining table
    if in_table:
        _render_table(pdf, table_rows)


def _clean_md(text: str) -> str:
    """去除 Markdown 内联标记，保留纯文本"""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)         # italic
    text = re.sub(r"`(.*?)`", r"\1", text)           # inline code
    text = re.sub(r"\$(.*?)\$", r"\1", text)         # inline math
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # links
    return text


def _render_table(pdf: PaperPDF, rows: list):
    """渲染表格"""
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    usable = 170
    col_w = usable / n_cols
    pdf.set_font("msyh", "", 9)
    for idx, row in enumerate(rows):
        if idx == 0:
            pdf.set_font("msyh", "B", 9)
            pdf.set_fill_color(230, 230, 230)
        else:
            pdf.set_font("msyh", "", 9)
            pdf.set_fill_color(255, 255, 255)
        for c in range(n_cols):
            cell_text = row[c] if c < len(row) else ""
            pdf.cell(col_w, 6, cell_text, border=1, fill=True)
        pdf.ln()
    pdf.ln(3)


def main():
    if not os.path.exists(INPUT_MD):
        print(f"错误: 找不到文件 {INPUT_MD}")
        sys.exit(1)

    with open(INPUT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    pdf = PaperPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    parse_and_render(pdf, md_text)
    pdf.output(OUTPUT_PDF)
    print(f"PDF 已生成: {OUTPUT_PDF}")
    print(f"总页数: {pdf.page_no()}")


if __name__ == "__main__":
    main()
