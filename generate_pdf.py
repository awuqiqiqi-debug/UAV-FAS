#!/usr/bin/env python3
"""使用fpdf2生成论文PDF"""
from fpdf import FPDF
import re

class PaperPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('SimSun', '', 9)
        self.cell(0, 10, f'第 {self.page_no()} 页', 0, 0, 'C')

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font('SimHei', '', 18)
            self.ln(10)
            self.cell(0, 12, title, 0, 1, 'C')
            self.ln(5)
        elif level == 2:
            self.set_font('SimHei', '', 14)
            self.ln(8)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(3)
        elif level == 3:
            self.set_font('SimHei', '', 12)
            self.ln(6)
            self.cell(0, 8, title, 0, 1, 'L')
            self.ln(2)
        elif level == 4:
            self.set_font('SimHei', '', 11)
            self.ln(4)
            self.cell(0, 7, title, 0, 1, 'L')
            self.ln(1)

    def chapter_body(self, text):
        self.set_font('SimSun', '', 11)
        # 处理段落
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # 检查是否是表格
                if '|' in para and '---' in para:
                    self.add_table(para)
                else:
                    self.multi_cell(0, 6, para.strip(), 0, 'J')
                    self.ln(2)

    def add_table(self, table_text):
        lines = table_text.strip().split('\n')
        if len(lines) < 3:
            return

        # 解析表头
        headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]

        # 跳过分隔行，解析数据行
        rows = []
        for line in lines[2:]:
            if line.strip():
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                rows.append(cells)

        if not rows:
            return

        # 计算列宽
        num_cols = len(headers)
        col_width = (self.w - 2 * self.l_margin) / num_cols

        # 设置表格样式
        self.set_font('SimHei', '', 10)

        # 绘制表头
        self.set_fill_color(240, 240, 240)
        for header in headers:
            self.cell(col_width, 7, header, 1, 0, 'C', True)
        self.ln()

        # 绘制数据行
        self.set_font('SimSun', '', 10)
        for row in rows:
            # 确保行数与列数匹配
            while len(row) < num_cols:
                row.append('')
            for cell in row[:num_cols]:
                self.cell(col_width, 6, cell, 1, 0, 'C')
            self.ln()

        self.ln(3)


def parse_markdown(md_content):
    """解析Markdown内容并添加到PDF"""
    pdf = PaperPDF()
    pdf.add_page()

    # 添加中文字体支持
    # 注意：需要系统中有相应的字体文件
    # 这里使用系统默认字体，实际使用时需要指定正确的字体路径

    lines = md_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 跳过空行
        if not line:
            i += 1
            continue

        # 处理标题
        if line.startswith('# '):
            pdf.chapter_title(line[2:], 1)
        elif line.startswith('## '):
            pdf.chapter_title(line[3:], 2)
        elif line.startswith('### '):
            pdf.chapter_title(line[4:], 3)
        elif line.startswith('#### '):
            pdf.chapter_title(line[5:], 4)
        # 处理表格
        elif '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
            table_lines = [line]
            i += 1
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            pdf.add_table('\n'.join(table_lines))
        # 处理代码块
        elif line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            pdf.set_font('Courier', '', 10)
            pdf.multi_cell(0, 5, '\n'.join(code_lines), 0, 'L')
            pdf.set_font('SimSun', '', 11)
        # 处理普通段落
        else:
            # 收集连续的段落行
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#') and not lines[i].strip().startswith('```') and '|' not in lines[i]:
                para_lines.append(lines[i].strip())
                i += 1
            para_text = ' '.join(para_lines)
            pdf.chapter_body(para_text)
            continue

        i += 1

    return pdf


def main():
    # 读取Markdown文件
    md_path = '强化学习用于保密能源——流体天线辅助无人机保密通信.md'
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 生成PDF
    pdf = parse_markdown(md_content)

    # 保存PDF文件
    pdf_path = '强化学习用于保密能源——流体天线辅助无人机保密通信_v2.pdf'
    pdf.output(pdf_path)

    print(f'PDF文件已生成: {pdf_path}')


if __name__ == '__main__':
    main()
