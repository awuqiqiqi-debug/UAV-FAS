#!/usr/bin/env python3
"""将论文Markdown转换为HTML格式"""
import re
import os

def markdown_to_html(md_content):
    """将Markdown内容转换为HTML"""
    html = md_content

    # 标题转换
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)

    # 粗体和斜体
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # 数学公式（简单处理）
    html = re.sub(r'\$\$(.+?)\$\$', r'<div class="math">\1</div>', html, flags=re.DOTALL)
    html = re.sub(r'\$(.+?)\$', r'<span class="math">\1</span>', html)

    # 表格
    def convert_table(match):
        lines = match.group(0).strip().split('\n')
        if len(lines) < 3:
            return match.group(0)

        # 解析表头
        headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]

        # 跳过分隔行
        rows = []
        for line in lines[2:]:
            if line.strip():
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                rows.append(cells)

        # 生成HTML表格
        table_html = '<table>\n<thead>\n<tr>\n'
        for header in headers:
            table_html += f'<th>{header}</th>\n'
        table_html += '</tr>\n</thead>\n<tbody>\n'

        for row in rows:
            table_html += '<tr>\n'
            for cell in row:
                table_html += f'<td>{cell}</td>\n'
            table_html += '</tr>\n'

        table_html += '</tbody>\n</table>'
        return table_html

    html = re.sub(r'(\|.+\|\n\|[-| ]+\|\n(?:\|.+\|\n?)+)', convert_table, html)

    # 代码块
    html = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code class="\1">\2</code></pre>', html, flags=re.DOTALL)

    # 行内代码
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

    # 列表
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*</li>\n?)+', lambda m: '<ul>\n' + m.group(0) + '</ul>', html)

    # 引用
    html = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)

    # 分隔线
    html = re.sub(r'^---$', '<hr>', html, flags=re.MULTILINE)

    # 段落（简单的换行处理）
    html = re.sub(r'\n\n', '</p>\n<p>', html)
    html = '<p>' + html + '</p>'

    # 清理空段落
    html = re.sub(r'<p>\s*</p>', '', html)
    html = re.sub(r'<p>(<h[1-6]>)', r'\1', html)
    html = re.sub(r'(</h[1-6]>)</p>', r'\1', html)
    html = re.sub(r'<p>(<table>)', r'\1', html)
    html = re.sub(r'(</table>)</p>', r'\1', html)
    html = re.sub(r'<p>(<pre>)', r'\1', html)
    html = re.sub(r'(</pre>)</p>', r'\1', html)
    html = re.sub(r'<p>(<div class="math">)', r'\1', html)
    html = re.sub(r'(</div>)</p>', r'\1', html)
    html = re.sub(r'<p>(<ul>)', r'\1', html)
    html = re.sub(r'(</ul>)</p>', r'\1', html)
    html = re.sub(r'<p>(<blockquote>)', r'\1', html)
    html = re.sub(r'(</blockquote>)</p>', r'\1', html)
    html = re.sub(r'<p>(<hr>)', r'\1', html)
    html = re.sub(r'(</?hr>)</p>', r'\1', html)

    return html


def generate_full_html(body_content):
    """生成完整的HTML文档"""
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>强化学习用于保密能源——流体天线辅助无人机保密通信</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        @page {{
            size: A4;
            margin: 20mm;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'SimSun', 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.8;
            color: #000;
            background: #fff;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20mm;
        }}
        h1 {{
            font-size: 22pt;
            font-weight: bold;
            text-align: center;
            margin: 24pt 0 12pt;
            line-height: 1.4;
        }}
        h2 {{
            font-size: 16pt;
            font-weight: bold;
            margin: 18pt 0 12pt;
            border-bottom: 1px solid #ccc;
            padding-bottom: 4pt;
        }}
        h3 {{
            font-size: 14pt;
            font-weight: bold;
            margin: 14pt 0 10pt;
        }}
        h4 {{
            font-size: 12pt;
            font-weight: bold;
            margin: 12pt 0 8pt;
        }}
        p {{
            text-indent: 2em;
            margin: 6pt 0;
            text-align: justify;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12pt 0;
            font-size: 10.5pt;
        }}
        th, td {{
            border: 1px solid #000;
            padding: 6pt 8pt;
            text-align: center;
        }}
        th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
        .math {{
            text-align: center;
            margin: 12pt 0;
            font-style: italic;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 12pt;
            border: 1px solid #ccc;
            border-radius: 4pt;
            overflow-x: auto;
            font-size: 10pt;
            line-height: 1.4;
            margin: 12pt 0;
        }}
        code {{
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            background-color: #f5f5f5;
            padding: 2pt 4pt;
            border-radius: 2pt;
        }}
        ul, ol {{
            margin: 8pt 0 8pt 2em;
        }}
        li {{
            margin: 4pt 0;
        }}
        blockquote {{
            margin: 12pt 0;
            padding: 8pt 16pt;
            border-left: 4px solid #ccc;
            background-color: #f9f9f9;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ccc;
            margin: 16pt 0;
        }}
        .title-block {{
            text-align: center;
            margin-bottom: 24pt;
        }}
        .abstract {{
            margin: 16pt 0;
            padding: 12pt;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
        }}
        .abstract h2 {{
            border: none;
            text-align: center;
            margin-bottom: 8pt;
        }}
        .keywords {{
            margin-top: 8pt;
            font-weight: bold;
        }}
        @media print {{
            body {{
                padding: 0;
            }}
            h1, h2, h3, h4 {{
                page-break-after: avoid;
            }}
            table, pre, blockquote {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{body_content}
</body>
</html>'''


def main():
    # 读取Markdown文件
    md_path = '强化学习用于保密能源——流体天线辅助无人机保密通信.md'
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 转换为HTML
    body_html = markdown_to_html(md_content)

    # 生成完整HTML
    full_html = generate_full_html(body_html)

    # 保存HTML文件
    html_path = '强化学习用于保密能源——流体天线辅助无人机保密通信.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f'HTML文件已生成: {html_path}')


if __name__ == '__main__':
    main()
