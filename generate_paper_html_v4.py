#!/usr/bin/env python3
"""生成论文HTML v4 - 修复粗体、斜体、图片、行内格式"""
import base64
import os
import re

def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def process_inline(text):
    """处理行内Markdown格式：粗体、斜体、行内代码"""
    # 粗体 **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # 斜体 *text*（但不匹配已处理的HTML标签内的*）
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    # 行内代码 `code`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return text

def md_to_html(md_content):
    """将Markdown转换为HTML，保留MathJax定界符"""
    lines = md_content.split('\n')
    html_lines = []
    i = 0
    in_code_block = False
    in_table = False
    in_list = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 代码块
        if stripped.startswith('```'):
            if in_code_block:
                html_lines.append('</code></pre>')
                in_code_block = False
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append('<pre><code>')
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            html_lines.append(line)
            i += 1
            continue

        # 表格
        if '|' in stripped and stripped.startswith('|'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            if not in_table:
                in_table = True
                html_lines.append('<table>')
                cells = [c.strip() for c in stripped.split('|')[1:-1]]
                html_lines.append('<thead><tr>')
                for c in cells:
                    html_lines.append(f'<th>{process_inline(c)}</th>')
                html_lines.append('</tr></thead>')
                html_lines.append('<tbody>')
                i += 1
                if i < len(lines) and '---' in lines[i]:
                    i += 1
                continue
            else:
                cells = [c.strip() for c in stripped.split('|')[1:-1]]
                html_lines.append('<tr>')
                for c in cells:
                    html_lines.append(f'<td>{process_inline(c)}</td>')
                html_lines.append('</tr>')
                i += 1
                if i >= len(lines) or not lines[i].strip().startswith('|'):
                    html_lines.append('</tbody></table>')
                    in_table = False
                continue
        elif in_table:
            html_lines.append('</tbody></table>')
            in_table = False

        # 空行
        if stripped == '':
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('')
            i += 1
            continue

        # 标题
        if stripped.startswith('#### '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h4>{process_inline(stripped[5:])}</h4>')
            i += 1
            continue
        if stripped.startswith('### '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h3>{process_inline(stripped[4:])}</h3>')
            i += 1
            continue
        if stripped.startswith('## '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h2>{process_inline(stripped[3:])}</h2>')
            i += 1
            continue
        if stripped.startswith('# '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h1>{process_inline(stripped[2:])}</h1>')
            i += 1
            continue

        # 分隔线
        if stripped == '---':
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<hr>')
            i += 1
            continue

        # 图片 ![alt](src)
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
        if img_match:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            alt = img_match.group(1)
            src = img_match.group(2)
            if os.path.exists(src):
                b64 = image_to_base64(src)
                html_lines.append('<div class="figure">')
                html_lines.append(f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:90%;">')
                html_lines.append('</div>')
            else:
                html_lines.append(f'<div class="figure"><p>[图片未找到: {src}]</p></div>')
            i += 1
            continue

        # 引用块（图片说明）
        if stripped.startswith('> '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            content = process_inline(stripped[2:])
            html_lines.append(f'<blockquote>{content}</blockquote>')
            i += 1
            continue

        # 列表
        if stripped.startswith('- '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            content = process_inline(stripped[2:])
            html_lines.append(f'<li>{content}</li>')
            i += 1
            continue
        if re.match(r'^\d+\. ', stripped):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            content = process_inline(re.sub(r'^\d+\. ', '', stripped))
            html_lines.append(f'<li>{content}</li>')
            i += 1
            continue

        # 有序列表结束检测：如果不是列表项但前面有<li>，关闭列表
        if in_list and not stripped.startswith('- ') and not re.match(r'^\d+\. ', stripped):
            html_lines.append('</ul>')
            in_list = False

        # 普通段落 - 处理行内格式但保留$数学公式
        html_lines.append(f'<p>{process_inline(stripped)}</p>')
        i += 1

    if in_list:
        html_lines.append('</ul>')
    if in_table:
        html_lines.append('</tbody></table>')
    if in_code_block:
        html_lines.append('</code></pre>')

    return '\n'.join(html_lines)


def main():
    base_dir = 'C:/Users/红/Desktop/0606强化学习用于保密能源——流体天线辅助无人机保密通信'
    os.chdir(base_dir)

    md_path = '强化学习用于保密能源——流体天线辅助无人机保密通信.md'
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    body_html = md_to_html(md_content)

    html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>强化学习用于保密能源——流体天线辅助无人机保密通信</title>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
@page {size: A4; margin: 25mm 20mm;}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'SimSun','Times New Roman',serif;font-size:12pt;line-height:1.8;color:#000;background:#fff;max-width:210mm;margin:0 auto;padding:25mm 20mm}
h1{font-size:22pt;font-weight:bold;text-align:center;margin:24pt 0 12pt;line-height:1.4}
h2{font-size:16pt;font-weight:bold;margin:18pt 0 12pt;border-bottom:2px solid #000;padding-bottom:4pt}
h3{font-size:14pt;font-weight:bold;margin:14pt 0 10pt}
h4{font-size:12pt;font-weight:bold;margin:12pt 0 8pt}
p{text-indent:2em;margin:6pt 0;text-align:justify}
table{width:100%;border-collapse:collapse;margin:12pt 0;font-size:10.5pt}
th,td{border:1px solid #000;padding:6pt 8pt;text-align:center}
th{background-color:#f0f0f0;font-weight:bold}
caption{font-weight:bold;margin-bottom:6pt;text-align:left}
pre{background-color:#f5f5f5;padding:12pt;border:1px solid #ccc;overflow-x:auto;font-size:10pt;line-height:1.4;margin:12pt 0}
code{font-family:'Courier New',monospace;font-size:10pt;background-color:#f5f5f5;padding:2pt 4pt}
ul,ol{margin:8pt 0 8pt 2em}
li{margin:4pt 0}
blockquote{margin:12pt 0;padding:8pt 16pt;border-left:4px solid #666;background-color:#f9f9f9;font-size:10.5pt}
hr{border:none;border-top:1px solid #ccc;margin:16pt 0}
.figure{text-align:center;margin:16pt 0;page-break-inside:avoid}
.figure img{max-width:90%;height:auto;border:1px solid #ddd}
.abstract{margin:16pt 0;padding:12pt;background-color:#f9f9f9;border:1px solid #ddd}
.abstract h2{border:none;text-align:center;margin-bottom:8pt}
.keywords{margin-top:8pt;font-weight:bold}
.reference{font-size:10.5pt;line-height:1.6}
.reference li{list-style:none;margin:4pt 0;padding-left:2em;text-indent:-2em}
@media print{body{padding:0}h1,h2,h3,h4{page-break-after:avoid}table,pre,blockquote,.figure{page-break-inside:avoid}}
</style>
</head>
<body>
''' + body_html + '''
</body>
</html>'''

    html_path = '强化学习用于保密能源——流体天线辅助无人机保密通信.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    # 验证
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    no_script = re.sub(r'<script>.*?</script>', '', content, flags=re.DOTALL)
    strong_count = no_script.count('<strong>')
    img_count = content.count('data:image/png;base64,')
    dollar_count = no_script.count(chr(36))
    dd_count = no_script.count('$$') // 2

    print(f'HTML generated: {html_path}')
    print(f'Size: {os.path.getsize(html_path)} bytes')
    print(f'<strong> tags: {strong_count}')
    print(f'Embedded images: {img_count}')
    print(f'Inline math $: {dollar_count - dd_count * 2}')
    print(f'Display math $$: {dd_count}')
    has_dd = '**' in no_script
    print(f'No remaining **: {not has_dd}')


if __name__ == '__main__':
    main()
