import os
import subprocess
from markdown_it import MarkdownIt

def build_pdf():
    md_path = r"d:\Applications\guimlab\paper\guimlab_whitepaper.md"
    html_path = r"d:\Applications\guimlab\paper\guimlab_whitepaper.html"
    pdf_path = r"d:\Applications\guimlab\paper\guimlab_whitepaper.pdf"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # replace relative figures path so images render locally
    md_text = md_text.replace("](figures/", "](file:///d:/Applications/guimlab/figures/")
    md_text = md_text.replace("](../figures/", "](file:///d:/Applications/guimlab/figures/")

    md = MarkdownIt()
    content_html = md.render(md_text)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GuimLab Whitepaper</title>
<style>
  @page {{
    size: letter;
    margin: 20mm 15mm 20mm 15mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: #1a1a1a;
    margin: 0 auto;
    max-width: 800px;
    padding: 20px;
  }}
  h1 {{
    font-size: 20pt;
    text-align: center;
    margin-bottom: 12px;
    color: #0f172a;
  }}
  h2 {{
    font-size: 14pt;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
    margin-top: 24px;
    margin-bottom: 10px;
    color: #1e293b;
  }}
  h3 {{
    font-size: 12pt;
    margin-top: 16px;
    margin-bottom: 8px;
    color: #334155;
  }}
  p, li {{
    text-align: justify;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 18px 0;
    font-size: 9.5pt;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 8px 10px;
    text-align: left;
  }}
  th {{
    background-color: #f1f5f9;
    font-weight: 600;
  }}
  img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
  }}
  code {{
    font-family: 'Consolas', 'Courier New', Courier, monospace;
    font-size: 9pt;
    background: #f1f5f9;
    padding: 2px 4px;
    border-radius: 3px;
  }}
  pre {{
    background: #0f172a;
    color: #f8fafc;
    padding: 12px;
    font-size: 8.5pt;
    border-radius: 6px;
    overflow-x: auto;
  }}
  pre code {{
    background: transparent;
    color: inherit;
  }}
</style>
</head>
<body>
{content_html}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    chrome_cmd = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "--headless",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--run-all-compositor-stages-before-draw",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    subprocess.run(chrome_cmd, check=True)
    print(f"Generated PDF: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
