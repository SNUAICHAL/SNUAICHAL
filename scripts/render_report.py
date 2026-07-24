from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    return escaped


def block_to_html(block: str) -> str:
    lines = block.strip().splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    list_open = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append("<p>" + inline(" ".join(x.strip() for x in paragraph)) + "</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush_paragraph()
            close_list()
            i += 1
            continue
        if line.startswith("### ") or line.startswith("## ") or line.startswith("# "):
            flush_paragraph()
            close_list()
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level + 1:])}</h{level}>")
            i += 1
            continue
        if line.strip() == "---":
            flush_paragraph()
            close_list()
            out.append("<hr>")
            i += 1
            continue
        if line.startswith("| ") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-+", lines[i + 1]):
            flush_paragraph()
            close_list()
            table_lines = [line]
            i += 2  # skip separator
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = [[inline(c.strip()) for c in row.strip().strip("|").split("|")] for row in table_lines]
            out.append("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr></thead><tbody>")
            for row in rows[1:]:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
            continue
        if line.startswith("- "):
            flush_paragraph()
            if not list_open:
                out.append("<ul>")
                list_open = True
            out.append("<li>" + inline(line[2:]) + "</li>")
            i += 1
            continue
        paragraph.append(line)
        i += 1
    flush_paragraph()
    close_list()
    return "\n".join(out)


def render(markdown_path: Path, html_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    blocks = re.split(r'\n?<div class="page-break"></div>\n?', text)
    pages = "\n".join(f'<section class="report-page">{block_to_html(block)}</section>' for block in blocks)
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 10mm 11mm 10.5mm; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: #172033; font-family: "Noto Sans KR", "Malgun Gothic", sans-serif; font-size: 9.6pt; line-height: 1.38; }}
.report-page {{ break-after: page; page-break-after: always; }}
.report-page:last-child {{ break-after: auto; page-break-after: auto; }}
h1 {{ margin: 0 0 4mm; color: #103b66; font-size: 18pt; line-height: 1.16; letter-spacing: -0.04em; }}
h2 {{ margin: 3.3mm 0 1.5mm; color: #0e5672; font-size: 13.5pt; border-bottom: 1.2px solid #7ba6bb; padding-bottom: 0.6mm; }}
h3 {{ margin: 2.6mm 0 0.8mm; color: #24526a; font-size: 10.8pt; }}
p {{ margin: 0 0 1.7mm; text-align: justify; word-break: keep-all; }}
ul {{ margin: 0.5mm 0 1.7mm 4.6mm; padding-left: 2.8mm; }}
li {{ margin: 0 0 0.6mm; }}
strong {{ color: #0d3f5c; }}
code {{ font-family: Consolas, monospace; font-size: 8.4pt; background: #eef3f6; padding: 0 0.5mm; overflow-wrap: anywhere; }}
table {{ width: 100%; border-collapse: collapse; margin: 1.2mm 0 1.9mm; font-size: 8.2pt; line-height: 1.22; table-layout: fixed; }}
th {{ background: #123f5d; color: white; font-weight: 700; }}
th, td {{ border: 0.35pt solid #9aaebb; padding: 0.75mm 0.8mm; vertical-align: top; word-break: keep-all; }}
tr:nth-child(even) td {{ background: #f2f6f8; }}
hr {{ border: 0; border-top: 0.5pt solid #96a7b0; margin: 2mm 0 1mm; }}
a {{ color: #115b7d; text-decoration: none; }}
</style></head><body>{pages}</body></html>"""
    html_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    render(args.markdown, args.html)


if __name__ == "__main__":
    main()
