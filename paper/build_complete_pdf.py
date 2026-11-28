import re
import subprocess
from pathlib import Path
import pypdf

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "paper" / "ADGF_paper_complete.md"
HTML_PATH = ROOT / "paper" / "ADGF_paper_complete.html"
PDF_ROOT = ROOT / "ADGF_paper_complete.pdf"
PDF_PAPER = ROOT / "paper" / "ADGF_paper_complete.pdf"


def latex_inline_to_html(text: str) -> str:
    # Convert $...$ and $$...$$ inline/display math into clean HTML typography
    def _conv_math(expr: str) -> str:
        s = expr.strip()
        repl = [
            (r"\gamma", "&gamma;"), (r"\lambda", "&lambda;"), (r"\kappa", "&kappa;"),
            (r"\alpha", "&alpha;"), (r"\beta", "&beta;"), (r"\epsilon", "&epsilon;"),
            (r"\tau", "&tau;"), (r"\sigma", "&sigma;"), (r"\mu", "&mu;"),
            (r"\pi", "&pi;"), (r"\Delta", "&Delta;"), (r"\nabla", "&nabla;"),
            (r"\partial", "&part;"), (r"\sum", "&Sigma;"), (r"\le", "&le;"),
            (r"\ge", "&ge;"), (r"\times", "&times;"), (r"\pm", "&plusmn;"),
            (r"\approx", "&approx;"), (r"\to", "&rarr;"), (r"\in", "&isin;"),
            (r"\uparrow", "&uarr;"), (r"\downarrow", "&darr;"),
            (r"\quad", " &nbsp; "), (r"\qquad", " &nbsp;&nbsp; "), (r"\,", " "),
            (r"\cdot", "&middot;"), (r"\mathbf{v}", "<b>v</b>"), (r"\mathbf{x}", "<b>x</b>"),
            (r"\ell", "&#8467;"), (r"\bmod", " mod "), (r"\%", "%"),
            (r"\lvert", "|"), (r"\rvert", "|"),
            (r"\bar{R}", "R&#772;"),
            (r"\arg\min", "arg min"), (r"\ln", "ln"), (r"\exp", "exp"), (r"\max", "max"),
        ]
        for k, v in repl:
            s = s.replace(k, v)
        s = re.sub(r"\\text\{([^}]+)\}", r"<span style='font-style:normal'>\1</span>", s)
        s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\sqrt\{([^}]+)\}", r"&radic;<span style='text-decoration:overline'>\1</span>", s)
        s = re.sub(r"\\left\(|\\right\)", lambda m: "(" if "left" in m.group(0) else ")", s)
        s = s.replace(r"\|", "|").replace(r"\{", "{").replace(r"\}", "}")
        # Subscripts and superscripts
        s = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", s)
        s = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", s)
        s = re.sub(r"\^([A-Za-z0-9*+-]+)", r"<sup>\1</sup>", s)
        s = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", s)
        s = s.replace("{", "").replace("}", "")
        return f"<span class='math'>{s}</span>"

    # Display $$...$$
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: f"<div class='eq'>{_conv_math(m.group(1))}</div>", text, flags=re.S)
    # Inline $...$
    text = re.sub(r"\$([^$\n]+?)\$", lambda m: _conv_math(m.group(1)), text)
    return text


def md_inline(text: str) -> str:
    code_spans = []
    def _save_code(m):
        code_spans.append(f"<code>{m.group(1)}</code>")
        return f"\x00CODE{len(code_spans)-1}\x00"
    text = re.sub(r"`([^`]+)`", _save_code, text)
    text = latex_inline_to_html(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    for idx, html_code in enumerate(code_spans):
        text = text.replace(f"\x00CODE{idx}\x00", html_code)
    return text


def parse_md_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    out = []
    i = 0
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            close_lists()
            i += 1
            continue

        if stripped == "---":
            close_lists()
            out.append("<hr/>")
            i += 1
            continue

        # Image
        img_m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if img_m:
            close_lists()
            alt, src = img_m.group(1), img_m.group(2)
            out.append(f"<figure><img src='{src}' alt='{alt}'/></figure>")
            i += 1
            continue

        # Headings
        if stripped.startswith("# "):
            close_lists()
            out.append(f"<h1>{md_inline(stripped[2:])}</h1>")
            i += 1
            continue
        if stripped.startswith("## "):
            close_lists()
            out.append(f"<h2>{md_inline(stripped[3:])}</h2>")
            i += 1
            continue
        if stripped.startswith("### "):
            close_lists()
            out.append(f"<h3>{md_inline(stripped[4:])}</h3>")
            i += 1
            continue

        # Blockquote
        if stripped.startswith("> "):
            close_lists()
            q_lines = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                q_lines.append(md_inline(lines[i].strip()[2:]))
                i += 1
            out.append("<blockquote>" + "<br/>".join(q_lines) + "</blockquote>")
            continue

        # Table
        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|---"):
            close_lists()
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                rows.append(cells)
                i += 1
            tbl = ["<table><thead><tr>"]
            for h in headers:
                tbl.append(f"<th>{md_inline(h)}</th>")
            tbl.append("</tr></thead><tbody>")
            for r in rows:
                tbl.append("<tr>")
                for c in r:
                    tbl.append(f"<td>{md_inline(c)}</td>")
                tbl.append("</tr>")
            tbl.append("</tbody></table>")
            out.append("".join(tbl))
            continue

        # Unordered list
        if stripped.startswith("- "):
            if in_ol:
                out.append("</ol>"); in_ol = False
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{md_inline(stripped[2:])}</li>")
            i += 1
            continue

        # Ordered list
        ol_m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if ol_m:
            if in_ul:
                out.append("</ul>"); in_ul = False
            if not in_ol:
                out.append("<ol>"); in_ol = True
            out.append(f"<li>{md_inline(ol_m.group(2))}</li>")
            i += 1
            continue

        # Paragraph
        close_lists()
        p_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith("#") or nxt.startswith("|") or nxt.startswith("- ") or nxt.startswith("> ") or nxt.startswith("![") or nxt == "---" or re.match(r"^\d+\.\s+", nxt):
                break
            p_lines.append(nxt)
            i += 1
        p_text = " ".join(p_lines)
        if p_text.startswith("*Table ") or p_text.startswith("*Figure "):
            out.append(f"<p class='caption'>{md_inline(p_text)}</p>")
        else:
            out.append(f"<p>{md_inline(p_text)}</p>")

    close_lists()
    return "\n".join(out)


def main():
    md_text = MD_PATH.read_text(encoding="utf-8")
    body = parse_md_to_html(md_text)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ADGF Complete Paper</title>
<style>
  @page {{
    size: letter;
    margin: 0.68in 0.72in 0.72in 0.72in;
  }}
  body {{
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 9.8pt;
    line-height: 1.38;
    color: #111;
    max-width: 100%;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 15.5pt;
    text-align: center;
    line-height: 1.25;
    margin-bottom: 8pt;
  }}
  h1 + p {{
    text-align: center;
    font-size: 9.2pt;
    margin-top: 0;
    margin-bottom: 10pt;
  }}
  h2 {{
    font-size: 11.5pt;
    border-bottom: 0.8pt solid #444;
    padding-bottom: 2pt;
    margin-top: 13pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 10.2pt;
    margin-top: 9pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
  }}
  p {{
    margin: 4pt 0 6pt 0;
    text-align: justify;
  }}
  p.caption {{
    font-size: 8.5pt;
    color: #333;
    margin-top: 3pt;
    margin-bottom: 9pt;
    text-align: left;
  }}
  hr {{
    border: none;
    border-top: 0.6pt solid #bbb;
    margin: 8pt 0;
  }}
  code {{
    font-family: Consolas, "Courier New", monospace;
    font-size: 8.4pt;
    background: #f5f5f5;
    padding: 0.5pt 2pt;
    border-radius: 2pt;
  }}
  .math {{
    font-style: italic;
  }}
  .eq {{
    text-align: center;
    margin: 6pt 0;
    font-size: 9.5pt;
  }}
  blockquote {{
    border-left: 2.5pt solid #666;
    margin: 6pt 0;
    padding: 3pt 10pt;
    background: #fafafa;
    font-size: 9.2pt;
  }}
  ul, ol {{
    margin: 4pt 0 6pt 18pt;
    padding: 0;
  }}
  li {{
    margin-bottom: 3pt;
    text-align: justify;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 7pt 0 3pt 0;
    font-size: 7.7pt;
    line-height: 1.25;
    page-break-inside: auto;
  }}
  tr {{
    page-break-inside: avoid;
    page-break-after: auto;
  }}
  th, td {{
    border-top: 0.5pt solid #ccc;
    border-bottom: 0.5pt solid #ccc;
    padding: 3pt 4pt;
    text-align: left;
    vertical-align: top;
  }}
  thead th {{
    border-top: 1.2pt solid #222;
    border-bottom: 1.0pt solid #222;
    background: #f2f3f5;
    font-weight: bold;
  }}
  tbody tr:last-child td {{
    border-bottom: 1.2pt solid #222;
  }}
  figure {{
    margin: 8pt 0 3pt 0;
    text-align: center;
    page-break-inside: avoid;
  }}
  figure img {{
    width: 100%;
    max-width: 100%;
    height: auto;
    border: 0.5pt solid #ccc;
  }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")

    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    for out_pdf in (PDF_ROOT, PDF_PAPER):
        cmd = [
            edge,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={out_pdf}",
            HTML_PATH.as_uri(),
        ]
        subprocess.run(cmd, check=True)
        reader = pypdf.PdfReader(str(out_pdf))
        print(f"Built {out_pdf} ({len(reader.pages)} pages, {out_pdf.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
