"""Build docs/BVR_RL_Project_Guide.pdf from body.html.

    python docs/guide/build.py

Needs Playwright with Chromium (pip install playwright; playwright install chromium).
Set CHROMIUM to use an existing Chromium instead. The cover shows the commit the
guide was built at and today's date; edit body.html when the code it describes changes.
"""
import datetime
import math
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "BVR_RL_Project_Guide.pdf")
COMMIT = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or "unknown"
_d = datetime.date.today()
DATE = f"{_d.day} {_d:%B %Y}"

CSS = open(os.path.join(HERE, "fonts", "fonts.css")).read() + """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7873;--rule:#e2e0da;--soft:#f4f3ef;--accent:#2a78d6;
      --blue:#2a78d6;--red:#d0452f;--warn:#b4541a;--warnbg:#fbf1ea;--ok:#1b7f55;--okbg:#eaf6ef}
*{box-sizing:border-box}
html{font-family:'IBM Plex Sans',sans-serif;color:var(--ink);font-size:10pt;line-height:1.5;background:#fff}
body{margin:0}
h1{font-size:15pt;margin:0 0 10px;font-weight:600;border-top:2.5px solid var(--ink);padding-top:9px;break-after:avoid}
h2{font-size:12pt;margin:14px 0 5px;font-weight:600;break-after:avoid}
h3{font-size:10.5pt;margin:12px 0 4px;font-weight:600;break-after:avoid}
p{margin:0 0 8px}
code,.mono{font-family:'IBM Plex Mono',monospace;font-size:8.8pt}
pre{font-family:'IBM Plex Mono',monospace;font-size:8.6pt;background:var(--soft);border-left:3px solid var(--accent);
    padding:7px 11px;margin:6px 0 10px;white-space:pre-wrap;break-inside:avoid;line-height:1.5}
.box{border:1px solid var(--rule);background:var(--soft);padding:9px 13px;margin:8px 0 11px;break-inside:avoid}
.box h4{margin:0 0 4px;font-size:10pt}
.warn{border:1px solid #e7c3a8;background:var(--warnbg);padding:9px 13px;margin:8px 0 11px;break-inside:avoid}
.warn h4{margin:0 0 4px;font-size:10pt;color:var(--warn)}
.tip{border:1px solid #b9dcc8;background:var(--okbg);padding:9px 13px;margin:8px 0 11px;break-inside:avoid}
.tip h4{margin:0 0 4px;font-size:10pt;color:var(--ok)}
ul,ol{margin:3px 0 8px;padding-left:18px} li{margin:2px 0}
table{border-collapse:collapse;width:100%;margin:5px 0 11px;font-size:8.9pt;break-inside:auto}
tr{break-inside:avoid}
th{text-align:left;font-weight:600;border-bottom:1.5px solid var(--ink);padding:4px 6px;vertical-align:bottom}
td{border-bottom:1px solid var(--rule);padding:4px 6px;vertical-align:top}
td.n,th.n{text-align:right;white-space:nowrap}
td.n{font-family:'IBM Plex Mono',monospace;font-size:8.6pt}
td.k{font-weight:600}
.steps{counter-reset:s;list-style:none;padding-left:0;margin:6px 0 12px}
.steps>li{counter-increment:s;position:relative;padding:3px 0 5px 32px;break-inside:avoid}
.steps>li::before{content:counter(s);position:absolute;left:0;top:3px;width:21px;height:21px;border-radius:50%;
  background:var(--ink);color:#fff;font-size:9pt;font-weight:600;text-align:center;line-height:21px}
.part{break-before:page;padding-top:40mm}
.part .pn{font-size:11pt;color:var(--muted);letter-spacing:3px}
.part .pt{font-size:28pt;font-weight:600;line-height:1.1;margin:6px 0 10px}
.part .pd{font-size:11.5pt;color:var(--ink2);max-width:150mm}
.part ol{margin-top:16px;font-size:10.5pt}
.chap{margin-top:16px}
.chap.first{break-before:page;margin-top:0}
.cover{height:250mm;display:flex;flex-direction:column;justify-content:space-between}
.cover .t{font-size:38pt;font-weight:600;line-height:1.05;letter-spacing:-.5px;margin-top:30mm}
.cover .s{font-size:15pt;color:var(--ink2);margin-top:10px;max-width:150mm}
.cover .m{font-size:9.5pt;color:var(--muted)}
.toc{columns:2;column-gap:12mm;font-size:9.6pt;padding-left:0;list-style:none}
.toc li{break-inside:avoid;margin:1px 0;display:flex;justify-content:space-between;gap:8px}
.toc .pg{color:var(--muted);font-variant-numeric:tabular-nums}
.toc .ph{font-weight:600;margin-top:8px;color:var(--ink)}
.flow{display:flex;align-items:stretch;gap:0;margin:10px 0 12px;break-inside:avoid}
.flow .nd{flex:1;border:1.5px solid var(--ink);padding:6px 8px;font-size:8.6pt;line-height:1.35;background:#fff}
.flow .nd b{display:block;font-size:9.4pt;margin-bottom:2px}
.flow .ar{width:18px;display:flex;align-items:center;justify-content:center;font-size:13pt;color:var(--muted)}
.chev{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0 10px}
.chev span{border:1.5px solid var(--ink);padding:3px 9px;font-size:8.8pt;font-weight:600}
.chev i{font-style:normal;color:var(--muted);align-self:center}
figure{margin:8px 0 12px;break-inside:avoid}
figcaption{font-size:8.6pt;color:var(--ink2);margin-top:3px}
.two{display:flex;gap:16px;align-items:flex-start} .two>*{flex:1;min-width:0}
.kbd{font-family:'IBM Plex Mono',monospace;font-size:8.6pt;border:1px solid var(--rule);background:var(--soft);padding:0 4px}
.small{font-size:8.8pt;color:var(--ink2)}
"""


def arrow(x, y, ang_deg, color, size=11):
    """Aircraft symbol: a triangle at (x, y) pointing along compass heading ang_deg (0 = up)."""
    a = math.radians(ang_deg)
    def pt(dx, dy):
        # rotate (dx, dy) where -dy is "forward"
        rx = dx * math.cos(a) - dy * math.sin(a)
        ry = dx * math.sin(a) + dy * math.cos(a)
        return f"{x + rx:.1f},{y + ry:.1f}"
    s = size
    return (f'<polygon points="{pt(0, -s)} {pt(s * 0.6, s * 0.7)} {pt(0, s * 0.35)} {pt(-s * 0.6, s * 0.7)}" '
            f'fill="{color}"/>')


def geometry_svg():
    panels = [("Head-on", 180, 1.0), ("Offset left", 180 + 30, 1.0), ("Offset right", 180 - 30, 1.0),
              ("Beam", 90, 1.0), ("Stern", 0, 0.65)]
    w, h = 118, 150
    out = [f'<svg width="{w * 5 + 8}" height="{h + 4}" viewBox="0 0 {w * 5 + 8} {h + 4}" '
           f'style="display:block;font-family:IBM Plex Sans,sans-serif">']
    for i, (name, hdg2, rr) in enumerate(panels):
        x0 = i * w + 4
        cx = x0 + w / 2
        y1, y2 = 120, 120 - 85 * rr
        out.append(f'<rect x="{x0 + 2}" y="2" width="{w - 6}" height="{h - 2}" fill="#f4f3ef" stroke="#e2e0da"/>')
        out.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="#9a9892" stroke-dasharray="3 3"/>')
        out.append(arrow(cx, y1, 0, "#2a78d6"))
        out.append(arrow(cx, y2, hdg2, "#d0452f"))
        out.append(f'<text x="{cx}" y="{h - 8}" text-anchor="middle" font-size="10" fill="#0b0b0b" '
                   f'font-weight="600">{name}</text>')
    out.append("</svg>")
    return "".join(out)


BODY = open(os.path.join(HERE, "body.html"), encoding="utf-8").read()
BODY = BODY.replace("{COMMIT}", COMMIT).replace("{DATE}", DATE).replace("{GEOMETRY_SVG}", geometry_svg())

html = (f'<!doctype html><html><head><meta charset="utf-8"><title>BVR RL project guide</title>'
        f'<style>{CSS}</style></head><body>{BODY}</body></html>')
open(os.path.join(HERE, "guide.html"), "w", encoding="utf-8").write(html)

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or None)
    pg = b.new_page()
    pg.goto("file://" + os.path.join(HERE, "guide.html"))
    pg.wait_for_timeout(800)
    pg.evaluate("document.fonts.ready")
    pg.pdf(path=OUT, format="A4", print_background=True,
           margin={"top": "16mm", "bottom": "18mm", "left": "17mm", "right": "17mm"},
           display_header_footer=True, header_template="<span></span>",
           footer_template='<div style="width:100%;font-size:7.5pt;color:#7a7873;font-family:sans-serif;'
                           'padding:0 17mm;display:flex;justify-content:space-between">'
                           '<span>BVR RL · project guide</span>'
                           '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
    b.close()
print("built", OUT, "at", COMMIT)
