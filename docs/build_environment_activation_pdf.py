#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, findfont


ROOT = Path(__file__).resolve().parent
TEX = ROOT / "environment_activation.tex"
PDF = ROOT / "environment_activation.pdf"
CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
LATIN_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
MONO_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def cleanup(text: str) -> str:
    text = re.sub(r"\\code\{([^}]*)\}", lambda m: f"`{m.group(1)}`", text)
    text = re.sub(r"\\texttt\{\\detokenize\{([^}]*)\}\}", lambda m: f"`{m.group(1)}`", text)
    replacements = {
        r"\_": "_",
        r"\&": "&",
        r"\%": "%",
        r"\#": "#",
        r"\$": "$",
        r"\textbackslash{}": "\\",
        r"--": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z*]+(?:\[[^]]*\])?(?:\{([^{}]*)\})?", lambda m: m.group(1) or "", text)
    text = text.replace("{", "").replace("}", "")
    return " ".join(text.split())


def parse_tex(tex: str):
    blocks = []
    lines = tex.splitlines()
    i = 0
    in_body = False
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if stripped == r"\mainmatter":
            in_body = True
            i += 1
            continue
        if not in_body:
            i += 1
            continue
        if stripped.startswith(r"\backmatter"):
            break
        if not stripped or stripped.startswith("%"):
            i += 1
            continue

        m = re.match(r"\\chapter\{(.+)\}", stripped)
        if m:
            blocks.append(("chapter", cleanup(m.group(1))))
            i += 1
            continue

        m = re.match(r"\\section\{(.+)\}", stripped)
        if m:
            blocks.append(("section", cleanup(m.group(1))))
            i += 1
            continue

        if stripped.startswith(r"\begin{lstlisting}"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(r"\end{lstlisting}"):
                code.append(lines[i].rstrip())
                i += 1
            blocks.append(("code", "\n".join(code).strip("\n")))
            i += 1
            continue

        if stripped.startswith(r"\begin{longtable}"):
            rows = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(r"\end{longtable}"):
                row = lines[i].strip()
                if row and not row.startswith("\\") and "&" in row:
                    row = row.rstrip("\\").strip()
                    cols = [cleanup(c.strip()) for c in row.split("&")]
                    rows.append(cols)
                i += 1
            if rows:
                blocks.append(("table", rows))
            i += 1
            continue

        if stripped.startswith(r"\begin{enumerate}"):
            items = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(r"\end{enumerate}"):
                item = lines[i].strip()
                if item.startswith(r"\item"):
                    items.append(cleanup(item[len(r"\item") :].strip()))
                i += 1
            if items:
                blocks.append(("list", items))
            i += 1
            continue

        if not stripped.startswith("\\"):
            para = [stripped]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if not nxt or nxt.startswith("\\"):
                    break
                para.append(nxt)
                i += 1
            blocks.append(("para", cleanup(" ".join(para))))
            continue

        i += 1
    return blocks


class PdfWriter:
    def __init__(self, output: Path):
        self.output = output
        self.pdf = PdfPages(output)
        self.page_num = 0
        cjk = str(CJK_FONT if CJK_FONT.exists() else findfont("DejaVu Sans"))
        latin = str(LATIN_FONT if LATIN_FONT.exists() else findfont("DejaVu Sans"))
        mono = str(MONO_FONT if MONO_FONT.exists() else findfont("DejaVu Sans Mono"))
        self.font = FontProperties(fname=cjk, size=10)
        self.bold = FontProperties(fname=cjk, size=15, weight="bold")
        self.section = FontProperties(fname=cjk, size=12, weight="bold")
        self.title_font = FontProperties(fname=cjk, size=24, weight="bold")
        self.latin = FontProperties(fname=latin, size=14)
        self.mono = FontProperties(fname=mono, size=8)
        self.fig = None
        self.ax = None
        self.y = 0.93

    def new_page(self):
        if self.fig is not None:
            self.ax.text(0.5, 0.025, str(self.page_num), ha="center", va="bottom", fontsize=8, color="#666")
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
        self.page_num += 1
        self.fig, self.ax = plt.subplots(figsize=(8.27, 11.69))
        self.ax.axis("off")
        self.y = 0.94

    def ensure(self, height: float):
        if self.fig is None or self.y - height < 0.07:
            self.new_page()

    def draw_line(self, text, x=0.08, size=10, prop=None, color="#111", y_gap=0.022):
        self.ensure(y_gap + 0.015)
        self.ax.text(x, self.y, text, fontproperties=prop or self.font, fontsize=size, va="top", color=color)
        self.y -= y_gap

    def draw_wrapped(self, text, x=0.08, width=54, prop=None, size=10, gap=0.022, color="#111"):
        lines = []
        for part in text.split("\n"):
            lines.extend(textwrap.wrap(part, width=width, break_long_words=False, replace_whitespace=False) or [""])
        self.ensure(len(lines) * gap + 0.015)
        for line in lines:
            self.ax.text(x, self.y, line, fontproperties=prop or self.font, fontsize=size, va="top", color=color)
            self.y -= gap
        self.y -= 0.006

    def title_page(self):
        self.new_page()
        self.ax.text(0.5, 0.68, "xgboost_timewindow 环境启用说明", ha="center", va="center",
                     fontproperties=self.title_font)
        self.ax.text(0.5, 0.61, "Environment Activation Guide", ha="center", va="center",
                     fontproperties=self.latin, color="#444")
        self.ax.text(0.5, 0.52, "/home/data/chenyiheng/workspace/2026-dac/xgboost_timewindow",
                     ha="center", va="center", fontproperties=self.mono, fontsize=8, color="#444")
        self.ax.text(0.5, 0.43, "陈一恒", ha="center", va="center", fontproperties=self.font, fontsize=12)
        self.ax.text(0.5, 0.38, "2026-08-16", ha="center", va="center",
                     fontproperties=self.latin, fontsize=10, color="#555")

    def write_blocks(self, blocks):
        for kind, data in blocks:
            if kind == "chapter":
                self.new_page()
                self.draw_wrapped(data, width=28, prop=self.bold, size=15, gap=0.034)
                self.y -= 0.01
            elif kind == "section":
                self.ensure(0.07)
                self.y -= 0.008
                self.draw_wrapped(data, width=36, prop=self.section, size=12, gap=0.028)
            elif kind == "para":
                self.draw_wrapped(data, width=58, prop=self.font, size=9.5, gap=0.021)
            elif kind == "code":
                lines = []
                for raw in data.splitlines():
                    lines.extend(textwrap.wrap(raw, width=76, break_long_words=True, replace_whitespace=False) or [""])
                self.ensure(len(lines) * 0.018 + 0.025)
                self.ax.add_patch(plt.Rectangle((0.065, self.y - len(lines) * 0.018 - 0.01),
                                                0.87, len(lines) * 0.018 + 0.02,
                                                color="#f4f4f4", ec="#d0d0d0", lw=0.4))
                self.y -= 0.006
                for line in lines:
                    self.ax.text(0.08, self.y, line, fontproperties=self.mono, fontsize=7.3, va="top", color="#222")
                    self.y -= 0.018
                self.y -= 0.016
            elif kind == "table":
                for row in data:
                    if len(row) >= 2:
                        text = f"{row[0]}：{'；'.join(row[1:])}"
                    else:
                        text = row[0]
                    self.draw_wrapped("• " + text, x=0.09, width=55, prop=self.font, size=8.8, gap=0.019)
                self.y -= 0.006
            elif kind == "list":
                for idx, item in enumerate(data, 1):
                    self.draw_wrapped(f"{idx}. {item}", x=0.09, width=55, prop=self.font, size=9.2, gap=0.020)

    def close(self):
        if self.fig is not None:
            self.ax.text(0.5, 0.025, str(self.page_num), ha="center", va="bottom", fontsize=8, color="#666")
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
        self.pdf.close()


def main():
    blocks = parse_tex(TEX.read_text(encoding="utf-8"))
    writer = PdfWriter(PDF)
    writer.title_page()
    writer.write_blocks(blocks)
    writer.close()
    print(PDF)


if __name__ == "__main__":
    main()
