"""Converte reports/relatorio_tecnico.md em PDF (UTF-8 + figuras)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MD_PATH = REPORTS / "relatorio_tecnico.md"
PDF_PATH = REPORTS / "relatorio_tecnico.pdf"
HTML_PATH = REPORTS / "relatorio_tecnico.html"

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


class ReportPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Report", size=9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, f"Tech Challenge IADT Fase 1 — página {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _plain(text: str) -> str:
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text.strip()


def _iter_blocks(md: str) -> list[str]:
    return re.split(r"\n\s*\n", md.strip())


def _write(pdf: ReportPDF, text: str, size: float = 11, bold: bool = False, h: float = 6, fill: bool = False) -> None:
    pdf.set_xy(pdf.l_margin, pdf.get_y())
    pdf.set_font("Report", "B" if bold else "", size)
    width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(width, h, text, fill=fill, new_x="LMARGIN", new_y="NEXT")


def render_pdf(md: str) -> ReportPDF:
    pdf = ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_font("Report", "", str(FONT_REGULAR))
    pdf.add_font("Report", "B", str(FONT_BOLD))
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)

    for raw_block in _iter_blocks(md):
        block = raw_block.strip()
        if not block or block == "---":
            continue

        if block.startswith("# "):
            _write(pdf, _plain(block[2:]), size=16, bold=True, h=8)
            pdf.ln(2)
            continue

        if block.startswith("## "):
            pdf.ln(2)
            _write(pdf, _plain(block[3:]), size=13, bold=True, h=7)
            pdf.ln(1)
            continue

        if block.startswith("### "):
            _write(pdf, _plain(block[4:]), size=11.5, bold=True, h=6.5)
            pdf.ln(1)
            continue

        img_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", block)
        if img_match:
            alt, src = img_match.group(1), img_match.group(2)
            path = Path(src)
            if not path.is_absolute():
                path = (REPORTS / src).resolve()
            if path.exists():
                pdf.set_x(pdf.l_margin)
                pdf.image(str(path), x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin)
                pdf.ln(2)
                if alt:
                    pdf.set_text_color(80, 80, 80)
                    _write(pdf, alt, size=9, h=5)
                    pdf.set_text_color(0, 0, 0)
                pdf.ln(1)
            else:
                _write(pdf, f"[figura não encontrada: {src}]", size=10, h=5)
            continue

        if block.startswith("|") and "\n|" in block:
            rows = [
                [cell.strip() for cell in line.strip().strip("|").split("|")]
                for line in block.splitlines()
                if line.strip().startswith("|")
            ]
            rows = [r for r in rows if r and not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
            for row in rows:
                _write(pdf, " | ".join(_plain(c) for c in row), size=8, h=4.5)
            pdf.ln(2)
            continue

        if block.startswith("```"):
            lines = block.splitlines()
            code = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            pdf.set_fill_color(245, 245, 245)
            _write(pdf, code, size=9, h=5, fill=True)
            pdf.ln(2)
            continue

        bullet_lines = block.splitlines()
        if all(
            line.lstrip().startswith(("- ", "* ", "1. ", "2. ", "3. ", "4. ")) or not line.strip()
            for line in bullet_lines
        ):
            for line in bullet_lines:
                line = line.strip()
                if not line:
                    continue
                text = re.sub(r"^(\d+\.\s+|- |\* )", "", line)
                _write(pdf, f"• {_plain(text)}", size=11, h=6)
            pdf.ln(1)
            continue

        _write(pdf, _plain(block), size=11, h=6)
        pdf.ln(1)

    return pdf


def md_to_simple_html(md: str) -> str:
    """HTML imprimível (fallback no navegador)."""
    from markdown import markdown

    def repl(match: re.Match) -> str:
        src = match.group(2)
        path = Path(src)
        if not path.is_absolute():
            path = (REPORTS / src).resolve()
        if not path.exists():
            return f"<p><em>[figura não encontrada: {src}]</em></p>"
        return f'<p><img src="{path}" width="720" alt="{match.group(1)}"></p>'

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)
    body = markdown(text, extensions=["tables", "fenced_code", "nl2br"])
    return (
        "<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
        "<title>Relatório técnico — IADT Fase 1</title>"
        "<style>body{font-family:DejaVu Sans,sans-serif;max-width:900px;"
        "margin:2rem auto;line-height:1.45} img{max-width:100%;height:auto}"
        "table{border-collapse:collapse;width:100%} td,th{border:1px solid #ccc;"
        "padding:.4rem;text-align:left}</style></head><body>"
        f"{body}</body></html>"
    )


def main() -> None:
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Relatório não encontrado: {MD_PATH}")
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        raise FileNotFoundError("Instale fonts-dejavu-core para gerar o PDF.")

    md = MD_PATH.read_text(encoding="utf-8")
    HTML_PATH.write_text(md_to_simple_html(md), encoding="utf-8")
    print(f"HTML: {HTML_PATH}")

    try:
        pdf = render_pdf(md)
        pdf.output(str(PDF_PATH))
        print(f"PDF:  {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")
    except Exception as exc:  # noqa: BLE001
        print(f"Falha ao gerar PDF: {exc}", file=sys.stderr)
        print("O HTML imprimível foi salvo (abra no navegador e use Ctrl+P).")
        raise


if __name__ == "__main__":
    main()
