"""Generate this README's dot-matrix SVGs. Usage: python3 assets/make_assets.py assets

An LED-board header with the project name and its 12x12 icon, plus a scrolling dot
equaliser for the footer. Same graphite palette as github.com/kartikeyaagr.
"""


# ── Shared below this line: same dot-matrix kit as github.com/kartikeyaagr ────
import math
import random
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
OUT.mkdir(parents=True, exist_ok=True)

CARD, BORDER, OFF = "#111317", "#22262e", "#22262e"          # tile, outline, unlit dot
TEXT, MID, ACCENT = "#e6e8ec", "#5d6470", "#5b8cff"           # lit dot colours
COLOUR = {"t": TEXT, "a": ACCENT, "m": MID}

# 5x7 dot font
FONT = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ", """
01110100011000111111100011000110001 11110100011000111110100011000111110 01110100011000010000100001000101110
11110100011000110001100011000111110 11111100001000011110100001000011111 11111100001000011110100001000010000
01110100011000010111100011000101111 10001100011000111111100011000110001 01110001000010000100001000010001110
00111000100001000010000101001001100 10001100101010011000101001001010001 10000100001000010000100001000011111
10001110111010110101100011000110001 10001100011100110101100111000110001 01110100011000110001100011000101110
11110100011000111110100001000010000 01110100011000110001101011001001101 11110100011000111110101001001010001
01111100001000001110000010000111110 11111001000010000100001000010000100 10001100011000110001100011000101110
10001100011000110001100010101000100 10001100011000110101101011010101010 10001100010101000100010101000110001
10001100010101000100001000010000100 11111000010001000100010001000011111 01110100011001110101110011000101110
00100011000010000100001000010001110 01110100010000100010001000100011111 11111000100010000010000011000101110
00010001100101010010111110001000010 11111100001111000001000011000101110 00110010001000011110100011000101110
11111000010001000100010000100001000 01110100011000101110100011000101110 01110100011000101111000010001001100
00000000000000011111000000000000000 00000000000000000000000000000000000""".split()))


def svg(name, w, h, body, label, defs=""):
    (OUT / f"{name}.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{escape(label)}">\n<defs>{defs}</defs>\n{body}\n</svg>\n')


def card(w, h, rx):
    return f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="{rx}" fill="{CARD}" stroke="{BORDER}"/>'


def word(text, col, row):
    cells, c = [], col
    for ch in text:
        bits = FONT[ch]
        cells += [(c + i % 5, row + i // 5) for i, b in enumerate(bits) if b == "1"]
        c += 6
    return cells


def line(grid, a, b, ch):
    """Bresenham line of `ch` cells on a mutable grid, from cell a to cell b."""
    (c0, r0), (c1, r1) = a, b
    dc, dr = abs(c1 - c0), -abs(r1 - r0)
    sc, sr, err = (1 if c0 < c1 else -1), (1 if r0 < r1 else -1), abs(c1 - c0) - abs(r1 - r0)
    while True:
        grid[r0][c0] = ch
        if (c0, r0) == (c1, r1):
            return
        e2 = 2 * err
        if e2 >= dr:
            err += dr; c0 += sc
        if e2 <= dc:
            err += dc; r0 += sr


# ── Header: LED board with the project name swept on, its icon lit beside it ──
def header():
    P, n = 20, len(TITLE)
    grid, alpha = ICON
    rows = max(9 * n + 4, 16)
    top = 3 if n > 1 else (rows - 7) // 2
    text_end = 4 + 6 * max(map(len, TITLE)) - 1
    cur_c, cur_r = 4 + 6 * len(TITLE[-1]) + 1, top + 9 * (n - 1)
    ic, ir = max(text_end, cur_c + 5) + 4, (rows - 12) // 2
    cols = ic + 12 + 3
    w, h = cols * P, rows * P
    lit = set()
    for i, t in enumerate(TITLE):
        lit |= set(word(t, 4, top + 9 * i))
    cursor = {(c, r) for c in range(cur_c, cur_c + 5) for r in range(cur_r, cur_r + 7)}
    icon_cells = {(ic + c, ir + r): (ch, alpha[r][c] if alpha else 1)
                  for r, row in enumerate(grid) for c, ch in enumerate(row) if ch != "."}
    taken = lit | cursor | set(icon_cells)
    rnd = random.Random(11)
    free = [(c, r) for c in range(1, cols - 1) for r in range(1, rows - 1) if (c, r) not in taken]
    twinkle = rnd.sample(free, max(6, cols * rows // 100))
    dot = lambda c, r, rad: f'<circle cx="{c * P + P / 2}" cy="{r * P + P / 2}" r="{rad}"'
    out = [card(w, h, 28)]
    out += [f'{dot(c, r, 2.6)} fill="{OFF}"/>' for c, r in free]
    by_col = {}
    for c, r in lit:
        by_col.setdefault(c, []).append(r)
    for c in sorted(by_col):          # one group per column: lights up in a left-to-right sweep
        dots = "".join(f"{dot(c, r, 7)}/>" for r in by_col[c])
        out.append(f'<g fill="{TEXT}" opacity="0"><set attributeName="opacity" to="1" '
                   f'begin="{0.2 + (c - 4) * 0.03:.2f}s" fill="freeze"/>{dots}</g>')
    dots = "".join(f"{dot(c, r, 7)}/>" for c, r in cursor)
    out.append(f'<g fill="{ACCENT}"><animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.5;0.5;1" '
               f'dur="1.1s" repeatCount="indefinite"/>{dots}</g>')
    icon_begin = 0.2 + (text_end - 4) * 0.03 + 0.15
    dots = "".join(f'{dot(c, r, 7)} fill="{COLOUR[ch]}" opacity="{a:.2f}"/>' for (c, r), (ch, a) in icon_cells.items())
    out.append(f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" begin="{icon_begin:.2f}s" '
               f'dur="0.6s" fill="freeze"/>{dots}</g>')
    for c, r in twinkle:
        out.append(f'{dot(c, r, 3.4)} fill="{ACCENT}" opacity="0">'
                   f'<animate attributeName="opacity" values="0;0.9;0" dur="{rnd.uniform(2.5, 4.5):.1f}s" '
                   f'begin="{rnd.uniform(1.5, 6):.1f}s" repeatCount="indefinite"/></circle>')
    svg("header", w, h, "\n".join(out), LABEL)


# ── Footer: an equaliser of dots, scrolling one wavelength per loop ───────────
def footer():
    P, cols, rows, L = 20, 61, 6, 20
    w, h = cols * P, rows * P
    dots = []
    for c in range(cols + L):
        height = 1 + round(2.5 + 2.5 * math.sin(2 * math.pi * c / L))
        for r in range(rows):
            lit = r >= rows - height
            fill, rad = (ACCENT, 4.2) if lit else (OFF, 2.4)
            op = f' opacity="{0.35 + 0.65 * (r + 1) / rows:.2f}"' if lit else ""
            dots.append(f'<circle cx="{c * P + P / 2}" cy="{r * P + P / 2}" r="{rad}" fill="{fill}"{op}/>')
    body = (f'<g clip-path="url(#clip)"><g>{"".join(dots)}'
            f'<animateTransform attributeName="transform" type="translate" from="0 0" to="{-L * P} 0" '
            f'dur="6s" repeatCount="indefinite"/></g></g>')
    svg("footer", w, h, body, "", f'<clipPath id="clip"><rect width="{w}" height="{h}"/></clipPath>')



# ── This repo ────────────────────────────────────────────────────────────────
TITLE = ["PMLM"]
LABEL = "PMLM: Per My Last Model"
ICON = ([
    ".tttttttttt.",
    "t..........t",
    "t.mmmmmmm..t",
    "t..........t",
    "t.aaaaaaaa.t",
    "t..........t",
    "t.mmmmm....t",
    ".tttttttttt.",
    "...tt.......",
    "..tt........",
    ".t..........",
    "............",
], None)  # a speech bubble with the real meaning lit up


assert all(len(v) == 35 for v in FONT.values()) and len(FONT) == 38, "5x7 font table is malformed"
assert len(ICON[0]) == 12 and all(len(r) == 12 for r in ICON[0]), "icon must be 12x12"
header()
footer()
print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))
