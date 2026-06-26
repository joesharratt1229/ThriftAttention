#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1200
HEIGHT = 675
SCALE = 2

BG = "#f8fafc"
INK = "#0f172a"
MUTED = "#475569"
GRID_LINE = "#cbd5e1"
FP4 = "#ef4444"
FP4_DARK = "#b91c1c"
FP16 = "#22c55e"
FP16_DARK = "#15803d"
SCAN = "#0284c7"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


TITLE_FONT = _font(44 * SCALE, bold=True)
SUBTITLE_FONT = _font(25 * SCALE)
LABEL_FONT = _font(20 * SCALE, bold=True)
SMALL_FONT = _font(18 * SCALE)
LEGEND_FONT = _font(22 * SCALE)
TINY_FONT = _font(15 * SCALE)


def s(value: int | float) -> int:
    return round(value * SCALE)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    width, height = text_size(draw, text, font)
    draw.text((xy[0] - width // 2, xy[1] - height // 2), text, font=font, fill=fill)


def selected_columns(row: int, *, cols: int, top_k: int) -> list[int]:
    if cols <= top_k:
        return list(range(cols))

    selected = {row % cols}
    selected.add((row * 7 + 3) % cols)
    while len(selected) < top_k:
        selected.add((row * 5 + len(selected) * 7 + 1) % cols)
    return sorted(selected)


def cell_rect(row: int, col: int, *, grid_x: int, grid_y: int, cell: int, gap: int) -> tuple[int, int, int, int]:
    x0 = grid_x + col * (cell + gap)
    y0 = grid_y + row * (cell + gap)
    return x0, y0, x0 + cell, y0 + cell


def blend_channel(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def blend_hex(a: str, b: str, t: float) -> str:
    a = a.lstrip("#")
    b = b.lstrip("#")
    ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
    br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        blend_channel(ar, br, t),
        blend_channel(ag, bg, t),
        blend_channel(ab, bb, t),
    )


def draw_side_key(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    items = [(FP16, FP16_DARK, "FP16"), (FP4, FP4_DARK, "FP4")]
    for i, (fill, outline, label) in enumerate(items):
        yy = y + s(i * 64)
        draw.rounded_rectangle(
            (x, yy, x + s(34), yy + s(34)),
            radius=s(6),
            fill=fill,
            outline=outline,
            width=s(2),
        )
        draw.text((x + s(48), yy + s(2)), label, font=LEGEND_FONT, fill=MUTED)


def draw_frame(
    frame_index: int,
    *,
    rows: int,
    cols: int,
    top_k: int,
    intro_frames: int,
    select_frames: int,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
    draw = ImageDraw.Draw(image)

    cell = s(22)
    gap = s(4)
    grid_w = cols * cell + (cols - 1) * gap
    grid_h = rows * cell + (rows - 1) * gap
    grid_x = (WIDTH * SCALE - grid_w) // 2
    grid_y = s(104)

    draw.text((s(56), s(32)), "ThriftAttention", font=TITLE_FONT, fill=INK)
    draw_side_key(draw, grid_x + grid_w + s(44), grid_y + grid_h // 2 - s(49))

    draw_centered_text(draw, (grid_x + grid_w // 2, grid_y - s(20)), "KV blocks", SMALL_FONT, MUTED)
    draw.text((grid_x - s(104), grid_y + grid_h // 2 - s(12)), "Q blocks", font=SMALL_FONT, fill=MUTED)

    if frame_index < intro_frames:
        progress_rows = 0.0
    else:
        progress_rows = min(1.0, (frame_index - intro_frames + 1) / select_frames) * rows
    current_row = min(rows - 1, max(0, int(progress_rows)))
    row_phase = progress_rows - math.floor(progress_rows)
    pulse = 0.5 + 0.5 * math.sin(frame_index * 0.48)

    for row in range(rows):
        selected = selected_columns(row, cols=cols, top_k=top_k)
        if row < progress_rows:
            reveal_count = len(selected)
        elif row == current_row:
            reveal_count = max(0, min(len(selected), int(row_phase * (len(selected) + 1))))
        else:
            reveal_count = 0
        revealed = set(selected[:reveal_count])

        for col in range(cols):
            x0, y0, x1, y1 = cell_rect(row, col, grid_x=grid_x, grid_y=grid_y, cell=cell, gap=gap)
            if col in revealed:
                fill = blend_hex(FP16, "#86efac", 0.22 * pulse)
                outline = FP16_DARK
            else:
                fill = FP4
                outline = FP4_DARK
            draw.rounded_rectangle((x0, y0, x1, y1), radius=s(4), fill=fill, outline=outline, width=s(1))

    if intro_frames <= frame_index < intro_frames + select_frames:
        y0 = grid_y + current_row * (cell + gap) - s(5)
        y1 = y0 + cell + s(10)
        draw.rounded_rectangle(
            (grid_x - s(8), y0, grid_x + grid_w + s(8), y1),
            radius=s(8),
            outline=SCAN,
            width=s(3),
        )

    caption_y = s(648)
    if frame_index < intro_frames + select_frames:
        caption = "A heuristic selects important query-key block pairs."
    else:
        caption = "Selected block pairs run in FP16, the remainder in FP4"
    draw_centered_text(draw, (WIDTH * SCALE // 2, caption_y), caption, SMALL_FONT, INK)

    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def build_frames(rows: int, cols: int, top_k: int) -> tuple[list[Image.Image], list[int]]:
    intro_frames = 12
    select_frames = 48
    hold_frames = 24
    frames: list[Image.Image] = []
    durations: list[int] = []

    total = intro_frames + select_frames + hold_frames
    for index in range(total):
        frames.append(
            draw_frame(
                index,
                rows=rows,
                cols=cols,
                top_k=top_k,
                intro_frames=intro_frames,
                select_frames=select_frames,
            )
        )
        durations.append(70)
    durations[-1] = 1200
    return frames, durations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the ThriftAttention method GIF.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/social/thriftattention_method.gif"),
        help="Path for the GIF output.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("assets/social/thriftattention_method_preview.png"),
        help="Path for a static PNG preview of the final frame.",
    )
    parser.add_argument("--blocks", type=int, default=20, help="Number of query/KV blocks per side.")
    parser.add_argument("--top-k", type=int, default=2, help="Selected FP16 blocks per query block.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.blocks < 4:
        raise SystemExit("--blocks must be at least 4")
    if args.top_k < 1 or args.top_k > args.blocks:
        raise SystemExit("--top-k must be in [1, --blocks]")

    frames, durations = build_frames(rows=args.blocks, cols=args.blocks, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    frames[-1].save(args.preview)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.preview}")


if __name__ == "__main__":
    main()
