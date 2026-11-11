"""Draw the HomeLens toolbar icons. Run with: python extension/icons/make_icons.py"""
from pathlib import Path

from PIL import Image, ImageDraw

GREEN = (11, 107, 87, 255)
WHITE = (255, 255, 255, 255)


def draw_icon(size: int) -> Image.Image:
    scale = 8                      # draw big, then shrink for smooth edges
    s = size * scale
    image = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(image)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=GREEN)
    # lens ring
    c, r, w = s * 0.44, s * 0.30, max(2, int(s * 0.075))
    d.ellipse([c - r, c - r, c + r, c + r], outline=WHITE, width=w)
    # handle
    d.line([c + r * 0.72, c + r * 0.72, s * 0.86, s * 0.86], fill=WHITE, width=int(w * 1.3))
    # house inside the lens
    h = r * 0.52
    d.polygon([(c - h, c + h * 0.15), (c, c - h * 0.85), (c + h, c + h * 0.15), (c + h, c + h), (c - h, c + h)], fill=WHITE)
    return image.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for size in (16, 48, 128):
        draw_icon(size).save(here / ("icon%d.png" % size))
    print("icons written to %s" % here)
