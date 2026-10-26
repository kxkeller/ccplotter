"""Generate input.png: a small synthetic drawing made only with Pillow primitives.

Nothing here is derived from a photograph or third-party artwork. The text is
drawn with the scalable default font that ships inside Pillow. Output is
deterministic for a given Pillow version.

    python make_input.py            # writes input.png next to this script
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 900, 1200
img = Image.new("L", (W, H), 255)
d = ImageDraw.Draw(img)

# Horizontal gradient bar (smooth tone, exercises the fill passes)
for x in range(60, 840):
    v = int(235 * (1 - (x - 60) / 780))
    d.line([(x, 70), (x, 190)], fill=v)

# Solid shapes
d.ellipse([90, 250, 330, 490], fill=20)
d.rectangle([380, 250, 620, 490], outline=0, width=10)
d.polygon([(720, 250), (840, 490), (600 + 120, 490)], fill=90)

# Concentric rings (line work)
for i, r in enumerate(range(150, 10, -22)):
    d.ellipse([450 - r, 700 - r, 450 + r, 700 + r], outline=0, width=6 if i % 2 else 3)

# Diagonal hatching inside a box (drawn on a tile, then pasted, so lines are clipped)
tile = Image.new("L", (240, 280), 255)
td = ImageDraw.Draw(tile)
for k in range(-280, 240, 18):
    td.line([(k, 0), (k + 280, 280)], fill=0, width=2)
img.paste(tile, (60, 560))
d.rectangle([60, 560, 300, 840], outline=0, width=4)

# Vertical tone ramp made of bars of increasing darkness
for i in range(8):
    d.rectangle([640 + i * 30, 560, 640 + i * 30 + 24, 840], fill=int(240 - i * 32))

# Text
big = ImageFont.load_default(size=96)
small = ImageFont.load_default(size=40)
d.text((60, 900), "ccPlotter", fill=0, font=big)
d.text((60, 1030), "synthetic test drawing", fill=40, font=small)
d.text((60, 1090), "shapes, rings, hatching, tone, text", fill=90, font=small)

img.save(Path(__file__).with_name("input.png"), optimize=True)
print("wrote input.png", img.size)
