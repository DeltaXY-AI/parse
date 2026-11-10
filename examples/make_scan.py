"""Make examples/scanned_sample.pdf: the sample page as a grey, noisy, slightly skewed image-only scan."""
from pathlib import Path

import numpy as np
import pymupdf

here = Path(__file__).parent

# 1. the sample page, rotated 1.5 degrees as from a flatbed scanner
skewed = pymupdf.open()
page = skewed.new_page(width=595, height=842)
with pymupdf.open(str(here / "sample.pdf")) as src:
    page.show_pdf_page(page.rect, src, 0, rotate=1.5)
pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)

# 2. grey paper and sensor noise
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).astype(np.float32)
img = img * 0.9 + 20 + np.random.default_rng(7).normal(0, 12, img.shape)
noisy = pymupdf.Pixmap(pymupdf.csGRAY, pix.width, pix.height,
                       np.clip(img, 0, 255).astype(np.uint8).tobytes(), False)

# 3. an image-only PDF (no text layer), JPEG like most scanners
scan = pymupdf.open()
scan.new_page(width=595, height=842).insert_image(pymupdf.Rect(0, 0, 595, 842),
                                                  stream=noisy.tobytes("jpeg", jpg_quality=70))
scan.save(str(here / "scanned_sample.pdf"), deflate=True)
print("wrote", here / "scanned_sample.pdf")
