"""Hand-made photo features from Harris corner detection (the method of DSP Lab 7).

Steps, the same as the lab:
1. grayscale: R*0.299 + G*0.587 + B*0.114
2. image gradients Ix, Iy with the horizontal and vertical gradient filter
3. Ix2, Iy2, Ixy smoothed with a Gaussian window (sigma 2)
4. corner response R = det(M) - k * trace(M)^2 with k = 0.04, mapped so the maximum is 1000
5. corners: local maxima of R (max filter, radius 6) above the threshold 20

The idea for house photos: a detailed exterior (windows, trim, railings,
brick) has many corners and strong edges, a plain wall or an empty street
has few. The photo is cut into a 3 x 3 grid and four numbers are measured in
each cell, so the model also knows roughly where the detail is:
    corners per 10,000 pixels, mean gradient strength, share of edge pixels
    (R below -20), and share of gradient energy that is horizontal.
That gives 9 x 4 = 36 numbers per photo.
"""
import io

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve

ENCODER_NAME = "Harris corner features (DSP Lab 7 method), 3 x 3 grid, 36 numbers"
EMBEDDING_SIZE = 36
WIDTH = 512
SIGMA = 2.0
N_X_SIGMA = 6
ALPHA = 0.04
THRESHOLD = 20.0
RADIUS = 6
GRID = 3

DX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
DY = DX.T


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    half = int(size) // 2
    x, y = np.mgrid[-half:half + 1, -half:half + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


WINDOW = gaussian_kernel(max(1, int(2 * N_X_SIGMA * SIGMA)), SIGMA)


def corner_response(gray: np.ndarray):
    """Ix, Iy and the corner response R (scaled so its maximum is 1000)."""
    ix = fftconvolve(gray, DX, mode="same")
    iy = fftconvolve(gray, DY, mode="same")
    ix2 = fftconvolve(ix * ix, WINDOW, mode="same")
    iy2 = fftconvolve(iy * iy, WINDOW, mode="same")
    ixy = fftconvolve(ix * iy, WINDOW, mode="same")
    r = ix2 * iy2 - ixy * ixy - ALPHA * (ix2 + iy2) ** 2
    peak = r.max()
    if peak > 0:
        r = r * (1000.0 / peak)
    return ix, iy, r


def corners(r: np.ndarray) -> np.ndarray:
    """True where a corner is: a local maximum of R above the threshold, away from the border."""
    found = (r == maximum_filter(r, size=2 * RADIUS + 1)) & (r > THRESHOLD)
    found[:RADIUS, :] = found[-RADIUS:, :] = False
    found[:, :RADIUS] = found[:, -RADIUS:] = False
    return found


def features_from_array(rgb: np.ndarray) -> np.ndarray:
    """36 numbers for one RGB image array (height x width x 3, values 0 to 255)."""
    rgb = rgb.astype(float)
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    ix, iy, r = corner_response(gray)
    is_corner = corners(r)
    magnitude = np.sqrt(ix ** 2 + iy ** 2) / 255.0
    is_edge = r < -THRESHOLD
    h, w = gray.shape
    out = []
    for i in range(GRID):
        for j in range(GRID):
            rows = slice(i * h // GRID, (i + 1) * h // GRID)
            cols = slice(j * w // GRID, (j + 1) * w // GRID)
            area = max(1, (rows.stop - rows.start) * (cols.stop - cols.start))
            ex, ey = float(np.sum(ix[rows, cols] ** 2)), float(np.sum(iy[rows, cols] ** 2))
            out += [
                1e4 * float(is_corner[rows, cols].sum()) / area,
                float(magnitude[rows, cols].mean()),
                float(is_edge[rows, cols].mean()),
                ex / (ex + ey) if ex + ey > 0 else 0.5,
            ]
    return np.array(out, dtype=np.float32)


def embed_images(jpeg_bytes_list) -> np.ndarray:
    rows = []
    for data in jpeg_bytes_list:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        if image.width != WIDTH:
            image = image.resize((WIDTH, max(1, round(image.height * WIDTH / image.width))))
        rows.append(features_from_array(np.asarray(image)))
    return np.array(rows, dtype=np.float32).reshape(-1, EMBEDDING_SIZE)
