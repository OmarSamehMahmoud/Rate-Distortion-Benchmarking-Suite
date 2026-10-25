import os
import torch
import numpy as np
from PIL import Image
import urllib.request
import time
import io

# ==========================================
# Student A: Data & Baseline Setup
# ==========================================
def download_kodak_sample():
    """Downloads a sample Kodak image for feasibility testing."""
    url = "http://r0k.us/graphics/kodak/kodak/kodim01.png"
    filename = "kodim01.png"
    if not os.path.exists(filename):
        print(f"[Data] Downloading sample image from Kodak dataset...")
        urllib.request.urlretrieve(url, filename)
        print("[Data] Download complete.")
    return filename

def load_and_prep_image(image_path):
    """Loads image and ensures dimensions are multiples of 64 for CompressAI."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    # Center crop to avoid padding artifacts in learned codecs
    new_width = (width // 64) * 64
    new_height = (height // 64) * 64
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))
    return img

def classical_jpeg_encode_decode(img, quality=75):
    """Baseline classical JPEG compression."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    file_size = buffer.tell()
    buffer.seek(0)
    decoded_img = Image.open(buffer).convert("RGB")
    bpp = (file_size * 8) / (img.width * img.height)
    return decoded_img, bpp