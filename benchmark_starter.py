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
    
# ==========================================
# Student B: Model/Algorithm Integration
# ==========================================
def load_learned_codec(model_name="bmshj2018-hyperprior", quality=3):
    """Loads a pretrained model from CompressAI."""
    try:
        from compressai.zoo import image_models
        model = image_models[model_name](quality=quality, pretrained=True)
        model.eval()
        return model
    except ImportError:
        print("[Error] CompressAI not installed. Run: pip install compressai")
        exit(1)

def learned_codec_inference(model, img):
    """Runs inference and returns decoded image, bpp, and encode time."""
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    start_time = time.time()
    with torch.no_grad():
        out = model(x)
    encode_time = time.time() - start_time
    
    bpp = out["bpp_loss"].item()
    
    # Reconstruct image
    decoded_x = out["x_hat"].clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    decoded_img = Image.fromarray((decoded_x * 255.0).astype(np.uint8))
    
    return decoded_img, bpp, encode_time