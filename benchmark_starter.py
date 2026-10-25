import os
import torch
import numpy as np
from PIL import Image
import urllib.request
import time
import io
import math

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
        model.eval() # Crucial for inference mode
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
    
    # FIX: Calculate bpp from likelihoods (Official CompressAI evaluation method)
    # "bpp_loss" is only for training. During inference, we calculate from "likelihoods".
    num_pixels = x.size(2) * x.size(3) # Height * Width
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out["likelihoods"].values()
    )
    
    # Reconstruct image
    decoded_x = out["x_hat"].clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    decoded_img = Image.fromarray((decoded_x * 255.0).astype(np.uint8))
    
    return decoded_img, bpp.item(), encode_time

# ==========================================
# Student C: Evaluation Harness
# ==========================================
def calculate_psnr(img1, img2):
    """Simple PSNR calculation."""
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

# ==========================================
# Main Feasibility Test Execution
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("Milestone 1: Feasibility Test & Pipeline Starter")
    print("="*50)
    
    # 1. Data Preparation
    img_path = download_kodak_sample()
    img = load_and_prep_image(img_path)
    print(f"[Preprocessing] Loaded and cropped image to {img.width}x{img.height}")
    
    # 2. Test Classical Baseline
    target_quality = 75
    dec_jpeg, jpeg_bpp = classical_jpeg_encode_decode(img, quality=target_quality)
    psnr_jpeg = calculate_psnr(img, dec_jpeg)
    print("-" * 50)
    print(f"[Classical JPEG] Quality: {target_quality:2d} | BPP: {jpeg_bpp:.4f} | PSNR: {psnr_jpeg:.2f} dB")
    
    # 3. Test Learned Codec
    print("[Learned Codec] Loading bmshj2018-hyperprior (Quality=3)...")
    model = load_learned_codec(model_name="bmshj2018-hyperprior", quality=3)
    dec_learned, learned_bpp, enc_time = learned_codec_inference(model, img)
    psnr_learned = calculate_psnr(img, dec_learned)
    print("-" * 50)
    print(f"[Learned Codec] Model: bmshj2018-hyperprior (Q=3)")
    print(f"                BPP: {learned_bpp:.4f} | PSNR: {psnr_learned:.2f} dB | Enc Time: {enc_time:.4f}s")
    print("="*50)
    print("[SUCCESS] Feasibility test passed. Pipeline is ready for M1 deliverable.")