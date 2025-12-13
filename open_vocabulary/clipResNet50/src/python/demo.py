import torch, clip, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# --- CLIP (CPU) ---
device = "cpu"
model, preprocess = clip.load("RN50", device=device)

# --- inputs ---
img_path = "../../assets/cat.jpg"
class_names = ["a photo of a diagram", "a photo of a dog", "a photo of a cat"]

# --- predict label ---
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
text = clip.tokenize(class_names).to(device)
with torch.no_grad():
    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
label = class_names[int(np.argmax(probs, axis=1)[0])]

# --- write label on ORIGINAL image (top-right, light yellow) ---
im = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
draw = ImageDraw.Draw(im)

try:
    font = ImageFont.truetype("DejaVuSans.ttf", max(16, int(0.06 * max(im.size))))
except Exception:
    font = ImageFont.load_default()

# Measure text with Pillow's textbbox (Pillow >= 10)
l, t, r, b = draw.textbbox((0, 0), label, font=font)
tw, th = r - l, b - t
pad = 16
x, y = im.width - tw - pad, pad

draw.text((x, y), label, font=font, fill=(255, 255, 224))  # light yellow

# --- save ---
im.save("../../assets/labeled_image.jpg", quality=95)
print("Predicted:", label)
print("Saved: labeled_image.jpg")
