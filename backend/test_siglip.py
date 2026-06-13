import torch
import torch.nn.functional as F
from transformers import SiglipProcessor, SiglipModel
from PIL import Image
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()

img_path = r"c:\Users\mishr\Documents\Coding\BTP\campus-lost-found\found_items\Screenshot 2026-03-31 165421.png"
if os.path.exists(img_path):
    img = Image.open(img_path).convert("RGB")
    
    # Image embedding
    inputs_img = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out_img = model.get_image_features(**inputs_img)
        emb_img = out_img.pooler_output if hasattr(out_img, 'pooler_output') else out_img
        emb_img = F.normalize(emb_img, dim=-1)

    texts = ["a screenshot of a website", "black leather wallet", "blue jbl earphones", "keys", "a photo"]
    
    for text in texts:
        inputs_text = processor(text=[text], padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out_text = model.get_text_features(**inputs_text)
            emb_text = out_text.pooler_output if hasattr(out_text, 'pooler_output') else out_text
            emb_text = F.normalize(emb_text, dim=-1)
        
        sim = (emb_text @ emb_img.T).item()
        print(f"Text: '{text}', Similarity: {sim}")
else:
    print("Image not found")
