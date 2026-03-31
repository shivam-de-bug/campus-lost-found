import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel
import faiss
import numpy as np
import os
import json

class LostAndFoundMatcher:
    def __init__(self):
        print("Loading SigLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(self.device).eval()
        self.index = faiss.IndexFlatIP(768)
        self.metadata = []
        print(f"Ready on {self.device}!")

    def get_embedding(self, img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
            emb = out.pooler_output if hasattr(out, 'pooler_output') else out.last_hidden_state[:,0,:]
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype('float32')

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype('float32')

    def add_found_item(self, img_path, location, contact, description="", category=""):
        emb = self.get_embedding(img_path)
        self.index.add(emb)
        self.metadata.append({
            "filename": os.path.basename(img_path),
            "location": location,
            "contact": contact,
            "description": description,
            "category": category
        })
        print(f"Added: {os.path.basename(img_path)}")

    def find_matches_by_image(self, lost_img_path, top_k=3):
        query = self.get_embedding(lost_img_path)
        scores, indices = self.index.search(query, k=top_k)
        return self._format_results(scores[0], indices[0])

    def find_matches_by_text(self, text_query, top_k=3):
        query = self.get_text_embedding(text_query)
        scores, indices = self.index.search(query, k=top_k)
        return self._format_results(scores[0], indices[0])

    def _format_results(self, scores, indices):
        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            item = self.metadata[idx]
            results.append({
                "rank": len(results)+1,
                "filename": item["filename"],
                "location": item["location"],
                "contact": item["contact"],
                "description": item["description"],
                "similarity": round(float(score), 4),
                "confidence": "High" if score > 0.85 else "Medium" if score > 0.70 else "Low"
            })
        return results

    def save(self, path="C:/btp/"):
        faiss.write_index(self.index, path+"found_items.index")
        with open(path+"metadata.json", "w") as f:
            json.dump(self.metadata, f)
        print("Saved!")

    def load(self, path="C:/btp/"):
        if os.path.exists(path+"found_items.index"):
            self.index = faiss.read_index(path+"found_items.index")
            with open(path+"metadata.json") as f:
                self.metadata = json.load(f)
            print(f"Loaded {self.index.ntotal} items")