import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import faiss
import numpy as np
import os
import json

class LostAndFoundMatcher:
    def __init__(self):
        print("Loading SigLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(self.device).eval()
        self.index = faiss.IndexFlatIP(768)
        self.metadata = []
        print(f"Ready on {self.device}!")

    def get_embedding(self, img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
            # Handle object vs tensor return
            emb = out.pooler_output if hasattr(out, 'pooler_output') else out
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype('float32')

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            # Handle object vs tensor return
            emb = out.pooler_output if hasattr(out, 'pooler_output') else out
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
        return self._format_results(scores[0], indices[0], is_text=False)

    def find_matches_by_text(self, text_query, top_k=3):
        query = self.get_text_embedding(text_query)
        scores, indices = self.index.search(query, k=top_k)
        return self._format_results(scores[0], indices[0], is_text=True)

    def _format_results(self, scores, indices, is_text=False):
        results = []
        # Text-to-image dot products can be much lower, even negative for non-matches.
        threshold = 0.05 if is_text else 0.10
        
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            
            raw_score = float(score)
            
            # Skip items below the "sanity" threshold to avoid random matches
            if raw_score < threshold:
                continue
                
            item = self.metadata[idx]
            
            # Scale the score for display (Human-friendly %)
            if is_text:
                # Text scores usually range from 0.0 to 0.2
                display_score = min(0.99, max(0.10, (raw_score * 5.0) + 0.10))
                confidence = "High" if raw_score > 0.10 else "Medium" if raw_score > 0.05 else "Low"
            else:
                # Image scores usually range from 0.1 to 0.4
                display_score = min(0.99, max(0.10, (raw_score * 3.5) + 0.10))
                confidence = "High" if raw_score > 0.20 else "Medium" if raw_score > 0.14 else "Low"
            
            results.append({
                "rank": len(results)+1,
                "filename": item["filename"],
                "location": item["location"],
                "contact": item["contact"],
                "description": item["description"],
                "similarity": round(display_score, 4),
                "raw_score": round(raw_score, 4),
                "confidence": confidence
            })
        return results

    def save(self, path="./"):
        faiss.write_index(self.index, os.path.join(path, "found_items.index"))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)
        print("Saved!")

    def load(self, path="./"):
        if os.path.exists(os.path.join(path, "found_items.index")):
            self.index = faiss.read_index(os.path.join(path, "found_items.index"))
            with open(os.path.join(path, "metadata.json")) as f:
                self.metadata = json.load(f)
            print(f"Loaded {self.index.ntotal} items")