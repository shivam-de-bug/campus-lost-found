import asyncio
from fastapi.testclient import TestClient
from main import app
import os
import time

client = TestClient(app)

def test_multiple_uploads():
    img_path = r"c:\Users\mishr\Documents\Coding\BTP\campus-lost-found\found_items\Screenshot 2026-03-31 165421.png"
    if not os.path.exists(img_path):
        print("Image not found")
        return
        
    for i in range(3):
        with open(img_path, "rb") as f:
            resp = client.post("/report-found", data={
                "location": f"loc_{i}",
                "contact": "123",
                "description": f"item {i}",
                "category": "Others"
            }, files={"file": ("test.png", f, "image/png")})
            print(f"Upload {i}:", resp.status_code)
            
    # Now search
    resp = client.post("/search-lost", data={"text_query": "item"})
    matches = resp.json().get("matches", [])
    print(f"Found {len(matches)} matches")
    for m in matches:
        print(m["description"], m["similarity"])

test_multiple_uploads()
