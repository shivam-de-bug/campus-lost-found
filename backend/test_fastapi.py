import asyncio
from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

def test_home():
    response = client.get("/")
    print("Home:", response.status_code)

def test_search_lost():
    # Test text search
    response = client.post("/search-lost", data={"text_query": "black wallet"})
    print("Search text:", response.status_code, response.json())
    
    # Test image search
    img_path = r"c:\Users\mishr\Documents\Coding\BTP\campus-lost-found\found_items\Screenshot 2026-03-31 165421.png"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            response = client.post("/search-lost", files={"file": ("test.png", f, "image/png")})
        print("Search image:", response.status_code, response.json())
        
    # Test both
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            response = client.post("/search-lost", data={"text_query": "screenshot"}, files={"file": ("test.png", f, "image/png")})
        print("Search both:", response.status_code, response.json())

test_home()
test_search_lost()
