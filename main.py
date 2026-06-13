from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, sys, time, logging
import uvicorn

# Use the current directory as base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from matcher_siglip import LostAndFoundMatcher
from storage_manager import StorageManager

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for now, can restrict to Vercel URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist locally
FOUND_DIR = os.path.join(BASE_DIR, "found_items")
LOST_DIR = os.path.join(BASE_DIR, "lost_items")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(FOUND_DIR, exist_ok=True)
os.makedirs(LOST_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

storage = StorageManager()

# Try to pull latest data from Hugging Face Dataset on startup
storage.download_data(BASE_DIR)

matcher = LostAndFoundMatcher()
matcher.load(path=BASE_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/found_items", StaticFiles(directory=FOUND_DIR), name="found_items")

@app.on_event("startup")
async def startup_event():
    print("="*50)
    print("404 FOUND BACKEND STARTING")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"FOUND_DIR: {FOUND_DIR}")
    print(f"Items currently in database: {len(matcher.metadata)}")
    print("="*50)

@app.get("/", response_class=HTMLResponse)
def home():
    # Serve the new index.html from the root directory
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()

@app.post("/report-found")
async def report_found(
    file: UploadFile = File(...),
    location: str = Form(...),
    contact: str = Form(...),
    description: str = Form(""),
    category: str = Form("")
):
    try:
        # Create directory if it doesn't exist
        os.makedirs(FOUND_DIR, exist_ok=True)
        
        # Save file with a safe name
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        save_path = os.path.join(FOUND_DIR, safe_filename)
        
        print(f"DEBUG: Saving reported item to: {os.path.abspath(save_path)}")
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        file_size = os.path.getsize(save_path)
        print(f"DEBUG: File saved successfully. Size: {file_size} bytes")
        
        print(f"DEBUG: Adding to matcher...")
        matcher.add_found_item(save_path, location, contact, description, category)
        matcher.save(path=BASE_DIR)
        
        # Persist to Hugging Face Hub
        storage.upload_data(BASE_DIR)
        
        print(f"DEBUG: Report success")
        return {"status": "success", "filename": safe_filename}
    except Exception as e:
        print(f"CRITICAL ERROR in report_found: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/search-lost")
async def search_lost(
    file: UploadFile = File(None),
    text_query: str = Form("")
):
    try:
        os.makedirs(LOST_DIR, exist_ok=True)
        results = []
        
        if text_query and text_query.strip():
            print(f"DEBUG: Searching by text: {text_query}")
            results = matcher.find_matches_by_text(text_query)
        elif file and file.filename:
            save_path = os.path.join(LOST_DIR, file.filename)
            print(f"DEBUG: Saving search query image to {save_path}")
            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            results = matcher.find_matches_by_image(save_path)
        else:
            return JSONResponse(status_code=400, content={"error": "Provide image or text"})
        
        print(f"DEBUG: Found {len(results)} matches")
        return {"matches": results}
    except Exception as e:
        print(f"CRITICAL ERROR in search_lost: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/all-found")
def all_found():
    return JSONResponse({"items": matcher.metadata, "total": len(matcher.metadata)})

if __name__ == "__main__":
    # Run on 7860 for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)