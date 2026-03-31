from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import shutil, os, sys
sys.path.insert(0, 'C:/btp')
from matcher_siglip import LostAndFoundMatcher

app = FastAPI()
os.makedirs("C:/btp/found_items", exist_ok=True)
os.makedirs("C:/btp/lost_items", exist_ok=True)
os.makedirs("C:/btp/static", exist_ok=True)

matcher = LostAndFoundMatcher()
matcher.load()

app.mount("/static", StaticFiles(directory="C:/btp/static"), name="static")
app.mount("/found_items", StaticFiles(directory="C:/btp/found_items"), name="found_items")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("C:/btp/static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/report-found")
async def report_found(
    file: UploadFile = File(...),
    location: str = Form(...),
    contact: str = Form(...),
    description: str = Form(""),
    category: str = Form("")
):
    save_path = f"C:/btp/found_items/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    matcher.add_found_item(save_path, location, contact, description, category)
    matcher.save()
    return JSONResponse({"message": "Found item reported!", "filename": file.filename})

@app.post("/search-lost")
async def search_lost(
    file: UploadFile = File(None),
    text_query: str = Form("")
):
    if file and file.filename:
        save_path = f"C:/btp/lost_items/{file.filename}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        results = matcher.find_matches_by_image(save_path)
    elif text_query:
        results = matcher.find_matches_by_text(text_query)
    else:
        return JSONResponse({"error": "Provide image or text"})
    return JSONResponse({"matches": results})

@app.get("/all-found")
def all_found():
    return JSONResponse({"items": matcher.metadata, "total": len(matcher.metadata)})