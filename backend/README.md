# 404 Found - Backend

AI-powered Lost & Found system backend using FastAPI and SigLIP2 vision-language model.

## 📋 Project Structure

```
backend/
├── main.py                 # FastAPI application
├── matcher_siglip.py      # AI matching engine using SigLIP2
├── storage_manager.py     # Hugging Face storage integration
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── dataset/               # Dataset files
├── found_items/           # Storage for found item images
├── lost_items/            # Storage for lost item images
└── metadata.json          # Item metadata
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda
- CUDA (optional, for GPU acceleration)

### Installation

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Set up environment (optional):**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Run the backend:**
```bash
python main.py
```

The server will start at `http://localhost:7860`

## 🔌 API Endpoints

### Main Routes

- `GET /` - Serves frontend index.html
- `GET /admin` - Serves admin dashboard
- `GET /guard` - Serves guard dashboard

### Lost & Found Routes

- `POST /report-found` - Report a found item
  - Parameters: `file` (image), `location`, `contact`, `description`, `category`
  
- `POST /search-lost` - Search for matching items
  - Parameters: `file` (image) OR `text_query`
  
- `GET /all-found` - Get all found items

## 🔧 Configuration

### Environment Variables

```
API_URL=http://localhost:7860
HUGGINGFACE_TOKEN=your_token_here
```

## 📦 Deployment

### Docker

```bash
cd backend
docker build -t 404-found-backend .
docker run -p 7860:7860 404-found-backend
```

### Hugging Face Spaces

1. Push code to repository
2. Create new Space on Hugging Face
3. Connect to this repository
4. Deploy automatically

### Traditional Server

```bash
# Using gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:7860
```

## 🤖 AI Model

Uses **SigLIP2** - a vision-language model from Google for:
- Image-to-image similarity matching
- Text-based search on images
- Confidence scoring (High/Medium/Low)

## 📊 Features

- ✅ Image upload and storage
- ✅ SigLIP2 AI matching (92% accuracy)
- ✅ FAISS similarity search (<100ms)
- ✅ Hugging Face Hub integration
- ✅ Real-time item database
- ✅ Multi-category support

## 🧪 Testing

```bash
# Run tests
python test_fastapi.py
python test_siglip.py
python test_multi.py
```

## 📝 License

See ../README.md for more information

## 🤝 Support

For issues and questions, refer to the main project README.
