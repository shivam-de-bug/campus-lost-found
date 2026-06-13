# 🔍 404 Found - Smart Campus Lost & Found System

> **A Multimodal Vision-Language Platform for Intelligent Item Retrieval**  

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![SigLIP 2](https://img.shields.io/badge/Model-SigLIP%202-orange?logo=google)](https://huggingface.co/google/siglip2-base-patch16-224)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-blue)](https://github.com/facebookresearch/faiss)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Problem Statement

Academic campuses like IIIT Delhi rely on **email broadcasts** for lost & found — a fundamentally broken workflow. Our user survey of the campus community found:

- **71%** of campus members lost an item in the past year
- **53%** ignore or mark lost-and-found emails as spam  
- Recovery rate through existing channels is extremely low

The root cause is structural: email is a broadcast system applied to a **matching problem**. The solution is not to improve email — it's to replace it with an intelligent, centralized, image-and-text-aware retrieval platform.

---

## 🚀 What This Project Does

The **Smart Lost & Found System** is a full-stack web application powered by a state-of-the-art vision-language model. It enables:

| Feature | Description |
|---|---|
| 📸 **Image-based search** | Upload a photo of your lost item → AI retrieves visually similar found items |
| 💬 **Text-based search** | Type a description like *"black JBL earphones with red cable"* → AI matches it to found items |
| 📋 **Found item reporting** | Finders upload a photo + metadata (location, contact) → auto-indexed and searchable instantly |
| 🎯 **Confidence-aware ranking** | Results ranked by cosine similarity with High / Medium / Low confidence labels |
| 🔐 **Role-based access** | Separate dashboards for item owners (students), security gate staff (handover verification), and admins |

---

## 🧠 AI / ML Architecture

This is the core technical contribution of the project — a systematic benchmark of 4 state-of-the-art vision-language models, followed by deployment of the best-performing one.

### Model Evaluation (Semester 1)

Evaluated on a **purpose-built dataset of 20 campus-relevant item categories** (water bottles, bags, phones, keys, wallets, earphones, laptops, ID cards, etc.) on an NVIDIA RTX 3050 GPU.

| Model | P@1 | P@5 | R@5 | MRR | Speed (ms) | Text Search |
|---|---|---|---|---|---|---|
| OpenCLIP | 94.0% | 62.0% | 97.0% | 95.0% | 52.2 | ✅ |
| **SigLIP** ✅ | **97.0%** | **65.6%** | **98.0%** | **97.5%** | **89.9** | ✅ |
| DINOv2 | 91.0% | 70.8% | 100.0% | 94.75% | 119.4 | ❌ |
| BLIP-2 | 98.0% | 73.0% | 100.0% | 98.75% | 7167.7 | ✅ |

**SigLIP selected** — 97% Precision@1 with only 89.9ms latency. BLIP-2 achieves marginally higher accuracy but is **79× slower**, making it impractical for real-time use. DINOv2 has no text encoder, ruling out text queries.

### Model Upgrade 

Evaluated **SigLIP 2** (Google DeepMind, Feb 2025) — same ViT-B/16 architecture, improved training recipe (captioning + self-distillation + masked prediction).

| Model | P@1 | R@5 | MRR | Speed (ms) | Text |
|---|---|---|---|---|---|
| SigLIP (baseline) | 33.33% | 83.33% | 53.68% | 21.6 | English |
| **SigLIP 2** ✅ | **50.00%** | **91.67%** | **67.86%** | **22.0** | **Multilingual** |

**SigLIP 2 selected for Semester 2** — 50% relative improvement in P@1, nearly identical inference speed, and multilingual text support at no extra cost.

### How the AI Pipeline Works

```
User Query (image or text)
        ↓
SigLIP 2 Encoder (google/siglip2-base-patch16-224)
        ↓
768-dimensional L2-normalized embedding
        ↓
FAISS IndexFlatIP (exact cosine similarity search)
        ↓
Top-K matches with confidence labels (High / Medium / Low)
```

- **Image query** → `SiglipModel.get_image_features()` 
- **Text query** → `SiglipModel.get_text_features()` — same embedding space, enabling cross-modal retrieval
- **FAISS** stores all found-item embeddings; new items are indexed in real time on upload

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Frontend Layer                     │
│         React.js / Vite / Vanilla CSS               │
│   Search Lost │ Report Found │ All Items │ Admin     │
└──────────────────────┬──────────────────────────────┘
                       │ REST API calls
┌──────────────────────▼──────────────────────────────┐
│                 Backend API Layer                    │
│                FastAPI (Python)                      │
│  POST /report-found │ POST /search-lost │ GET /all-found  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  ML Engine Layer                     │
│   SigLIP 2  →  768-dim embeddings  →  FAISS index   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Storage Layer                       │
│     found_items/ (images) │ metadata.json │ .index   │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
campus-lost-found/
├── backend/                          # Python FastAPI backend
│   ├── main.py                       # FastAPI application
│   ├── matcher_siglip.py            # SigLIP2 AI matching engine
│   ├── storage_manager.py           # Cloud storage integration
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Docker configuration
│   ├── dataset/                     # Benchmark dataset
│   ├── found_items/                 # Logged found item images
│   ├── lost_items/                  # Lost item search query images
│   ├── metadata.json                # Item metadata store
│   └── users.json                   # User account database (flat file)
│
├── frontend/                         # React.js SPA frontend
│   ├── index.html                   # HTML template & CDN configurations
│   ├── package.json                 # Node dependencies
│   ├── vite.config.js               # Vite environment config & dev proxies
│   ├── src/
│   │   ├── main.jsx                 # Entry point
│   │   ├── App.jsx                  # Main router definitions
│   │   ├── App.css                  # Core CSS design variables and overrides
│   │   ├── index.css                # Global body parameters
│   │   ├── api/
│   │   │   └── apiClient.js         # API fetch interceptor and wrappers
│   │   ├── components/
│   │   │   ├── Header.jsx           # Global sticky top header
│   │   │   ├── ProtectedRoute.jsx   # Role checks safety gate
│   │   │   ├── ItemCard.jsx         # Card component with image fit
│   │   │   ├── ReportForm.jsx       # Intake logging form
│   │   │   ├── SearchBar.jsx        # Multimodal AI search bar
│   │   │   └── Sidebar.jsx          # Bottom bar navigation for Admin
│   │   └── pages/
│   │       ├── Home.jsx             # Student dashboard (floating report + nav)
│   │       ├── Guard.jsx            # Security staff portal (intake + claims)
│   │       ├── Admin.jsx            # Control panel (resolution + analytics)
│   │       └── Login.jsx            # Glassmorphic authentication page
│   └── dist/                        # Compiled production assets
│
├── README.md                         # This file
├── .gitignore                        # Git ignore specifications
└── Smart_Lost_and_Found_Report.pdf  # Project BTP thesis report
```

---

## ⚡ Quick Start

### 1. Backend Setup

```bash
# Navigate to backend
cd backend

# Install python packages
pip install -r requirements.txt

# Start the uvicorn API server
python main.py
```
The FastAPI server will boot up and host on `http://localhost:7860`.

### 2. Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install Node modules
npm install

# Run the local development server (Vite)
npm run dev
```
Vite will boot the dev server on `http://localhost:5173`. Proxies configured in `vite.config.js` will automatically redirect API calls to the Python backend on port `7860`.

To build the static assets for production:
```bash
npm run build
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| AI Matching Accuracy | 92% (Precision@1) |
| Search Response Time | <100ms |
| Speed Advantage vs BLIP-2 | 79× faster |
| Supported Categories | 20+ item types |

---

## 👥 Team

**B.Tech Project** — IIIT-Delhi

* **Shivam** (Roll No: 2023504) - IIIT Delhi
* **Utkarsh Mishra** (Roll No: 2023571) - IIIT Delhi

**Advisor:** Dr. Anuj Grover, Dept. of Electronics and Communications Engineering, IIIT-Delhi

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
