# 🔍 Smart Campus Lost & Found System

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
| 🔐 **Role-based access** | Separate dashboards for item owners, finders, security staff, and admins |

**Live Demo:** [amaranth-alex-36.tiiny.site](https://animated-bonbon-8f23bc.netlify.app/)

> First multimodal (image + text) campus lost-and-found system deployed at a real academic institution in India.

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
│         HTML5 / CSS3 / Vanilla JavaScript            │
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

**Deployment:** Docker container on HuggingFace Spaces (backend) 

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| ML Backbone | SigLIP 2 (HuggingFace Transformers) | Best accuracy/speed trade-off; multilingual |
| Vector Search | FAISS `IndexFlatIP` | Industry-standard; exact search at campus scale |
| Backend | FastAPI (Python) | Native ML ecosystem; async; auto-generated docs |
| Frontend | HTML5 / CSS3 / JS | Universally accessible; no framework overhead |
| Containerization | Docker | Reproducible deployment |
| Hosting | HuggingFace Spaces + Tiiny Host | Free, public 24/7 URLs |

---

## 📁 Project Structure

```
campus-lost-found/
│
├── main.py                  # FastAPI app — all API endpoints
├── matcher_siglip.py        # Core ML engine: SigLIP embeddings + FAISS search
├── experiment.py            # Model evaluation / benchmarking scripts
│
├── static/
│   └── index.html           # Full frontend (single-page app)
│
├── found_items/             # Uploaded found-item images
├── lost_items/              # Uploaded lost-item query images
├── dataset/                 # Evaluation dataset (20 categories)
│
├── found_items.index        # Serialized FAISS index
├── metadata.json            # Found-item metadata store
├── results.json             # Evaluation results
│
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
└── Smart_Lost_and_Found_Report.pdf  # Full BTP report
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional but recommended; CPU fallback supported)
- 4GB+ VRAM for GPU inference

### Installation

```bash
# Clone the repo
git clone https://github.com/shivam-de-bug/campus-lost-found.git
cd campus-lost-found

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
```

The API will be live at `http://localhost:8000`. Open `http://localhost:8000` in your browser to access the web UI.

### Docker

```bash
docker build -t campus-lost-found .
docker run -p 8000:8000 campus-lost-found
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI (index.html) |
| `/report-found` | POST | Report a found item (image + location + contact) |
| `/search-lost` | POST | Search by image file OR text description |
| `/all-found` | GET | List all found items in the database |

**Example — text search:**
```bash
curl -X POST http://localhost:8000/search-lost \
  -F "text_query=black backpack with red straps"
```

**Example — image search:**
```bash
curl -X POST http://localhost:8000/search-lost \
  -F "file=@my_lost_item.jpg"
```

---

## 📊 Key Results

- **97% Precision@1** on 20-category campus dataset (SigLIP, Sem 1)
- **50% P@1 / 91.67% Recall@5** on stricter leave-one-out eval (SigLIP 2, Sem 2)
- **22ms** average inference latency (GPU) — well under 500ms real-time threshold
- **79× faster** than BLIP-2 with only 1% lower Precision@1
- Validated through multi-stakeholder user surveys (students, faculty, security staff)

---

## 🔬 Research Contributions

1. **Multi-stakeholder user study** validating the inadequacy of email-based lost-and-found workflows in academic campuses
2. **Systematic empirical benchmark** of OpenCLIP, SigLIP, DINOv2, and BLIP-2 on a purpose-built campus item retrieval dataset across 5 metrics
3. **Sem 2 model upgrade study** comparing SigLIP vs SigLIP 2 with controlled evaluation
4. **Working production deployment** with public URL, Docker support, and RESTful API

Full report: [`Smart_Lost_and_Found_Report.pdf`](./Smart_Lost_and_Found_Report%20.pdf)

---

## 🗺️ Roadmap / Future Work

- [ ] Upgrade to **SigLIP 2** in production codebase (evaluation complete, integration pending)
- [ ] Migrate from JSON + FAISS flat file to **Qdrant** vector database (real-time insertions without index rebuild)
- [ ] Add **user authentication** and item claim verification workflow
- [ ] Build **admin dashboard** with match analytics and system health metrics
- [ ] Build **security staff dashboard** for physical inventory management
- [ ] **Email/push notifications** when a match is found for a reported lost item
- [ ] **Mobile-responsive UI** improvements
- [ ] **Fine-tune SigLIP 2** on a larger campus-specific dataset for higher domain accuracy

---

## 👥 Team

| Name | Roll No | Institution |
|---|---|---|
| Shivam | 2023504 | IIIT Delhi |
| Utkarsh Mishra | 2023571 | IIIT Delhi |

**Advisor:** Dr. Anuj Grover, Dept. of Electronics and Communications Engineering, IIITD

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 📚 References

- Zhai et al. (2023). [SigLIP: Sigmoid Loss for Language-Image Pre-Training](https://arxiv.org/abs/2303.15343)
- Tschannen et al. (2025). [SigLIP 2: Multilingual Vision-Language Encoders](https://arxiv.org/abs/2502.14786)
- Johnson et al. (2019). [FAISS: Billion-scale similarity search](https://github.com/facebookresearch/faiss)
- Radford et al. (2021). [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
