# ─── config.py ────────────────────────────────────────────────────────────────
# Central configuration for the Smart Lost & Found System.
# Change values here — no need to touch any other file.

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "google/siglip2-base-patch16-224"   # Semester 2 upgraded model
EMBEDDING_DIM = 768                               # SigLIP / SigLIP2 embedding size

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K = 3                      # Number of matches to return

# Minimum cosine similarity to show a result (filters out junk matches)
TEXT_THRESHOLD  = 0.05         # Text-to-image scores are naturally lower
IMAGE_THRESHOLD = 0.10         # Image-to-image scores are higher

# Confidence label thresholds (raw cosine similarity)
IMAGE_HIGH_CONFIDENCE   = 0.20
IMAGE_MEDIUM_CONFIDENCE = 0.14
TEXT_HIGH_CONFIDENCE    = 0.10
TEXT_MEDIUM_CONFIDENCE  = 0.05

# ── Storage ────────────────────────────────────────────────────────────────────
FOUND_ITEMS_DIR  = "found_items"
LOST_ITEMS_DIR   = "lost_items"
STATIC_DIR       = "static"
FAISS_INDEX_FILE = "found_items.index"
METADATA_FILE    = "metadata.json"

# ── Server ─────────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 7860
