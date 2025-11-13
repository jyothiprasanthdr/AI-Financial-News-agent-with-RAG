"""AI Qdrant client utilities.

Provides functions to initialize the Qdrant client, ensure the collection
exists, and seed it from a JSON file if necessary.
"""

import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "news_embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
SEED_FILE = os.getenv("QDRANT_SEED_FILE", "eb9f97b1-74a3-4299-ab60-293131883956.json")

# ─────────────────────────────────────────────
# Initialize
# ─────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)
VECTOR_SIZE = embedder.get_sentence_embedding_dimension()
qdrant = QdrantClient(url=QDRANT_URL)


def ensure_collection_exists():
    """Check and create collection if missing."""
    try:
        collections = qdrant.get_collections().collections
        existing = [c.name for c in collections]
        if COLLECTION_NAME not in existing:
            print(f"Qdrant: creating collection '{COLLECTION_NAME}' ({VECTOR_SIZE}-dim, cosine metric)")
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE),
            )
            print(f"Qdrant: created collection '{COLLECTION_NAME}'.")
        else:
            print(f"Qdrant: collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        print(f"Qdrant: failed to verify collection: {e}")


def seed_from_json():
    """Seed the collection with JSON data if it's empty."""
    try:
        count = qdrant.count(COLLECTION_NAME).count
        if count > 0:
            print(f"Qdrant: '{COLLECTION_NAME}' already has {count} points.")
            return

        if not os.path.exists(SEED_FILE):
            print(f"Qdrant: no seed file found at {SEED_FILE}; skipping seeding")
            return

        with open(SEED_FILE, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("Qdrant: invalid JSON format (expected list of documents)")
            return

        print(f"Qdrant: seeding {len(data)} documents into '{COLLECTION_NAME}'")
        points = []
        for i, doc in enumerate(data):
            text = f"{doc.get('title', '')} {doc.get('full_text', '')}"
            vector = embedder.encode(text).tolist()
            points.append(
                rest.PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "title": doc.get("title", ""),
                        "ticker": doc.get("ticker", ""),
                        "full_text": doc.get("full_text", ""),
                    },
                )
            )

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Qdrant: inserted {len(points)} documents")
    except Exception as e:
        print(f"Qdrant: seeding failed: {e}")


# ─────────────────────────────────────────────
# Bootstrap routine (called automatically)
# ─────────────────────────────────────────────
def initialize_qdrant():
    """Run Qdrant initialization steps and return the client."""
    ensure_collection_exists()
    seed_from_json()
    return qdrant


qdrant = initialize_qdrant()
