from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import uuid

# ─────────────────────────────────────────────
# 1️⃣ Load your dataset (JSON file)
# ─────────────────────────────────────────────
with open("stock_news.json", "r") as f:
    data = json.load(f)

# Flatten structure → list of documents
documents = []
for ticker, articles in data.items():
    for article in articles:
        full_text = article.get("full_text", "").strip()
        if not full_text:
            continue
        documents.append({
            "ticker": ticker,
            "title": article.get("title", ""),
            "link": article.get("link", ""),
            "full_text": full_text
        })

print(f"Loaded {len(documents)} documents for embedding.")

# ─────────────────────────────────────────────
# 2️⃣ Initialize embedding model and Qdrant client
# ─────────────────────────────────────────────
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

qdrant = QdrantClient(url="http://localhost:6333")
collection_name = "news_embeddings"

# ─────────────────────────────────────────────
# 3️⃣ Create / reset collection
# ─────────────────────────────────────────────
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
)
print(f"✅ Collection '{collection_name}' created/reset.")

# ─────────────────────────────────────────────
# 4️⃣ Generate embeddings from full_text
# ─────────────────────────────────────────────
texts = [doc["full_text"] for doc in documents]
print("Creating embeddings... this may take a moment.")
embeddings = model.encode(texts, show_progress_bar=True)

# ─────────────────────────────────────────────
# 5️⃣ Prepare and insert into Qdrant
# ─────────────────────────────────────────────
points = []
for i, doc in enumerate(documents):
    points.append(
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),   # vector = embedding of full_text
            payload={                       # metadata payload
                "ticker": doc["ticker"],
                "title": doc["title"],
                "link": doc["link"],
                "full_text": doc["full_text"]
            }
        )
    )

qdrant.upsert(collection_name=collection_name, points=points)
print(f"✅ Inserted {len(points)} embedded documents into Qdrant.")

