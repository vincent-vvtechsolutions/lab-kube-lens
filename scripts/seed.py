import json
import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.services.ai_service import ai_service

DATA_PATH = "data/knowledge_base.json"


def run_seed():
    # 1. Charger les données
    with open(DATA_PATH) as f:
        knowledge_data = json.load(f)

    # 2. Initialiser la collection (Dim 768 pour nomic-embed)
    ai_service.qdrant.recreate_collection(
        collection_name="sre_knowledge",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # 3. Processus d'ingestion
    points = []
    print(f"Starting ingestion of {len(knowledge_data)} procedures...")

    for item in knowledge_data:
        vector = ai_service.get_embedding(
            f"RUNTIME_TAG: {item['runtime']} | TECHNOLOGY: {item['runtime'].upper()} SYSTEM | PROCEDURE: {item['content']}",
            is_query=False,
        )
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=item))

    ai_service.qdrant.upsert(collection_name="sre_knowledge", points=points)
    print("✅ Ingestion complete. Qdrant is ready.")


if __name__ == "__main__":
    run_seed()
