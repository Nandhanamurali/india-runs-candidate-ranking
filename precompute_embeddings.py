"""
precompute_embeddings.py
One-time step: computes a semantic similarity score between the JD and
every candidate, saves results to semantic_scores.json.
This step is NOT time-limited -- only the final ranking script is.
"""

import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

JD_QUERY = """
Senior AI Engineer. Owns ranking, retrieval, and matching systems for a recruiting platform.
Must have production experience with embeddings-based retrieval systems
(sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users,
handling embedding drift, index refresh, and retrieval-quality regression.
Must have production experience with vector databases or hybrid search
infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
Strong Python and code quality. Hands-on experience designing evaluation
frameworks for ranking systems: NDCG, MRR, MAP, offline-to-online correlation,
A/B testing.
Ideal candidate has 6-8 years total experience, 4-5 years in applied ML/AI
roles at product companies (not pure consulting/services). Has shipped at
least one end-to-end ranking, search, or recommendation system to real users
at meaningful scale. Has strong opinions on hybrid vs dense retrieval,
offline vs online evaluation, and when to fine-tune vs prompt an LLM.
"""


def candidate_text(c):
    parts = [
        c["profile"].get("headline", ""),
        c["profile"].get("summary", ""),
    ]
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    skill_names = [s["name"] for s in c.get("skills", [])]
    parts.append("Skills: " + ", ".join(skill_names))
    return " ".join(parts)


def main():
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding job description...")
    jd_embedding = model.encode(JD_QUERY, normalize_embeddings=True)

    print("Reading candidates...")
    candidate_ids = []
    texts = []
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            candidate_ids.append(c["candidate_id"])
            texts.append(candidate_text(c))

    print(f"Encoding {len(texts)} candidates... this is the slow part, please wait.")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"Done in {time.time() - start:.1f} seconds")

    print("Computing similarity scores...")
    similarities = embeddings @ jd_embedding  # cosine similarity since both normalized

    print("Saving results...")
    results = {cid: float(score) for cid, score in zip(candidate_ids, similarities)}
    with open("semantic_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f)

    print("Saved semantic_scores.json")
    print("\nTop 10 by semantic similarity alone:")
    top = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]
    for cid, score in top:
        print(cid, round(score, 4))


if __name__ == "__main__":
    main()