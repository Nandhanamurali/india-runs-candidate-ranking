"""
generate_submission.py
The OFFICIAL submission script. Produces top_100_ranked.csv in the
required format: candidate_id, rank, score, reasoning.
"""

import json
import csv
import time
from features import extract_features
from reasoning import generate_reasoning

with open("semantic_scores.json", "r", encoding="utf-8") as f:
    SEMANTIC_SCORES = json.load(f)


def composite_score(feats: dict, semantic_score: float) -> float:
    score = 50.0

    if feats["flag_non_engineering_title"]:
        score -= 60
    if feats["flag_consulting_only"]:
        score -= 25
    if feats["flag_pure_research_no_production"]:
        score -= 30
    if feats["flag_cv_only_no_nlp"]:
        score -= 25
    if feats["flag_recent_ai_only"]:
        score -= 20
    if feats["flag_no_code_recently"]:
        score -= 20
    if feats["flag_title_chasing"]:
        score -= 10
    if feats["num_expert_skills_zero_duration"] > 0:
        score -= 15 * feats["num_expert_skills_zero_duration"]

    if feats["has_vector_db_or_hybrid_search"]:
        score += 15
    if feats["has_eval_framework_experience"]:
        score += 15
    if feats["has_embeddings_production_experience"]:
        score += 15

    yoe = feats["years_of_experience"]
    if 5 <= yoe <= 9:
        score += 10
    elif yoe < 2:
        score -= 10

    if feats["days_inactive"] is not None:
        if feats["days_inactive"] > 180:
            score -= 10
        elif feats["days_inactive"] < 30:
            score += 5

    score += feats["recruiter_response_rate"] * 10
    if feats["notice_period_days"] <= 30:
        score += 5
    if feats["open_to_work_flag"]:
        score += 5
    if feats["logistics_fit"]:
        score += 5

    score += semantic_score * 55

    suspicion = feats["honeypot_suspicion_score"]
    if suspicion >= 4:
        score -= 80
    elif suspicion >= 2:
        score -= 20

    return score


def main():
    start = time.time()
    scored = []

    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidate = json.loads(line)
            feats = extract_features(candidate)
            sem_score = SEMANTIC_SCORES.get(feats["candidate_id"], 0.0)
            score = round(composite_score(feats, sem_score), 2)
            scored.append((score, candidate, feats, sem_score))

    # Sort best first; tie-break by candidate_id for determinism
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))

    top_100 = scored[:100]

    rows = []
    for rank, (score, candidate, feats, sem_score) in enumerate(top_100, start=1):
        reasoning = generate_reasoning(feats, sem_score)
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "rank": rank,
            "score": round(score, 2),
            "reasoning": reasoning,
        })

    with open("top_100_ranked.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f} seconds")
    print(f"Wrote {len(rows)} rows to top_100_ranked.csv")
    print(f"\nRank 1: {rows[0]}")
    print(f"\nRank 100: {rows[-1]}")


if __name__ == "__main__":
    main()