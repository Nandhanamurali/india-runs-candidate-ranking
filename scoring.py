"""
scoring.py
Combines rule-based features + semantic similarity + honeypot suspicion
into one score per candidate, then ranks the full pool.
"""

import json
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

    # Honeypot penalty -- heavy, scales with how many red flags piled up
    suspicion = feats["honeypot_suspicion_score"]
    if suspicion >= 4:
        score -= 80  # almost certainly fake, push to bottom
    elif suspicion >= 2:
        score -= 20  # moderately suspicious, meaningful penalty

    return round(score, 2)


def main():
    start = time.time()
    results = []

    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidate = json.loads(line)
            feats = extract_features(candidate)
            sem_score = SEMANTIC_SCORES.get(feats["candidate_id"], 0.0)
            score = composite_score(feats, sem_score)
            results.append((
                score, feats["candidate_id"], feats["current_title"],
                feats["current_company"], feats["years_of_experience"],
                round(sem_score, 3), feats["honeypot_suspicion_score"]
            ))

    results.sort(key=lambda r: r[0], reverse=True)

    print(f"Scored {len(results)} candidates in {time.time() - start:.1f} seconds\n")

    top_100 = results[:100]
    suspicious_in_top_100 = sum(1 for r in top_100 if r[6] >= 2)
    print(f"Honeypot check: {suspicious_in_top_100} out of top 100 flagged as suspicious (threshold for disqualification is >10, i.e. >10%)\n")


    print("TOP 5 WITH REASONING:")
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        all_lines = {}
        for line in f:
            c = json.loads(line)
            all_lines[c["candidate_id"]] = c

    for r in results[:5]:
        score, cid, title, company, yoe, sem, suspicion = r
        candidate = all_lines[cid]
        feats = extract_features(candidate)
        reasoning = generate_reasoning(feats, sem)
        print(f"\n{cid} | score={score} | {title} at {company}")
        print("Reasoning:", reasoning)

    print("\nBOTTOM 10:")
    for r in results[-10:]:
        print(r)


if __name__ == "__main__":
    main()