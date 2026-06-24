"""
reasoning.py
Generates a 1-2 sentence, fact-based explanation for why a candidate ranked
where they did. No LLM call -- built directly from extracted features, so
it can never hallucinate and runs instantly.
"""


def generate_reasoning(feats: dict, semantic_score: float) -> str:
    title = feats["current_title"]
    company = feats["current_company"]
    yoe = feats["years_of_experience"]

    strengths = []
    if feats["has_vector_db_or_hybrid_search"]:
        strengths.append("hands-on production experience with vector databases or hybrid search")
    if feats["has_eval_framework_experience"]:
        strengths.append("experience designing ranking evaluation frameworks (NDCG/MRR/MAP)")
    if feats["has_embeddings_production_experience"]:
        strengths.append("production embeddings-based retrieval experience")

    base = f"{title} at {company} with {yoe} years of experience"
    if strengths:
        sentence1 = base + ", bringing " + " and ".join(strengths[:2]) + "."
    else:
        sentence1 = base + "."

    extras = []
    if feats["notice_period_days"] <= 30:
        extras.append("available within a 30-day notice period")
    if feats["logistics_fit"]:
        extras.append("a logistics fit for the Pune/Noida hybrid setup")
    if feats["recruiter_response_rate"] >= 0.5:
        extras.append("a strong recruiter-response history")
    if semantic_score >= 0.7:
        extras.append("profile substance that closely matches the role, not just keyword overlap")

    sentence2 = ""
    if extras:
        sentence2 = "Also shows " + ", ".join(extras[:2]) + "."

    return (sentence1 + " " + sentence2).strip()