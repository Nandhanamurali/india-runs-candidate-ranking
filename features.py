"""
features.py
Rule-based feature extraction + honeypot suspicion detection for the
Redrob Intelligent Candidate Discovery challenge.

No ML, no network calls -- fast and fully explainable.
"""

from datetime import datetime, date

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree",
    "l&t infotech", "mphasis"
}

PRODUCTION_EVIDENCE_TERMS = [
    "deployed", "production", "scale", "shipped", "users", "live", "launched",
    "real-time", "real time", "throughput", "latency", "rollout", "on-call",
    "monitoring", "a/b test", "ab test", "rollback"
]

RESEARCH_ONLY_TERMS = ["research scientist", "research assistant", "phd", "postdoc", "academic"]

NLP_IR_TERMS = [
    "nlp", "natural language", "retrieval", "search", "ranking", "embedding",
    "vector", "rag", "llm", "information retrieval", "bm25", "semantic search"
]

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "object detection", "speech recognition",
    "robotics", "tts", "asr", "lidar", "slam", "autonomous"
]

CORE_ML_INFRA_SKILLS = [
    "sentence-transformers", "openai embeddings", "bge", "e5", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "faiss"
]

EVAL_TERMS = ["ndcg", "mrr", "map", "a/b test", "ab test", "offline evaluation", "online evaluation"]

NON_ENGINEERING_TITLE_TERMS = [
    "manager", "operations", "marketing", "sales", "support", "hr",
    "recruiter", "customer", "administrator", "coordinator"
]

TARGET_LOCATIONS = {"pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram", "bengaluru", "bangalore"}

REFERENCE_DATE = date(2026, 6, 19)


def _text_blob(candidate):
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
        candidate["profile"].get("current_title", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(parts).lower()


def _parse_date(d, default=None):
    if not d:
        return default
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except ValueError:
        return default


def honeypot_signals(candidate: dict) -> dict:
    """Returns a suspicion score (higher = more likely fake) plus the individual flags."""
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    edu = candidate.get("education", [])
    signals = candidate["redrob_signals"]
    profile = candidate["profile"]

    suspicion = 0
    reasons = []

    # 1. Overlapping full-time jobs (more than 2 months overlap)
    intervals = []
    for job in career:
        sd = _parse_date(job.get("start_date"))
        ed = _parse_date(job.get("end_date"), default=REFERENCE_DATE)
        if sd:
            intervals.append((sd, ed))
    intervals.sort()
    for i in range(len(intervals) - 1):
        overlap_days = (intervals[i][1] - intervals[i + 1][0]).days
        if overlap_days > 60:
            suspicion += 3
            reasons.append("overlapping_jobs")
            break

    # 2. Expert skill claimed with zero months of use
    zero_dur_experts = [s["name"] for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0]
    if zero_dur_experts:
        suspicion += 3
        reasons.append("expert_zero_duration")

    # 3. Expert skill but very low assessment score
    assess = signals.get("skill_assessment_scores", {})
    low_assessment_experts = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and s["name"] in assess and assess[s["name"]] < 40
    ]
    if low_assessment_experts:
        suspicion += 2
        reasons.append("expert_low_assessment")

    # 4. Implausibly many expert-level skills
    num_experts = sum(1 for s in skills if s.get("proficiency") == "expert")
    if num_experts >= 15:
        suspicion += 2
        reasons.append("too_many_expert_skills")

    # 5. Career duration sum vs stated years_of_experience -- big mismatch
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    yoe_months = profile.get("years_of_experience", 0) * 12
    if yoe_months > 0 and abs(total_career_months - yoe_months) > 36:  # more than 3 years off
        suspicion += 1
        reasons.append("experience_math_mismatch")

    # 6. Zero verified contact channels despite a near-complete profile
    verified_count = sum([
        bool(signals.get("verified_email")),
        bool(signals.get("verified_phone")),
        bool(signals.get("linkedin_connected")),
    ])
    if verified_count == 0 and signals.get("profile_completeness_score", 0) >= 95:
        suspicion += 2
        reasons.append("unverified_but_complete")

    # 7. Years of experience wildly exceeds time since graduation (generous threshold)
    if edu:
        last_grad_year = max((e.get("end_year", 2026) for e in edu), default=2026)
        years_since_grad = 2026 - last_grad_year
        yoe = profile.get("years_of_experience", 0)
        if yoe > years_since_grad + 6:
            suspicion += 1
            reasons.append("experience_exceeds_graduation_generous")

    return {
        "honeypot_suspicion_score": suspicion,
        "honeypot_reasons": reasons,
    }


def extract_features(candidate: dict) -> dict:
    blob = _text_blob(candidate)
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]

    f = {"candidate_id": candidate["candidate_id"]}

    research_terms_present = any(t in blob for t in RESEARCH_ONLY_TERMS)
    has_production_evidence = any(t in blob for t in PRODUCTION_EVIDENCE_TERMS)
    f["flag_pure_research_no_production"] = research_terms_present and not has_production_evidence

    companies = [job.get("company", "").lower() for job in career]
    all_consulting = len(companies) > 0 and all(
        any(firm in c for firm in CONSULTING_FIRMS) for c in companies
    )
    f["flag_consulting_only"] = all_consulting

    has_cv_speech_robotics = any(t in blob for t in CV_SPEECH_ROBOTICS_TERMS)
    has_nlp_ir = any(t in blob for t in NLP_IR_TERMS)
    f["flag_cv_only_no_nlp"] = has_cv_speech_robotics and not has_nlp_ir

    ai_skill_names = {s["name"].lower() for s in skills if any(
        k in s["name"].lower() for k in ["llm", "langchain", "gpt", "openai", "rag"]
    )}
    ai_skill_durations = [s.get("duration_months", 0) for s in skills if s["name"].lower() in ai_skill_names]
    only_recent_ai = bool(ai_skill_durations) and max(ai_skill_durations, default=0) < 12
    years_exp = profile.get("years_of_experience", 0)
    f["flag_recent_ai_only"] = only_recent_ai and years_exp < 3

    current_title = profile.get("current_title", "").lower()
    is_pure_architect_lead = any(t in current_title for t in ["architect", "tech lead", "engineering manager", "director"])
    current_role = next((j for j in career if j.get("is_current")), None)
    long_in_role_no_code = (
        is_pure_architect_lead
        and current_role is not None
        and current_role.get("duration_months", 0) >= 18
        and not has_production_evidence
    )
    f["flag_no_code_recently"] = long_in_role_no_code

    durations = [j.get("duration_months", 0) for j in career]
    avg_tenure = sum(durations) / len(durations) if durations else 0
    f["flag_title_chasing"] = len(career) >= 3 and avg_tenure < 18

    f["flag_non_engineering_title"] = any(t in current_title for t in NON_ENGINEERING_TITLE_TERMS)

    skill_names_lower = {s["name"].lower() for s in skills}
    f["has_vector_db_or_hybrid_search"] = any(
        any(infra in sn for infra in CORE_ML_INFRA_SKILLS) for sn in skill_names_lower
    ) or any(t in blob for t in ["vector database", "hybrid search", "vector search"])
    f["has_eval_framework_experience"] = any(t in blob for t in EVAL_TERMS)
    f["has_embeddings_production_experience"] = "embedding" in blob and has_production_evidence

    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    f["num_expert_skills"] = len(expert_skills)
    f["num_expert_skills_zero_duration"] = sum(
        1 for s in expert_skills if s.get("duration_months", 0) == 0
    )

    location = profile.get("location", "").lower()
    f["in_target_location"] = any(loc in location for loc in TARGET_LOCATIONS)
    f["willing_to_relocate"] = signals.get("willing_to_relocate", False)
    f["logistics_fit"] = f["in_target_location"] or f["willing_to_relocate"]

    last_active = signals.get("last_active_date")
    days_inactive = None
    if last_active:
        d = _parse_date(last_active)
        if d:
            days_inactive = (REFERENCE_DATE - d).days
    f["days_inactive"] = days_inactive
    f["recruiter_response_rate"] = signals.get("recruiter_response_rate", 0)
    f["notice_period_days"] = signals.get("notice_period_days", 999)
    f["open_to_work_flag"] = signals.get("open_to_work_flag", False)
    f["github_activity_score"] = signals.get("github_activity_score", -1)
    f["interview_completion_rate"] = signals.get("interview_completion_rate", 0)
    f["offer_acceptance_rate"] = signals.get("offer_acceptance_rate", -1)

    f["verified_count"] = sum([
        bool(signals.get("verified_email")),
        bool(signals.get("verified_phone")),
        bool(signals.get("linkedin_connected")),
    ])

    f["years_of_experience"] = years_exp
    f["avg_tenure_months"] = round(avg_tenure, 1)
    f["current_title"] = profile.get("current_title", "")
    f["current_company"] = profile.get("current_company", "")

    # merge in honeypot signals
    f.update(honeypot_signals(candidate))

    return f


if __name__ == "__main__":
    import json

    with open("sample_candidates.json", "r", encoding="utf-8") as fh:
        candidates = json.load(fh)

    print("Honeypot suspicion scores for all 50 sample candidates:")
    for c in candidates:
        feats = extract_features(c)
        if feats["honeypot_suspicion_score"] > 0:
            print(feats["candidate_id"], feats["honeypot_suspicion_score"], feats["honeypot_reasons"])
    print("\n(If nothing printed above, none of the 50 samples triggered any honeypot flags -- expected, since real honeypots are rare.)")