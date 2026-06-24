# India Runs — Intelligent Candidate Discovery & Ranking

AI-driven candidate ranking system built for the India Runs Data & AI Challenge
(Hack2skill x Redrob AI), ranking 100,000 candidate profiles against a job
description using a 3-layer hybrid approach.

## Approach

1. **Rule-based feature extraction** (`features.py`) — pulls explicit
   disqualifier signals directly from the job description (consulting-only
   backgrounds, pure-research-no-production candidates, title mismatches,
   etc.) plus honeypot/fake-profile suspicion scoring.

2. **Semantic similarity** (`precompute_embeddings.py`) — uses
   `sentence-transformers` (all-MiniLM-L6-v2) to compare the JD's meaning
   against each candidate's profile text, catching cases keyword search
   would miss in both directions.

3. **Composite scoring + reasoning** (`scoring.py`, `reasoning.py`) —
   combines rule-based signals, semantic similarity, and behavioral signals
   into one score, with a fact-based (non-LLM, hallucination-free) reasoning
   sentence per candidate.

4. **Submission generation** (`generate_submission.py`) — produces the final
   validated `top_100_ranked.csv`.

## How to run

1. Place `candidates.jsonl` (the 100K candidate dataset, not included in
   this repo due to size) in this folder.
2. Install dependencies:
3. Precompute semantic similarity scores (one-time, ~30-50 min on CPU):
4. Generate the final ranked submission:
This produces `top_100_ranked.csv` in under 5 minutes (the precompute
   step above is not time-limited; only this final step is).

## Validation

Top-100 results contain 0% flagged honeypot/suspicious profiles
(well under the 10% disqualification threshold), and the output passes
the organizers' official `validate_submission.py` format checker.

## Author

Nandhana K M