# AI Resume Screening & Ranking System

A small Python CLI that extracts text from resumes, applies explicit Python + AI/agentic eligibility gates, assigns an explainable 100-point score, optionally enriches public GitHub profiles, and writes JSON. The supplied dataset contains 50 PDFs.

## Run

Python 3.10+ is recommended. Install dependencies and run from this directory:

```bash
python -m pip install -r requirements.txt
python main.py --input ./resume_files --output ./results.json
```

Run the built-in unit tests with:

```bash
python -m unittest discover -s tests -v
```

Set `GITHUB_TOKEN` in the environment to raise GitHub API rate limits. It is optional and never stored in source. Without a token the program uses the public unauthenticated API and records any rate limit or lookup error. Network calls have a short timeout and results are cached per username for the run.

### Optional LLM project assessment

The default run does not send resume text to an LLM. To enable structured AI-project scoring, set `LLM_API_KEY` and `LLM_MODEL` in your environment, optionally set `LLM_BASE_URL`, then add `--use-llm`:

```powershell
$env:LLM_API_KEY = "your-api-key"
$env:LLM_MODEL = "your-supported-model"
python main.py --input ./resume_files --output ./results.json --use-llm
```

When enabled, extracted resume text for eligible candidates is sent to the configured OpenAI-compatible Chat Completions endpoint. The model returns a strict JSON-schema assessment for the AI project-depth category only; Python + AI eligibility remains deterministic, and a failed or invalid model response falls back to the heuristic score. Model evidence must quote text from the resume. Use this opt-in only when you are allowed to send these resumes to that provider. The `.env.example` file documents variable names; the CLI reads environment variables and does not automatically load a `.env` file.

PDFs with selectable text are supported; `.txt` and `.md` are also accepted. Image-only/scanned PDFs are marked as unreadable because OCR is not included. An individual parse failure does not stop the batch. Exact duplicate extracted text is skipped and listed under `processing_errors`.

## Output

`results.json` contains `batch_summary`, `ranked_candidates`, `rejected_candidates`, and `processing_errors`. Eligible candidates include rank, score breakdown, matched skills, resume evidence snippets, strengths, concerns, and GitHub status. Rejected candidates include explicit reasons. Duplicate files are reported separately from unreadable files. By default, scoring is deterministic and makes no LLM calls; with `--use-llm`, the AI project-depth score uses the structured assessment when available and falls back to the deterministic score on failure.

## Design Decisions

- **Filtering:** Python and meaningful AI/LLM/RAG/agentic vocabulary must both occur in extracted resume text. JavaScript/Java/React do not count against a candidate. The rules are deliberately outside any model and include evidence snippets for review.
- **Scoring:** Eligible profiles receive up to 40 AI project depth, 30 Python/backend, 15 cloud/full-stack, 10 GitHub, and 5 engineering depth points. Project ownership verbs and concrete implementation evidence influence project depth; a thin AI mention receives a penalty. Skill mentions alone do not trigger maximum points. The simple keyword/sentence heuristics are a transparent baseline, not a substitute for recruiter review.
- **LLM usage:** Optional adapter in `src/llm.py`. Structured output returns an AI project-depth score, exact evidence quotes, short summary, strengths, and concerns. Evidence is checked against resume text; the hard eligibility gate stays outside the model. Calls are opt-in, sequential, bounded by a timeout, and failures retain deterministic scores.
- **GitHub:** Public user repositories are read from the GitHub REST API. Up to 5 activity points reflect recent repository pushes (last 180 days), and up to 5 relevance points reflect Python/AI-related repositories among the 30 most recently updated. Missing URLs score zero and do not affect eligibility. Failures and rate limits are recorded per profile.
- **Configuration:** Score weights, heuristic point values, and GitHub lookup limits/timeouts are grouped in `src/config.py`; credentials are read from the environment.

## If I Had More Time

- Add OCR for scanned PDFs and richer DOCX parsing.
- Calibrate and test extraction/scoring against a recruiter-labelled sample, with section-aware evidence attribution.
- Evaluate LLM scoring consistency and provider cost across a recruiter-labelled sample.
- Improve GitHub activity estimation with public events/commit data and conditional HTTP caching.

## Limitations

This is a heuristic screening aid. PDF layouts vary, keyword matches can miss equivalent language or match irrelevant context, and repository update times are only a proxy for activity. Review evidence and source resumes before making decisions.

