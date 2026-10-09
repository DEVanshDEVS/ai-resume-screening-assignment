"""Tunable thresholds and score weights for the resume screening baseline."""

SCORE_WEIGHTS = {
    "ai_project_depth": 40,
    "python_backend": 30,
    "cloud_fullstack": 15,
    "github": 10,
    "engineering_depth": 5,
}

AI_BASE_POINTS = 12
AI_EVIDENCE_POINTS = 5
AI_EVIDENCE_CAP = 18
AI_OWNERSHIP_POINTS = 5
AI_OWNERSHIP_CAP = 10
AI_SHALLOW_PENALTY = 8

PYTHON_BASE_POINTS = 8
PYTHON_STACK_POINTS = 2
PYTHON_PROJECT_POINTS = 8
CLOUD_SIGNAL_POINTS = 3

GITHUB_TIMEOUT_SECONDS = 6
GITHUB_RECENT_DAYS = 180
GITHUB_REPOSITORY_LIMIT = 30

LLM_TIMEOUT_SECONDS = 25
LLM_MAX_RESUME_CHARACTERS = 20_000

