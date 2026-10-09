"""Deterministic, evidence-oriented resume screening primitives."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from .config import (
    AI_BASE_POINTS, AI_EVIDENCE_CAP, AI_EVIDENCE_POINTS, AI_OWNERSHIP_CAP,
    AI_OWNERSHIP_POINTS, AI_SHALLOW_PENALTY, CLOUD_SIGNAL_POINTS,
    PYTHON_BASE_POINTS, PYTHON_PROJECT_POINTS, PYTHON_STACK_POINTS,
    SCORE_WEIGHTS,
)

SKILL_PATTERNS = {
    "Python": r"\bpython\b", "FastAPI": r"\bfastapi\b", "Django": r"\bdjango\b",
    "Flask": r"\bflask\b", "PostgreSQL": r"\bpostgres(?:ql)?\b", "Redis": r"\bredis\b",
    "Docker": r"\bdocker\b", "GCP": r"\b(?:gcp|google cloud|cloud run|vertex ai)\b",
    "React": r"\breact\b", "Next.js": r"\bnext\.js\b", "LangChain": r"\blangchain\b",
    "LangGraph": r"\blanggraph\b", "LlamaIndex": r"\bllamaindex\b", "Google ADK": r"\bgoogle adk\b",
    "RAG": r"\brag\b|retrieval augmented generation", "Embeddings": r"\bembeddings?\b",
    "Vector search": r"vector (?:database|search|store)|\bfaiss\b|\bchroma(?:db)?\b|\bpgvector\b",
    "Tool calling": r"tool[- ]calling|function calling", "Multi-agent": r"multi[- ]agent",
    "LLM": r"\bllms?\b|large language model", "Pytest": r"\bpytest\b",
    "Async": r"\basync(?:io)?\b|asynchronous", "Kubernetes": r"\bkubernetes\b|\bk8s\b",
}
AI_RE = re.compile(r"\b(?:langchain|langgraph|llamaindex|google adk|rag|retrieval[- ]augmented|embeddings?|vector (?:search|store|database)|faiss|chromadb|pgvector|tool[- ]calling|function calling|multi[- ]agent|llm|large language model|generative ai|agentic)\b", re.I)
PYTHON_RE = re.compile(r"\bpython\b", re.I)
GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/([A-Za-z0-9-]+)(?:/[^\s,;)]*)?", re.I)
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)

@dataclass
class GithubInfo:
    status: str = "not_found"
    summary: str = "No GitHub profile found in resume."
    score: int = 0
    username: str | None = None

@dataclass
class Candidate:
    source_file: str
    candidate_name: str
    email: str | None
    github: GithubInfo
    eligible: bool
    rejection_reasons: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    project_summary: str = ""
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    score_breakdown: dict[str, int] = field(default_factory=dict)
    total_score: int = 0
    rank: int | None = None
    llm_assessment: dict[str, Any] = field(default_factory=lambda: {"status": "not_requested"})

def extract_text(path: Path) -> str:
    """Extract text from PDF or plain text. A bad file raises to the batch boundary."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path), strict=False)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path.suffix}")

def _snippets(text: str, pattern: re.Pattern[str], limit: int = 3) -> list[str]:
    out=[]
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence=" ".join(sentence.split())
        if pattern.search(sentence) and sentence not in out:
            out.append(sentence[:280])
        if len(out)>=limit: break
    return out

def _name(text: str, path: Path) -> str:
    for line in text.splitlines()[:12]:
        line=" ".join(line.split()).strip(" |•")
        if line and "," not in line and ":" not in line and not EMAIL_RE.search(line) and not re.search(r"resume|curriculum vitae|https?://|phone|linkedin|professional summary|skills?(?: summary)?|technical skills|work experience|education|objective|certifications?|projects?|languages?", line, re.I) and len(line)<70:
            words=line.split()
            if 1<len(words)<=5 and not any(c.isdigit() for c in line): return line
    return path.stem.replace("_", " ").replace("-", " ").title()

def analyze_text(text: str, path: Path, github_info: GithubInfo | None = None) -> Candidate:
    skills=[name for name, pat in SKILL_PATTERNS.items() if re.search(pat, text, re.I)]
    py_evidence=_snippets(text, PYTHON_RE)
    ai_evidence=_snippets(text, AI_RE)
    reasons=[]
    if not py_evidence: reasons.append("No evidence of Python stack")
    if not ai_evidence: reasons.append("No AI/agentic project evidence")
    eligible=not reasons
    email_match=EMAIL_RE.search(text)
    gh=github_info or GithubInfo()
    if not gh.username:
        match=GITHUB_RE.search(text)
        if match: gh.username=match.group(1); gh.status="pending"
    evidence={}
    if py_evidence: evidence["python"]=py_evidence
    if ai_evidence: evidence["ai_agentic"]=ai_evidence
    summary=ai_evidence[0] if ai_evidence else "No qualifying AI/agentic project evidence identified."
    result=Candidate(path.name,_name(text,path),email_match.group(0) if email_match else None,gh,eligible,reasons,skills,evidence,summary)
    if eligible: score_candidate(result,text)
    return result

def score_candidate(c: Candidate, text: str) -> None:
    """Heuristic score with a visible split; skill lists count less than project evidence."""
    lower=text.lower()
    projects=re.split(r"\n(?=.{0,40}(?:projects?|experience|internship|work experience)\b)", text, flags=re.I)
    substantive=sum(bool(re.search(r"\b(?:built|developed|implemented|designed|deployed|integrated|created|reduced|improved)\b", p, re.I) and AI_RE.search(p)) for p in projects)
    ai=(AI_BASE_POINTS + min(AI_EVIDENCE_CAP, len(c.evidence.get("ai_agentic",[]))*AI_EVIDENCE_POINTS)
        + min(AI_OWNERSHIP_CAP, substantive*AI_OWNERSHIP_POINTS))
    if substantive==0:
        ai=max(0,ai-AI_SHALLOW_PENALTY)
        note="AI evidence is limited; implementation ownership is unclear (shallow-project penalty applied)."
        if note not in c.concerns: c.concerns.append(note)
    ai=min(SCORE_WEIGHTS["ai_project_depth"],ai)
    python=PYTHON_BASE_POINTS + min(9, sum(1 for s in ("fastapi","django","flask","postgres","redis","async") if s in lower)*PYTHON_STACK_POINTS)
    if re.search(r"\b(?:built|developed|implemented|deployed)\b.{0,100}\bpython\b|\bpython\b.{0,100}\b(?:built|developed|implemented|deployed)\b", lower): python+=PYTHON_PROJECT_POINTS
    python=min(SCORE_WEIGHTS["python_backend"],python)
    cloud=min(SCORE_WEIGHTS["cloud_fullstack"], sum(CLOUD_SIGNAL_POINTS for s in ("gcp","google cloud","docker","kubernetes","react","next.js") if s in lower))
    eng=min(5, sum(1 for s in ("pytest","unit test","integration test","caching","queue","observability","retry","concurrency","async") if s in lower))
    c.score_breakdown={"ai_project_depth":ai,"python_backend":python,"cloud_fullstack":cloud,"github":min(SCORE_WEIGHTS["github"],c.github.score),"engineering_depth":min(SCORE_WEIGHTS["engineering_depth"],eng)}
    c.total_score=sum(c.score_breakdown.values())
    c.strengths=[]
    if substantive: c.strengths.append("AI project includes implementation ownership signals")
    if python>=20: c.strengths.append("Python backend experience beyond a skill-list mention")
    if cloud: c.strengths.append("Cloud, deployment, or full-stack evidence")
    if not c.concerns: c.concerns.append("No major concern detected by the deterministic heuristic; review resume evidence manually.")

def to_dict(candidate: Candidate) -> dict[str, Any]: return asdict(candidate)

