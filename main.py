from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.screening import GithubInfo, analyze_text, extract_text, score_candidate, to_dict
from src.config import GITHUB_RECENT_DAYS, GITHUB_REPOSITORY_LIMIT, GITHUB_TIMEOUT_SECONDS
from src.llm import LLMConfig, assess_project_depth, config_from_environment

def github_enrichment(username: str, cache: dict[str, GithubInfo]) -> GithubInfo:
    if username in cache: return cache[username]
    headers={"Accept":"application/vnd.github+json", "User-Agent":"resume-screening-assignment"}
    token=os.getenv("GITHUB_TOKEN")
    if token: headers["Authorization"]="Bearer "+token
    try:
        def get(url):
            req=Request(url,headers=headers)
            with urlopen(req,timeout=GITHUB_TIMEOUT_SECONDS) as response: return json.load(response)
        user=get(f"https://api.github.com/users/{username}")
        repos=get(f"https://api.github.com/users/{username}/repos?sort=updated&per_page={GITHUB_REPOSITORY_LIMIT}&type=public")
        cutoff=time.time()-GITHUB_RECENT_DAYS*86400
        recent=0; relevant=0
        for repo in repos if isinstance(repos,list) else []:
            updated=repo.get("pushed_at") or repo.get("updated_at") or ""
            try:
                from datetime import datetime
                dt=datetime.fromisoformat(updated.replace("Z","+00:00")).timestamp()
                recent+=dt>=cutoff
            except (ValueError,TypeError): pass
            topics=" ".join(repo.get("topics") or [])
            text=" ".join((repo.get("name") or "", repo.get("description") or "", repo.get("language") or "",topics)).lower()
            relevant+=bool(re.search(r"python|ai|llm|agent|rag|langchain|machine learning",text))
        score=min(5, recent) + min(5, relevant)
        summary=f"{recent} public repositories updated in the last {GITHUB_RECENT_DAYS} days; {relevant} relevant Python/AI repositories among up to {GITHUB_REPOSITORY_LIMIT} recent repositories."
        info=GithubInfo("success",summary,score,username)
    except (HTTPError,URLError,TimeoutError,OSError,ValueError) as exc:
        status="rate_limited" if isinstance(exc,HTTPError) and exc.code==403 else "failed"
        info=GithubInfo(status,f"Public GitHub lookup failed ({status}); screening continued.",0,username)
    cache[username]=info
    return info

def run(input_dir: Path, output_file: Path, llm_config: LLMConfig | None = None) -> dict:
    files=sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf",".txt",".md"})
    records=[]; errors=[]; seen=set(); cache={}; duplicate_count=0
    for path in files:
        try:
            text=extract_text(path)
            if not text.strip(): raise ValueError("No extractable text (possibly scanned PDF)")
            digest=__import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                duplicate_count+=1
                errors.append({"source_file":path.name,"error":"Duplicate content; skipped."}); continue
            seen.add(digest)
            c=analyze_text(text,path)
            if c.github.username: c.github=github_enrichment(c.github.username,cache)
            if c.eligible:
                c.llm_assessment={"status":"disabled" if llm_config is None else "pending"}
                if llm_config is not None:
                    try:
                        assessment=assess_project_depth(text,llm_config)
                        c.llm_assessment=assessment.__dict__
                        if assessment.status=="success" and assessment.score is not None:
                            score_candidate(c,text)
                            c.score_breakdown["ai_project_depth"]=assessment.score
                            c.total_score=sum(c.score_breakdown.values())
                            if assessment.project_summary: c.project_summary=assessment.project_summary
                            c.strengths.extend(x for x in (assessment.strengths or []) if x not in c.strengths)
                            c.concerns.extend(x for x in (assessment.concerns or []) if x not in c.concerns)
                    except Exception as exc:
                        c.llm_assessment={"status":"failed","error":f"{type(exc).__name__}: {str(exc)[:220]}"}
            records.append(c)
        except Exception as exc:
            errors.append({"source_file":path.name,"error":f"{type(exc).__name__}: {exc}"})
    eligible=sorted((c for c in records if c.eligible),key=lambda c:(-c.total_score,c.candidate_name.casefold(),c.source_file))
    for rank,c in enumerate(eligible,1): c.rank=rank
    rejected=[c for c in records if not c.eligible]
    llm_statuses=[c.llm_assessment.get("status") for c in records if c.eligible]
    result={"batch_summary":{"total_resumes":len(files),"successfully_parsed":len(records),"eligible":len(eligible),"rejected":len(rejected),"duplicates_skipped":duplicate_count,"failed_or_unreadable":len(errors)-duplicate_count,"llm_enabled":llm_config is not None,"llm_model":llm_config.model if llm_config else None,"llm_succeeded":llm_statuses.count("success"),"llm_failed":llm_statuses.count("failed")},"ranked_candidates":[to_dict(c) for c in eligible],"rejected_candidates":[to_dict(c) for c in rejected],"processing_errors":errors}
    output_file.parent.mkdir(parents=True,exist_ok=True)
    output_file.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    return result

def main():
    parser=argparse.ArgumentParser(description="Screen resumes for Python and AI/agentic engineering evidence.")
    parser.add_argument("--input",required=True,type=Path); parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--use-llm",action="store_true",help="Send eligible resume text to the configured LLM for project-depth assessment.")
    args=parser.parse_args()
    try: llm_config=config_from_environment() if args.use_llm else None
    except ValueError as exc: parser.error(str(exc))
    result=run(args.input,args.output,llm_config)
    print(json.dumps(result["batch_summary"],indent=2))
    print(f"Results written to {args.output}")

if __name__=="__main__": main()

