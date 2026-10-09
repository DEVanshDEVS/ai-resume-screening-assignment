import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
import json
from io import BytesIO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.screening import analyze_text, GithubInfo
from src.llm import LLMConfig, LLMAssessment, assess_project_depth
from main import run

class ScreeningTests(unittest.TestCase):
    def test_eligibility_requires_python_and_ai(self):
        self.assertTrue(analyze_text("Python developer built a LangChain RAG app.",Path("a.txt")).eligible)
        candidate=analyze_text("JavaScript React developer built a frontend.",Path("b.txt"))
        self.assertFalse(candidate.eligible)
        self.assertEqual(len(candidate.rejection_reasons),2)

    def test_shallow_ai_wrapper_penalty_and_score_cap(self):
        shallow=analyze_text("Python and LangChain skills. LLM.",Path("a.txt"))
        self.assertTrue(shallow.eligible)
        self.assertLessEqual(shallow.score_breakdown["ai_project_depth"],17)
        rich=analyze_text("Built Python FastAPI service. Implemented a LangGraph multi-agent RAG workflow with embeddings, vector search, tool calling, evaluation, Docker, PostgreSQL, pytest, caching and retries.",Path("b.txt"),GithubInfo("success","active",10,"dev"))
        self.assertLessEqual(rich.total_score,100)
        self.assertEqual(rich.score_breakdown["github"],10)

class BatchReliabilityTests(unittest.TestCase):
    def test_malformed_pdf_does_not_abort_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"broken.pdf").write_bytes(b"not a PDF")
            result=run(root,root/"result.json")
        self.assertEqual(result["batch_summary"]["total_resumes"],1)
        self.assertEqual(result["batch_summary"]["failed_or_unreadable"],1)
        self.assertEqual(result["batch_summary"]["successfully_parsed"],0)

    def test_duplicate_text_is_counted_separately_from_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"one.txt").write_text("Python developer built a LangChain RAG service.",encoding="utf-8")
            (root/"two.txt").write_text("Python developer built a LangChain RAG service.",encoding="utf-8")
            result=run(root,root/"result.json")
        summary=result["batch_summary"]
        self.assertEqual(summary["duplicates_skipped"],1)
        self.assertEqual(summary["failed_or_unreadable"],0)
        self.assertEqual(summary["successfully_parsed"],1)

    def test_github_failure_does_not_reject_candidate_or_abort_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            resume=root/"candidate.txt"
            resume.write_text("Jane Doe\nPython developer implemented a LangChain RAG pipeline. https://github.com/example-user",encoding="utf-8")
            with patch("main.urlopen",side_effect=URLError("offline")):
                result=run(root,root/"result.json")
        self.assertEqual(result["batch_summary"]["eligible"],1)
        candidate=result["ranked_candidates"][0]
        self.assertEqual(candidate["github"]["status"],"failed")
        self.assertEqual(candidate["score_breakdown"]["github"],0)

    def test_llm_failure_falls_back_to_deterministic_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"candidate.txt").write_text("Jane Doe\nPython developer implemented a LangChain RAG pipeline.",encoding="utf-8")
            with patch("main.assess_project_depth",side_effect=RuntimeError("provider unavailable")):
                result=run(root,root/"result.json",LLMConfig("test-key","test-model"))
        candidate=result["ranked_candidates"][0]
        self.assertEqual(candidate["llm_assessment"]["status"],"failed")
        self.assertGreater(candidate["score_breakdown"]["ai_project_depth"],0)
        self.assertEqual(result["batch_summary"]["llm_failed"],1)

    def test_successful_llm_score_updates_ai_category_and_total(self):
        resume_text="Jane Doe\nPython developer implemented a LangChain RAG pipeline."
        assessment=LLMAssessment("success",37,"Implemented a retrieval pipeline.",["implemented a LangChain RAG pipeline"],["Concrete implementation"],[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"candidate.txt").write_text(resume_text,encoding="utf-8")
            with patch("main.assess_project_depth",return_value=assessment):
                result=run(root,root/"result.json",LLMConfig("test-key","test-model"))
        candidate=result["ranked_candidates"][0]
        self.assertEqual(candidate["score_breakdown"]["ai_project_depth"],37)
        self.assertEqual(candidate["llm_assessment"]["status"],"success")
        self.assertEqual(candidate["project_summary"],assessment.project_summary)

class LLMAdapterTests(unittest.TestCase):
    def test_structured_output_requires_resume_grounded_evidence(self):
        content={"ai_project_depth":24,"project_summary":"Built retrieval.","evidence":["Implemented a RAG pipeline"],"strengths":["Retrieval workflow"],"concerns":[]}
        api_response={"choices":[{"message":{"content":json.dumps(content)}}]}
        request_capture={}
        def fake_urlopen(request,timeout):
            request_capture["request"]=request
            request_capture["timeout"]=timeout
            return BytesIO(json.dumps(api_response).encode("utf-8"))
        with patch("src.llm.urlopen",side_effect=fake_urlopen):
            result=assess_project_depth("Python developer. Implemented a RAG pipeline.",LLMConfig("secret","test-model"))
        self.assertEqual(result.status,"success")
        self.assertEqual(result.score,24)
        sent=json.loads(request_capture["request"].data)
        self.assertEqual(sent["response_format"]["type"],"json_schema")
        self.assertTrue(sent["response_format"]["json_schema"]["strict"])
        self.assertEqual(request_capture["request"].get_header("Authorization"),"Bearer secret")

    def test_unverifiable_llm_evidence_is_rejected(self):
        content={"ai_project_depth":40,"project_summary":"Excellent.","evidence":["Claims a complex multi-agent system"],"strengths":[],"concerns":[]}
        api_response={"choices":[{"message":{"content":json.dumps(content)}}]}
        with patch("src.llm.urlopen",return_value=BytesIO(json.dumps(api_response).encode("utf-8"))):
            with self.assertRaisesRegex(ValueError,"evidence quoted"):
                assess_project_depth("Python and LangChain skills.",LLMConfig("secret","test-model"))

if __name__ == "__main__":
    unittest.main()

