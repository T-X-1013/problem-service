# -*- coding: utf-8 -*-

import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("problem-classify-service")


KEY_PROBLEM = "\u95ee\u9898"
KEY_TARGET_PROBLEM = "\u9488\u5bf9\u7684\u95ee\u9898"
KEY_MAJOR_CODE = "\u95ee\u9898\u5927\u7c7b\u7f16\u53f7"
KEY_MAJOR_NAME = "\u95ee\u9898\u5927\u7c7b\u540d\u79f0"
KEY_MINOR_CODE = "\u95ee\u9898\u5c0f\u7c7b\u7f16\u53f7"
KEY_MINOR_NAME = "\u95ee\u9898\u5c0f\u7c7b\u540d\u79f0"
KEY_ANSWER = "\u5ba2\u670d\u56de\u7b54"
KEY_RAW_SUMMARY = "\u539f\u6587\u6458\u8981"
KEY_EXPLAIN = "\u89e3\u91ca"

KEY_LIST = "\u5f02\u8bae\u5217\u8868"
KEY_CLASSIFY_LIST = "\u5206\u7c7b\u5217\u8868"
KEY_CODE = "\u7f16\u53f7"
KEY_BIG_CODE = "\u5927\u7c7b\u7f16\u53f7"
KEY_SMALL_CODE = "\u5c0f\u7c7b\u7f16\u53f7"
KEY_BIG_TITLE = "\u5927\u7c7b\u6807\u9898"
KEY_SMALL_TITLE = "\u5c0f\u7c7b\u6807\u9898"

CUSTOMER_LABEL = "\u5ba2\u6237"
USER_LABEL = "\u7528\u6237"
SERVICE_LABEL = "\u5ba2\u670d"


class ClassifyRequest(BaseModel):
    info: str = ""
    problem: str | None = None
    problems: Any | None = None


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_adapter_dir() -> str:
    candidates = [
        r"E:\hangdong_tool\model\classification",
        r"E:\hangdong_tool\model\classifacation",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


class Settings:
    mode = os.getenv("CLASSIFIER_MODE", "mock").strip().lower()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "9002"))
    base_model_dir = os.getenv("BASE_MODEL_DIR", "").strip()
    adapter_dir = os.getenv("ADAPTER_DIR", default_adapter_dir()).strip()
    knowledge_dir = os.getenv("KNOWLEDGE_DIR", "").strip()
    rag_top_k = int(os.getenv("RAG_TOP_K", "3"))
    rag_min_score = float(os.getenv("RAG_MIN_SCORE", "1.0"))
    rag_context_limit = int(os.getenv("RAG_CONTEXT_LIMIT", "3000"))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "384"))
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    top_p = float(os.getenv("TOP_P", "1.0"))
    trust_remote_code = env_bool("TRUST_REMOTE_CODE", True)
    return_raw = env_bool("RETURN_RAW", False)
    torch_dtype = os.getenv("TORCH_DTYPE", "auto").strip().lower()
    max_retries = int(os.getenv("MAX_RETRIES", "1"))
    hf_allow_cpu = env_bool("HF_ALLOW_CPU", False)


SETTINGS = Settings()
app = FastAPI(title="Problem Classify Service", version="2.0.0")


def torch_runtime_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "torchAvailable": False,
            "cudaAvailable": False,
            "deviceName": "",
            "error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    device_name = ""
    if cuda_available:
        try:
            device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            device_name = "cuda"

    return {
        "torchAvailable": True,
        "cudaAvailable": cuda_available,
        "deviceName": device_name,
        "error": None,
    }


def hf_execution_state() -> dict[str, Any]:
    runtime = torch_runtime_info()
    effective_mode = SETTINGS.mode
    warning = ""

    if SETTINGS.mode == "hf":
        if not runtime["torchAvailable"]:
            effective_mode = "rag-fallback"
            warning = "torch is unavailable, falling back to retrieval-only classification"
        elif not runtime["cudaAvailable"] and not SETTINGS.hf_allow_cpu:
            effective_mode = "rag-fallback"
            warning = "CUDA is unavailable, falling back to retrieval-only classification"
        elif not runtime["cudaAvailable"] and SETTINGS.hf_allow_cpu:
            effective_mode = "hf-cpu"
            warning = "CUDA is unavailable, hf classifier will run on CPU and may be very slow"

    return {
        "effectiveMode": effective_mode,
        "warning": warning,
        **runtime,
    }


def json_utf8_response(payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        content=payload,
        media_type="application/json",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )


def clean_result(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        first_line_break = cleaned.find("\n")
        if first_line_break > 0:
            cleaned = cleaned[first_line_break + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    if "<" in cleaned and ">" in cleaned:
        cleaned = re.sub(r"<[^>]+>", "", cleaned).strip()

    first_object = cleaned.find("{")
    first_array = cleaned.find("[")
    starts = [index for index in (first_object, first_array) if index >= 0]
    if not starts:
        return None

    cleaned = cleaned[min(starts):].strip()
    if not (cleaned.startswith("{") or cleaned.startswith("[")):
        return None
    return cleaned


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def extract_first_json_block(text: str) -> str:
    if not text:
        return ""

    start = -1
    stack: list[str] = []
    in_string = False
    escape = False

    for index, ch in enumerate(text):
        if start < 0 and ch in "[{":
            start = index
            stack.append("]" if ch == "[" else "}")
            continue
        if start < 0:
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in "[{":
            stack.append("]" if ch == "[" else "}")
            continue

        if ch in "]}":
            if not stack or ch != stack[-1]:
                continue
            stack.pop()
            if not stack:
                return text[start:index + 1]

    return text


def normalize_json_like_text(text: str) -> str:
    normalized = text
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u3000": " ",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def collect_nested_object_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    starts: list[int] = []
    in_string = False
    escape = False

    for index, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            starts.append(index)
            continue

        if ch == "}" and starts:
            start = starts.pop()
            snippets.append(text[start:index + 1])

    return snippets


def first_non_blank(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def canonical_problem_item(item: Any) -> dict[str, str]:
    if not isinstance(item, dict):
        return {KEY_PROBLEM: "", KEY_RAW_SUMMARY: "", KEY_EXPLAIN: ""}
    return {
        KEY_PROBLEM: first_non_blank(item.get(KEY_PROBLEM), item.get("\u95ee\u9898\u5185\u5bb9"), item.get(KEY_TARGET_PROBLEM)),
        KEY_RAW_SUMMARY: first_non_blank(item.get(KEY_RAW_SUMMARY), item.get("\u539f\u6587\u6458\u8981\u5185\u5bb9"), item.get("\u6458\u8981"), item.get("\u539f\u6587")),
        KEY_EXPLAIN: first_non_blank(item.get(KEY_EXPLAIN), item.get("\u89e3\u8bfb"), item.get("\u5206\u6790"), item.get("\u8bf4\u660e")),
    }


def normalize_problem_payload(payload: Any) -> list[dict[str, str]]:
    if payload is None:
        return []
    if isinstance(payload, str):
        cleaned = clean_result(payload)
        if cleaned is None:
            return []
        try:
            payload = json.loads(cleaned)
        except Exception:
            return []

    if isinstance(payload, dict) and KEY_LIST in payload:
        payload = payload[KEY_LIST]

    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in payload:
        canonical = canonical_problem_item(item)
        if any(canonical.values()):
            normalized.append(canonical)
    return normalized


def normalize_classify_result(payload: Any, fallback_problem: dict[str, str] | None = None) -> list[dict[str, str]]:
    if payload is None:
        return []
    if isinstance(payload, str):
        cleaned = clean_result(payload)
        if cleaned is None:
            return []
        try:
            payload = json.loads(cleaned)
        except Exception:
            return []

    if isinstance(payload, dict):
        if KEY_CLASSIFY_LIST in payload:
            payload = payload[KEY_CLASSIFY_LIST]
        elif "data" in payload and isinstance(payload["data"], (list, dict)):
            payload = payload["data"]
        else:
            payload = [payload]

    if not isinstance(payload, list):
        return []

    fallback_problem = fallback_problem or {KEY_PROBLEM: "", KEY_RAW_SUMMARY: "", KEY_EXPLAIN: ""}
    normalized: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        record = {
            KEY_TARGET_PROBLEM: first_non_blank(item.get(KEY_TARGET_PROBLEM), item.get(KEY_PROBLEM), item.get("\u95ee\u9898\u5185\u5bb9"), fallback_problem.get(KEY_PROBLEM)),
            KEY_MAJOR_CODE: first_non_blank(item.get(KEY_MAJOR_CODE), item.get("\u5927\u7c7b\u7f16\u53f7"), item.get("\u4e00\u7ea7\u5206\u7c7b\u7f16\u53f7"), "00"),
            KEY_MAJOR_NAME: first_non_blank(item.get(KEY_MAJOR_NAME), item.get("\u5927\u7c7b\u540d\u79f0"), item.get("\u4e00\u7ea7\u5206\u7c7b\u540d\u79f0"), "\u65b0\u5206\u7c7b"),
            KEY_MINOR_CODE: first_non_blank(item.get(KEY_MINOR_CODE), item.get("\u5c0f\u7c7b\u7f16\u53f7"), item.get("\u4e8c\u7ea7\u5206\u7c7b\u7f16\u53f7"), "000"),
            KEY_MINOR_NAME: first_non_blank(item.get(KEY_MINOR_NAME), item.get("\u5c0f\u7c7b\u540d\u79f0"), item.get("\u4e8c\u7ea7\u5206\u7c7b\u540d\u79f0"), "\u65b0\u5206\u7c7b"),
            KEY_ANSWER: first_non_blank(item.get(KEY_ANSWER), item.get("\u5ba2\u670d\u56de\u590d"), item.get("\u56de\u7b54")),
            KEY_RAW_SUMMARY: first_non_blank(item.get(KEY_RAW_SUMMARY), item.get("\u539f\u6587\u6458\u8981\u5185\u5bb9"), item.get("\u6458\u8981"), fallback_problem.get(KEY_RAW_SUMMARY)),
            KEY_EXPLAIN: first_non_blank(item.get(KEY_EXPLAIN), item.get("\u89e3\u8bfb"), item.get("\u5206\u6790"), item.get("\u8bf4\u660e"), fallback_problem.get(KEY_EXPLAIN)),
        }
        if any(record.values()):
            normalized.append(record)
    return normalized


def salvage_classification_items(text: str, fallback_problem: dict[str, str] | None = None) -> list[dict[str, str]]:
    normalized_text = normalize_json_like_text(text)
    salvaged: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for snippet in collect_nested_object_snippets(normalized_text):
        try:
            parsed = json.loads(snippet)
        except Exception:
            continue

        for item in normalize_classify_result([parsed], fallback_problem):
            key = (
                item.get(KEY_TARGET_PROBLEM, ""),
                item.get(KEY_MAJOR_CODE, ""),
                item.get(KEY_MINOR_CODE, ""),
                item.get(KEY_ANSWER, ""),
            )
            if key in seen:
                continue
            seen.add(key)
            salvaged.append(item)

    return salvaged


def normalize_problem_input(problem: str | None, problems: Any | None) -> tuple[str, list[dict[str, str]]]:
    problem_text = str(problem or "").strip()
    if problems is not None:
        normalized = normalize_problem_payload(problems)
        if normalized:
            if not problem_text:
                problem_text = json.dumps(problems, ensure_ascii=False)
            return problem_text, normalized

    if problem_text:
        normalized = normalize_problem_payload(problem_text)
        return problem_text, normalized

    return "", []


def split_dialogue(info: str) -> list[dict[str, Any]]:
    compact = info.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(
        r"(\u5ba2\u670d|\u5ba2\u6237|\u7528\u6237)[:\uff1a]\s*(.*?)(?=\n(?:\u5ba2\u670d|\u5ba2\u6237|\u7528\u6237)[:\uff1a]|$)",
        re.S,
    )
    segments: list[dict[str, Any]] = []

    for order, match in enumerate(pattern.finditer(compact)):
        speaker = match.group(1).strip()
        text = match.group(2).strip()
        if text:
            segments.append({"speaker": speaker, "text": text, "order": order})

    return segments


def tokenize_text(text: str) -> Counter[str]:
    lowered = text.lower()
    tokens: list[str] = []

    for token in re.findall(r"[a-z0-9]{2,}", lowered):
        tokens.append(token)

    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        tokens.append(chunk)
        for size in (2, 3, 4):
            if len(chunk) < size:
                continue
            for index in range(len(chunk) - size + 1):
                tokens.append(chunk[index:index + size])

    return Counter(tokens)


@dataclass(slots=True)
class KnowledgeItem:
    code: str
    major_code: str
    minor_code: str
    major_title: str
    minor_title: str
    source: str
    body: str
    retrieval_text: str
    token_counts: Counter[str] = field(default_factory=Counter)


@dataclass(slots=True)
class RetrievalHit:
    item: KnowledgeItem
    score: float


def resolve_knowledge_dir() -> Path | None:
    candidates: list[Path] = []

    if SETTINGS.knowledge_dir:
        candidates.append(Path(SETTINGS.knowledge_dir))

    here = Path(__file__).resolve()
    candidates.extend([
        here.parents[1] / "knowledge" / "CustomerObjectionClassification-md",
        here.parents[1] / "document" / "CustomerObjectionClassification-md",
        Path.cwd() / "document" / "CustomerObjectionClassification-md",
        Path.cwd() / "src" / "main" / "resources" / "document" / "CustomerObjectionClassification-md",
        Path(r"D:\project\mutliAgent\multi-agent-service\src\main\resources\document\CustomerObjectionClassification-md"),
    ])

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


class KnowledgeRetriever:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._knowledge_dir: Path | None = None
        self._load_error: str | None = None
        self._items: list[KnowledgeItem] = []
        self._idf: dict[str, float] = {}

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def knowledge_dir(self) -> str | None:
        return str(self._knowledge_dir) if self._knowledge_dir else None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def item_count(self) -> int:
        return len(self._items)

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                knowledge_dir = resolve_knowledge_dir()
                if knowledge_dir is None:
                    raise FileNotFoundError("classification knowledge directory not found")

                items: list[KnowledgeItem] = []
                for path in sorted(knowledge_dir.glob("Classification-*.md")):
                    items.extend(self._parse_markdown_file(path))

                if not items:
                    raise ValueError(f"no knowledge entries found in {knowledge_dir}")

                doc_freq: Counter[str] = Counter()
                for item in items:
                    item.token_counts = tokenize_text(item.retrieval_text)
                    for token in item.token_counts.keys():
                        doc_freq[token] += 1

                total_docs = len(items)
                idf = {
                    token: math.log((total_docs + 1.0) / (df + 0.5)) + 1.0
                    for token, df in doc_freq.items()
                }

                self._items = items
                self._idf = idf
                self._knowledge_dir = knowledge_dir
                self._load_error = None
                self._loaded = True
                logger.info("knowledge retriever loaded %s entries from %s", len(items), knowledge_dir)
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("failed to load classification knowledge")
                raise

    def _parse_markdown_file(self, path: Path) -> list[KnowledgeItem]:
        text = path.read_text(encoding="utf-8")
        sections = [section.strip() for section in re.split(r"\n-{3,}\n", text) if section.strip()]
        items: list[KnowledgeItem] = []

        for section in sections:
            header = ""
            header_match = re.search(r"^###\s*(.+)$", section, flags=re.M)
            if header_match:
                header = header_match.group(1).strip()

            fields: dict[str, str] = {}
            for line in section.splitlines():
                match = re.match(r"^-\s*([^:：]+)\s*[:：]\s*(.+?)\s*$", line.strip())
                if not match:
                    continue
                fields[match.group(1).strip()] = match.group(2).strip()

            code = fields.get(KEY_CODE, "")
            major_code = fields.get(KEY_BIG_CODE, "")
            minor_code = fields.get(KEY_SMALL_CODE, "")
            major_title = fields.get(KEY_BIG_TITLE, header)
            minor_title = fields.get(KEY_SMALL_TITLE, "")
            if not major_code or not minor_code or not major_title or not minor_title:
                continue

            retrieval_text = "\n".join([
                header,
                f"{KEY_CODE}: {code}",
                f"{KEY_BIG_CODE}: {major_code}",
                f"{KEY_SMALL_CODE}: {minor_code}",
                f"{KEY_BIG_TITLE}: {major_title}",
                f"{KEY_SMALL_TITLE}: {minor_title}",
                section,
            ])
            items.append(
                KnowledgeItem(
                    code=code,
                    major_code=major_code,
                    minor_code=minor_code,
                    major_title=major_title,
                    minor_title=minor_title,
                    source=path.name,
                    body=section,
                    retrieval_text=retrieval_text,
                )
            )

        return items

    def retrieve(self, query_text: str, top_k: int | None = None) -> list[RetrievalHit]:
        self.ensure_loaded()

        query_counts = tokenize_text(query_text)
        if not query_counts:
            return []

        top_k = top_k or SETTINGS.rag_top_k
        hits: list[RetrievalHit] = []
        lowered_query = query_text.lower()

        for item in self._items:
            overlap = set(query_counts.keys()) & set(item.token_counts.keys())
            if not overlap:
                continue

            score = 0.0
            for token in overlap:
                score += self._idf.get(token, 1.0) * min(query_counts[token], item.token_counts[token])

            minor_title_no_desc = re.split(r"[:：]", item.minor_title, maxsplit=1)[0].strip()
            if minor_title_no_desc and minor_title_no_desc in query_text:
                score += 6.0
            if item.major_title and item.major_title in query_text:
                score += 4.0

            problem_core = first_non_blank(query_text[:120])
            if problem_core and problem_core.lower() in item.retrieval_text.lower():
                score += 2.0

            if score > 0:
                hits.append(RetrievalHit(item=item, score=score))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


KNOWLEDGE_RETRIEVER = KnowledgeRetriever()


def retrieval_query(info: str, problem_item: dict[str, str]) -> str:
    return "\n".join([
        problem_item.get(KEY_PROBLEM, ""),
        problem_item.get(KEY_RAW_SUMMARY, ""),
        problem_item.get(KEY_EXPLAIN, ""),
        info[:300],
    ])


def knowledge_context(hits: list[RetrievalHit]) -> str:
    chunks: list[str] = []
    total_len = 0
    for index, hit in enumerate(hits, start=1):
        chunk = "\n".join([
            f"[知识 {index}]",
            f"{KEY_BIG_CODE}: {hit.item.major_code}",
            f"{KEY_BIG_TITLE}: {hit.item.major_title}",
            f"{KEY_SMALL_CODE}: {hit.item.minor_code}",
            f"{KEY_SMALL_TITLE}: {hit.item.minor_title}",
            f"得分: {hit.score:.4f}",
            f"来源: {hit.item.source}",
            "知识内容:",
            hit.item.body,
        ])
        if total_len + len(chunk) > SETTINGS.rag_context_limit and chunks:
            break
        chunks.append(chunk)
        total_len += len(chunk)
    return "\n\n".join(chunks)


def trace_hits(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    return [
        {
            "score": round(hit.score, 4),
            "majorCode": hit.item.major_code,
            "majorName": hit.item.major_title,
            "minorCode": hit.item.minor_code,
            "minorName": hit.item.minor_title,
            "source": hit.item.source,
            "bodyPreview": hit.item.body[:300],
        }
        for hit in hits
    ]


def split_title_description(text: str) -> tuple[str, str]:
    parts = re.split(r"[:：]", text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


def find_service_answer(info: str, problem_item: dict[str, str]) -> str:
    segments = split_dialogue(info)
    if not segments:
        return ""

    query = retrieval_query(info, problem_item)
    query_tokens = tokenize_text(query)
    summary = problem_item.get(KEY_RAW_SUMMARY, "")
    matched_index = -1
    best_score = -1.0

    for index, segment in enumerate(segments):
        if segment["speaker"] not in {CUSTOMER_LABEL, USER_LABEL}:
            continue
        score = 0.0
        if summary and summary in segment["text"]:
            score += 10.0
        segment_tokens = tokenize_text(segment["text"])
        overlap = set(query_tokens.keys()) & set(segment_tokens.keys())
        score += float(len(overlap))
        if score > best_score:
            best_score = score
            matched_index = index

    if matched_index >= 0:
        for segment in segments[matched_index + 1:]:
            if segment["speaker"] == SERVICE_LABEL:
                return segment["text"]

    best_answer = ""
    best_answer_score = 0.0
    for segment in segments:
        if segment["speaker"] != SERVICE_LABEL:
            continue
        segment_tokens = tokenize_text(segment["text"])
        overlap = set(query_tokens.keys()) & set(segment_tokens.keys())
        score = float(len(overlap))
        if score > best_answer_score:
            best_answer_score = score
            best_answer = segment["text"]
    return best_answer


def fallback_record(problem_item: dict[str, str], answer: str) -> dict[str, str]:
    return {
        KEY_TARGET_PROBLEM: problem_item.get(KEY_PROBLEM, ""),
        KEY_MAJOR_CODE: "00",
        KEY_MAJOR_NAME: "\u65b0\u5206\u7c7b",
        KEY_MINOR_CODE: "000",
        KEY_MINOR_NAME: "\u65b0\u5206\u7c7b",
        KEY_ANSWER: answer,
        KEY_RAW_SUMMARY: problem_item.get(KEY_RAW_SUMMARY, ""),
        KEY_EXPLAIN: problem_item.get(KEY_EXPLAIN, ""),
    }


def rag_record_from_hit(problem_item: dict[str, str], answer: str, hit: RetrievalHit | None) -> dict[str, str]:
    if hit is None or hit.score < SETTINGS.rag_min_score:
        return fallback_record(problem_item, answer)

    return {
        KEY_TARGET_PROBLEM: problem_item.get(KEY_PROBLEM, ""),
        KEY_MAJOR_CODE: hit.item.major_code,
        KEY_MAJOR_NAME: hit.item.major_title,
        KEY_MINOR_CODE: hit.item.minor_code,
        KEY_MINOR_NAME: hit.item.minor_title,
        KEY_ANSWER: answer,
        KEY_RAW_SUMMARY: problem_item.get(KEY_RAW_SUMMARY, ""),
        KEY_EXPLAIN: problem_item.get(KEY_EXPLAIN, ""),
    }


def mock_classify(info: str, problems: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    results: list[dict[str, str]] = []
    traces: list[dict[str, Any]] = []

    for problem_item in problems:
        query = retrieval_query(info, problem_item)
        hits = KNOWLEDGE_RETRIEVER.retrieve(query)
        answer = find_service_answer(info, problem_item)
        top_hit = hits[0] if hits else None
        results.append(rag_record_from_hit(problem_item, answer, top_hit))
        traces.append({
            "problem": problem_item.get(KEY_PROBLEM, ""),
            "query": query,
            "retrieved": trace_hits(hits),
        })

    return results, traces


def build_prompt(info: str, one_problem_json: str, context: str) -> str:
    return f"""
/no_think
You are a telecom objection classification assistant.
Classify the single customer problem using the retrieved knowledge.

Rules:
1. Output must be a pure JSON array.
2. Each item must contain these exact fields:
   "{KEY_TARGET_PROBLEM}", "{KEY_MAJOR_CODE}", "{KEY_MAJOR_NAME}", "{KEY_MINOR_CODE}", "{KEY_MINOR_NAME}", "{KEY_ANSWER}", "{KEY_RAW_SUMMARY}", "{KEY_EXPLAIN}".
3. "{KEY_TARGET_PROBLEM}" must stay exactly the same as the input problem.
4. Prefer the retrieved knowledge. If there is no suitable match, output "{KEY_MAJOR_CODE}" as "00" and "{KEY_MAJOR_NAME}" as "新分类".
5. "{KEY_ANSWER}" should copy the matching service response from <info>. If not found, leave it empty.
6. Do not output any explanation outside JSON.

<knowledge>
{context if context else "No retrieved knowledge."}
</knowledge>

<info>
{info}
</info>

<problem>
{one_problem_json}
</problem>
""".strip()


def build_java_aligned_prompt(info: str, one_problem_json: str, context: str) -> str:
    return f"""
/no_think
你是一个电信公司的客服总管。你将对一段客服与客户对话进行分析。
你的主要任务是：根据对话内容，以及已经抽取好的单个客户问题，结合 RAG 检索到的“客户异议分类”知识，
对该问题进行精确归类，并从 <info> 中找到与该问题对应的客服回答。

其中：
- 对话全文放在 <info></info> 标签中；
- 当前待分类的问题放在 <problem></problem> 标签中；
- 客户异议分类知识通过 <knowledge></knowledge> 标签注入。

<knowledge>
{context if context else "没有检索到可用的分类知识。"}
</knowledge>

<info>
{info}
</info>

<problem>
{one_problem_json}
</problem>

输出要求：
1. 输出格式必须是一个 JSON 数组，例如：
   [
     {{
       "{KEY_TARGET_PROBLEM}": "",
       "{KEY_MAJOR_CODE}": "",
       "{KEY_MAJOR_NAME}": "",
       "{KEY_MINOR_CODE}": "",
       "{KEY_MINOR_NAME}": "",
       "{KEY_ANSWER}": "",
       "{KEY_RAW_SUMMARY}": "",
       "{KEY_EXPLAIN}": ""
     }}
   ]
2. "{KEY_TARGET_PROBLEM}" 必须与输入问题保持完全一致，不允许改写。
3. 大类/小类编号和名称必须直接复制检索知识中的原文，尤其不要省略冒号后的说明。
4. 如果没有匹配项，输出 "{KEY_MAJOR_CODE}" 为 "00"、"{KEY_MAJOR_NAME}" 为 "新分类"，
   "{KEY_MINOR_CODE}" 为 "000"、"{KEY_MINOR_NAME}" 为 "新分类"。
5. "{KEY_ANSWER}" 必须尽量从 <info> 中找到并原样复制对应客服回复；找不到则输出空字符串，严禁编造。
6. "{KEY_RAW_SUMMARY}"、"{KEY_EXPLAIN}" 必须从输入问题中复制，不允许丢失。
7. 严禁输出 JSON 以外的解释、注释、Markdown 代码块、思考过程或 <think> 标签。
""".strip()


class HFClassifier:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._tokenizer = None
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _resolve_dtype(self, torch_module):
        if SETTINGS.torch_dtype == "float16":
            return torch_module.float16
        if SETTINGS.torch_dtype == "float32":
            return torch_module.float32
        if SETTINGS.torch_dtype == "bfloat16":
            return torch_module.bfloat16
        return torch_module.float16 if torch_module.cuda.is_available() else torch_module.float32

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                if not SETTINGS.base_model_dir:
                    raise ValueError("BASE_MODEL_DIR is not configured")

                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
                import torch

                if not torch.cuda.is_available() and not SETTINGS.hf_allow_cpu:
                    raise RuntimeError(
                        "CUDA is unavailable and HF_ALLOW_CPU is false; "
                        "refusing to load qwen3-8b on CPU"
                    )

                logger.info("Loading tokenizer from %s", SETTINGS.base_model_dir)
                tokenizer = AutoTokenizer.from_pretrained(
                    SETTINGS.base_model_dir,
                    trust_remote_code=SETTINGS.trust_remote_code,
                )

                model_kwargs = {
                    "torch_dtype": self._resolve_dtype(torch),
                    "trust_remote_code": SETTINGS.trust_remote_code,
                }
                if torch.cuda.is_available():
                    model_kwargs["device_map"] = "auto"

                logger.info("Loading base model from %s", SETTINGS.base_model_dir)
                base_model = AutoModelForCausalLM.from_pretrained(
                    SETTINGS.base_model_dir,
                    **model_kwargs,
                )

                logger.info("Loading adapter from %s", SETTINGS.adapter_dir)
                model = PeftModel.from_pretrained(base_model, SETTINGS.adapter_dir)
                model.eval()
                if not torch.cuda.is_available():
                    model = model.cpu()

                self._tokenizer = tokenizer
                self._model = model
                self._loaded = True
                self._load_error = None
                logger.info("HF classifier loaded successfully")
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("Failed to load HF classifier")
                raise

    def _generate_one(self, info: str, problem_item: dict[str, str], hits: list[RetrievalHit]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        self.ensure_loaded()

        import torch

        one_problem_json = json.dumps(problem_item, ensure_ascii=False)
        context = knowledge_context(hits)
        prompt = build_java_aligned_prompt(info, one_problem_json, context)
        tokenizer = self._tokenizer
        model = self._model

        messages = [
            {
                "role": "system",
                "content": "你是问题分类助手。你只能输出合法 JSON，不能输出任何额外解释。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        last_raw_text = ""
        last_error = None
        fallback = rag_record_from_hit(problem_item, find_service_answer(info, problem_item), hits[0] if hits else None)

        for attempt in range(SETTINGS.max_retries + 1):
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=SETTINGS.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )

                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                last_raw_text = raw_text
                logger.info("hf classify raw output preview=%s", raw_text[:1000] if raw_text else "<empty>")

                sanitized = normalize_json_like_text(raw_text)
                cleaned = extract_first_json_block(clean_json_text(sanitized))
                if cleaned:
                    parsed = json.loads(cleaned)
                    normalized = normalize_classify_result(parsed, problem_item)
                    if normalized:
                        return normalized, {"raw": raw_text, "retrieved": trace_hits(hits)}

                salvaged = salvage_classification_items(sanitized, problem_item)
                if salvaged:
                    logger.warning("hf classify salvaged %s items from partial output", len(salvaged))
                    return salvaged, {"raw": raw_text, "retrieved": trace_hits(hits)}

                raise ValueError("no classification items extracted from model output")
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "hf classify parse failed on attempt %s/%s: %s",
                    attempt + 1,
                    SETTINGS.max_retries + 1,
                    exc,
                )
                if attempt >= SETTINGS.max_retries:
                    break

        logger.warning("hf classify falling back to retrieved knowledge: %s", last_error)
        return [fallback], {"raw": last_raw_text, "retrieved": trace_hits(hits), "fallback": True}

    def classify(self, info: str, problems: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        all_results: list[dict[str, str]] = []
        traces: list[dict[str, Any]] = []

        for problem_item in problems:
            query = retrieval_query(info, problem_item)
            hits = KNOWLEDGE_RETRIEVER.retrieve(query)
            one_result, trace = self._generate_one(info, problem_item, hits)
            trace["problem"] = problem_item.get(KEY_PROBLEM, "")
            trace["query"] = query
            all_results.extend(one_result)
            traces.append(trace)

        return all_results, traces


HF_CLASSIFIER = HFClassifier()


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        KNOWLEDGE_RETRIEVER.ensure_loaded()
    except Exception:
        pass

    runtime = hf_execution_state()

    payload = {
        "status": "ok",
        "mode": SETTINGS.mode,
        "effectiveMode": runtime["effectiveMode"],
        "warning": runtime["warning"],
        "baseModelDir": SETTINGS.base_model_dir,
        "baseModelExists": bool(SETTINGS.base_model_dir and Path(SETTINGS.base_model_dir).exists()),
        "adapterDir": SETTINGS.adapter_dir,
        "adapterExists": bool(SETTINGS.adapter_dir and Path(SETTINGS.adapter_dir).exists()),
        "hfAllowCpu": SETTINGS.hf_allow_cpu,
        "torchAvailable": runtime["torchAvailable"],
        "cudaAvailable": runtime["cudaAvailable"],
        "deviceName": runtime["deviceName"],
        "torchError": runtime["error"],
        "loaded": HF_CLASSIFIER.loaded,
        "loadError": HF_CLASSIFIER.load_error,
        "knowledgeDir": KNOWLEDGE_RETRIEVER.knowledge_dir,
        "knowledgeLoaded": KNOWLEDGE_RETRIEVER.loaded,
        "knowledgeLoadError": KNOWLEDGE_RETRIEVER.load_error,
        "knowledgeCount": KNOWLEDGE_RETRIEVER.item_count,
        "ragTopK": SETTINGS.rag_top_k,
        "ragMinScore": SETTINGS.rag_min_score,
    }
    return json_utf8_response(payload)


@app.post("/classify")
async def classify(request: Request, req: ClassifyRequest | None = None) -> dict[str, Any]:
    start = time.time()
    try:
        KNOWLEDGE_RETRIEVER.ensure_loaded()

        raw_text = ""
        info = ""
        problem_text = ""
        problems_payload: Any = None

        if req is not None:
            info = req.info or ""
            problem_text = str(req.problem or "").strip()
            problems_payload = req.problems
        else:
            raw_bytes = await request.body()
            raw_text = raw_bytes.decode("utf-8", errors="ignore").strip()
            if raw_text:
                try:
                    payload = json.loads(raw_text)
                    if isinstance(payload, dict):
                        info = str(payload.get("info", "")).strip()
                        problem_text = str(payload.get("problem", "")).strip()
                        problems_payload = payload.get("problems")
                    elif isinstance(payload, str):
                        info = payload.strip()
                except Exception:
                    info = raw_text

        normalized_problem_text, problems = normalize_problem_input(problem_text, problems_payload)

        logger.info("classify request raw body: %s", raw_text if raw_text else "<empty>")
        logger.info(
            "classify request parsed info length=%s preview=%s",
            len(info),
            info[:120] if info else "<empty>",
        )
        logger.info("classify request problem preview=%s", normalized_problem_text[:500] if normalized_problem_text else "<empty>")
        logger.info("classify request problem count=%s", len(problems))

        if SETTINGS.mode == "mock":
            data, traces = mock_classify(info, problems)
            response = {
                "code": 0,
                "message": "ok",
                "data": data,
                "model": "local-rag-mock-classifier",
                "effectiveMode": "mock",
                "latencyMs": int((time.time() - start) * 1000),
            }
            if SETTINGS.return_raw:
                response["raw"] = traces
            return json_utf8_response(response)

        if SETTINGS.mode != "hf":
            return json_utf8_response({
                "code": 1,
                "message": f"unsupported CLASSIFIER_MODE: {SETTINGS.mode}",
                "data": [],
                "model": SETTINGS.mode,
                "effectiveMode": SETTINGS.mode,
                "latencyMs": int((time.time() - start) * 1000),
            })

        runtime = hf_execution_state()
        if runtime["effectiveMode"] == "rag-fallback":
            logger.warning("hf mode fallback activated: %s", runtime["warning"])
            data, traces = mock_classify(info, problems)
            response = {
                "code": 0,
                "message": runtime["warning"] or "ok",
                "data": data,
                "model": "local-rag-cpu-fallback-classifier",
                "effectiveMode": runtime["effectiveMode"],
                "latencyMs": int((time.time() - start) * 1000),
            }
            if SETTINGS.return_raw:
                response["raw"] = traces
            return json_utf8_response(response)

        data, traces = HF_CLASSIFIER.classify(info, problems)
        response = {
            "code": 0,
            "message": "ok",
            "data": data,
            "model": "qwen3-8b-lora-rag-classification",
            "effectiveMode": runtime["effectiveMode"],
            "latencyMs": int((time.time() - start) * 1000),
        }
        if SETTINGS.return_raw:
            response["raw"] = traces
        return json_utf8_response(response)
    except Exception as exc:
        logger.exception("Classification failed")
        return json_utf8_response({
            "code": 1,
            "message": f"classify_failed: {exc}",
            "data": [],
            "model": SETTINGS.mode,
            "effectiveMode": SETTINGS.mode,
            "latencyMs": int((time.time() - start) * 1000),
        })


if __name__ == "__main__":
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port)
