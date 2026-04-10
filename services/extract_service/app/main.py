# -*- coding: utf-8 -*-

import json
import logging
import os
import re
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("problem-service")


class ExtractRequest(BaseModel):
    info: str


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    mode = os.getenv("EXTRACTOR_MODE", "mock").strip().lower()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "9001"))
    base_model_dir = os.getenv("BASE_MODEL_DIR", "").strip()
    adapter_dir = os.getenv("ADAPTER_DIR", r"E:\hangdong_tool\model\extraction").strip()
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    top_p = float(os.getenv("TOP_P", "1.0"))
    trust_remote_code = env_bool("TRUST_REMOTE_CODE", True)
    return_raw = env_bool("RETURN_RAW", False)
    torch_dtype = os.getenv("TORCH_DTYPE", "auto").strip().lower()
    max_retries = int(os.getenv("MAX_RETRIES", "1"))


SETTINGS = Settings()
app = FastAPI(title="Problem Extraction Service", version="1.0.0")


def json_utf8_response(payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        content=payload,
        media_type="application/json",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )


def build_prompt(info: str) -> str:
    return f"""
/no_think
你是一个电信公司的客服总管，你将对一段客服与客户对话录音进行分析。你的主要任务是，精准分析该会话中2-3种最明显的客户提出的异议问题，并给出提出问题的原文语句内容和解释。
=======================
【客服与客户的录音】
{info}
=======================

你的任务：
1、必须给出2-3个最明显的客户异议问题，并针对每一个问题给出对应的原文摘要和解释。
2、无异议与其他异议互斥，有其他异议时，不能判断为无异议，没有其他异议时，则可判断为无异议。若客户没有提问，则直接判定为无问题。
3、输出结果必须是一个合法的 JSON 对象，且顶层只有一个字段：`异议列表`。`异议列表` 的值是一个长度为 1–3 的 JSON 数组。数组中的每个元素是一个对象，包含字段：`问题`、`原文摘要`、`解释`。
4、JSON 中必须全部使用半角英文标点（如逗号`,`和冒号`:`），不得使用中文全角标点。
5、严禁使用 ```json 或任何 Markdown 代码块包裹输出，不能输出解释性文字，只能输出纯 JSON。

=======================
【严格输出格式，必须是 JSON 数组】
[
  {{
    "问题": "",
    "原文摘要": "",
    "解释": ""
  }}
]
=======================

绝对禁止输出注释、额外说明、自然语言，只能输出纯 JSON。
""".strip()


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    first_object = cleaned.find("{")
    first_array = cleaned.find("[")
    starts = [index for index in (first_object, first_array) if index >= 0]
    if not starts:
        return ""
    return cleaned[min(starts):].strip()


def extract_first_json_block(text: str) -> str:
    if not text:
        return ""
    start = -1
    stack = []
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
        "\u201c": "'",
        "\u201d": "'",
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

        if ch == '{':
            starts.append(index)
            continue

        if ch == '}' and starts:
            start = starts.pop()
            snippets.append(text[start:index + 1])

    return snippets


def salvage_problem_items(text: str) -> list[dict[str, str]]:
    normalized = normalize_json_like_text(text)
    list_key = "\u5f02\u8bae\u5217\u8868"
    search_text = normalized

    marker = normalized.find(list_key)
    if marker >= 0:
        array_start = normalized.find("[", marker)
        if array_start >= 0:
            search_text = normalized[array_start + 1:]
    else:
        array_start = normalized.find("[")
        if array_start >= 0:
            search_text = normalized[array_start + 1:]

    salvaged: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for snippet in collect_nested_object_snippets(search_text):
        try:
            parsed = json.loads(snippet)
        except Exception:
            continue

        for item in normalize_result([parsed]):
            key = (
                item.get("\u95ee\u9898", ""),
                item.get("\u539f\u6587\u6458\u8981", ""),
                item.get("\u89e3\u91ca", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            salvaged.append(item)
            if len(salvaged) >= 3:
                return salvaged

    return salvaged



def normalize_result(parsed: Any) -> list[dict[str, str]]:
    if isinstance(parsed, dict) and "异议列表" in parsed:
        parsed = parsed["异议列表"]

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        problem = str(item.get("问题", item.get("问题内容", item.get("针对的问题", "")))).strip()
        raw_summary = str(item.get("原文摘要", item.get("原文摘要内容", item.get("摘要", "")))).strip()
        explain = str(item.get("解释", item.get("解读", item.get("分析", item.get("说明", ""))))).strip()
        if problem or raw_summary or explain:
            normalized.append({
                "问题": problem,
                "原文摘要": raw_summary,
                "解释": explain,
            })
    return normalized


def customer_lines(info: str) -> list[str]:
    compact = info.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(r"(客户|用户)[:：]\s*(.*?)(?=(客服|客户|用户)[:：]|$)", re.S)
    lines = []
    for match in pattern.finditer(compact):
        text = match.group(2).strip()
        if text:
            lines.append(f"客户：{text}")
    return lines


def summarize_issue(line: str) -> dict[str, str]:
    text = re.sub(r"^(客户|用户)[:：]\s*", "", line).strip()
    if "收费" in text or "资费" in text or "扣费" in text:
        return {"问题": "是否会额外收费", "解释": "客户确认后续是否存在额外资费"}
    if "5G" in text or "升级" in text:
        return {"问题": "套餐升级是否真实有效", "解释": "客户确认套餐升级或网络升级的真实性与内容"}
    if "流量" in text:
        return {"问题": "流量权益如何变化", "解释": "客户询问流量包、流量权益或使用规则"}
    if "公众号" in text or "微信" in text:
        return {"问题": "公众号相关操作如何进行", "解释": "客户咨询公众号或微信渠道操作"}
    if "为什么" in text:
        return {"问题": "客户对业务办理原因存疑", "解释": "客户对客服所述业务内容或原因存在疑问"}
    return {"问题": text[:30], "解释": "客户对当前业务内容提出疑问"}


def mock_extract(info: str) -> list[dict[str, str]]:
    candidate_lines = customer_lines(info)
    selected: list[dict[str, str]] = []
    seen = set()

    for line in candidate_lines:
        if not any(token in line for token in ("?", "？", "收费", "资费", "扣费", "为什么", "5G", "升级", "流量", "公众号", "微信")):
            continue
        issue = summarize_issue(line)
        key = issue["问题"]
        if key in seen:
            continue
        seen.add(key)
        selected.append({
            "问题": issue["问题"],
            "原文摘要": re.sub(r"^(客户|用户)[:：]\s*", "", line).strip(),
            "解释": issue["解释"],
        })
        if len(selected) >= 3:
            break

    if selected:
        return selected

    if candidate_lines:
        last_line = re.sub(r"^(客户|用户)[:：]\s*", "", candidate_lines[-1]).strip()
        issue = summarize_issue(candidate_lines[-1])
        return [{
            "问题": issue["问题"] if issue["问题"] else (last_line[:30] if last_line else "无明显问题"),
            "原文摘要": last_line,
            "解释": issue["解释"] if issue["解释"] else "mock 模式下未识别到明显疑问，回退到最后一句客户表达",
        }]

    plain_text = info.strip()
    if plain_text:
        fallback_text = plain_text[:50]
        return [{
            "问题": "客户对当前业务内容存在疑问",
            "原文摘要": fallback_text,
            "解释": "mock 模式兜底返回，用于验证 Java 与 Python 联调链路",
        }]

    return []


class HFExtractor:
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
                logger.info("HF extractor loaded successfully")
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("Failed to load HF extractor")
                raise

    def extract(self, info: str) -> tuple[list[dict[str, str]], str]:
        self.ensure_loaded()

        import torch

        prompt = build_prompt(info)
        tokenizer = self._tokenizer
        model = self._model

        messages = [
            {
                "role": "system",
                "content": "你是一个问题提取助手。请严格按要求输出纯 JSON，不要输出额外说明。",
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
                logger.info("hf raw output preview=%s", raw_text[:1000] if raw_text else "<empty>")

                sanitized = normalize_json_like_text(raw_text)
                cleaned = extract_first_json_block(clean_json_text(sanitized))
                if cleaned:
                    parsed = json.loads(cleaned)
                    normalized = normalize_result(parsed)
                    if normalized:
                        return normalized, raw_text

                salvaged = salvage_problem_items(sanitized)
                if salvaged:
                    logger.warning("hf salvaged %s problem items from partial output", len(salvaged))
                    return salvaged, raw_text

                raise ValueError("no problem items extracted from model output")
            except Exception as exc:
                salvaged = salvage_problem_items(last_raw_text)
                if salvaged:
                    logger.warning(
                        "hf output parse failed on attempt %s/%s but salvaged %s items: %s",
                        attempt + 1,
                        SETTINGS.max_retries + 1,
                        len(salvaged),
                        exc,
                    )
                    return salvaged, last_raw_text

                last_error = exc
                logger.warning(
                    "hf output parse failed on attempt %s/%s: %s",
                    attempt + 1,
                    SETTINGS.max_retries + 1,
                    exc,
                )
                if attempt >= SETTINGS.max_retries:
                    break

        raise ValueError(f"{last_error}; raw={last_raw_text[:2000]}")


HF_EXTRACTOR = HFExtractor()


@app.get("/health")
def health() -> dict[str, Any]:
    payload = {
        "status": "ok",
        "mode": SETTINGS.mode,
        "baseModelDir": SETTINGS.base_model_dir,
        "adapterDir": SETTINGS.adapter_dir,
        "loaded": HF_EXTRACTOR.loaded,
        "loadError": HF_EXTRACTOR.load_error,
    }
    return json_utf8_response(payload)


@app.post("/extract")
async def extract(request: Request, req: ExtractRequest | None = None) -> dict[str, Any]:
    start = time.time()
    try:
        raw_text = ""
        info = ""
        if req is not None:
            info = req.info
        else:
            raw_bytes = await request.body()
            raw_text = raw_bytes.decode("utf-8", errors="ignore").strip()
            if raw_text:
                try:
                    payload = json.loads(raw_text)
                    if isinstance(payload, dict):
                        info = str(payload.get("info", "")).strip()
                    elif isinstance(payload, str):
                        info = payload.strip()
                except Exception:
                    info = raw_text

        logger.info("extract request raw body: %s", raw_text if raw_text else "<empty>")
        logger.info(
            "extract request parsed info length=%s preview=%s",
            len(info),
            info[:120] if info else "<empty>",
        )

        if SETTINGS.mode == "mock":
            data = mock_extract(info)
            response = {
                "code": 0,
                "message": "ok",
                "data": data,
                "model": "mock-extractor",
                "latencyMs": int((time.time() - start) * 1000),
            }
            return json_utf8_response(response)

        if SETTINGS.mode != "hf":
            return json_utf8_response({
                "code": 1,
                "message": f"unsupported EXTRACTOR_MODE: {SETTINGS.mode}",
                "data": [],
                "model": SETTINGS.mode,
                "latencyMs": int((time.time() - start) * 1000),
            })

        data, raw_text = HF_EXTRACTOR.extract(info)
        response = {
            "code": 0,
            "message": "ok",
            "data": data,
            "model": "qwen3-8b-lora-extraction",
            "latencyMs": int((time.time() - start) * 1000),
        }
        if SETTINGS.return_raw:
            response["raw"] = raw_text
        return json_utf8_response(response)
    except Exception as exc:
        logger.exception("Extraction failed")
        return json_utf8_response({
            "code": 1,
            "message": f"extract_failed: {exc}",
            "data": [],
            "model": SETTINGS.mode,
            "latencyMs": int((time.time() - start) * 1000),
        })


if __name__ == "__main__":
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port)
