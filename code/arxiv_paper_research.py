#!/usr/bin/env python3
"""
arXiv paper research helper.

Pipeline:
1. Use an LLM to rewrite a user's rough research interest into arXiv-friendly
   English search terms.
2. Query the arXiv Atom API and collect title/abstract/metadata.
3. Use the LLM again to produce Chinese summaries, key points, and reusable
   innovation ideas.
4. Save a JSON file that can be rendered by arxiv_papers_viewer.html.

The LLM client uses OpenAI-compatible Chat Completions endpoints. It works with
providers such as DeepSeek or OpenAI when the corresponding base URL/model/key
are supplied through environment variables or CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

DEFAULT_MAX_RESULTS = 30
DEFAULT_REQUEST_DELAY = 3.0
DEFAULT_MIN_PER_QUERY = 50


def resolve_candidate_results(args: argparse.Namespace) -> int:
    return args.candidate_results or args.max_results


def resolve_per_query_limit(args: argparse.Namespace, query_count: int) -> int:
    if args.per_query_max:
        return args.per_query_max
    candidate_results = resolve_candidate_results(args)
    # Fetch a little more than the exact split because different query groups
    # often overlap. This keeps "300 candidates -> 30 final" useful without
    # forcing the user to hand-tune per-query limits.
    auto_limit = (candidate_results * 5 + max(query_count, 1) * 4 - 1) // (max(query_count, 1) * 4)
    return max(DEFAULT_MIN_PER_QUERY, auto_limit)


class ProgressReporter:
    def __init__(self, path: Path, interest: str):
        self.path = path
        self.data: Dict[str, Any] = {
            "status": "running",
            "stage": "starting",
            "message": "任务准备中",
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "user_interest": interest,
            "search_plan": None,
            "search": {
                "current_query_index": 0,
                "total_queries": 0,
                "current_query_name": "",
                "current_query_arxiv_query": "",
                "current_query_fetched": 0,
                "current_query_limit": 0,
                "current_query_total": 0,
                "total_unique_papers": 0,
                "queries": [],
            },
            "analysis": {
                "current_batch": 0,
                "total_batches": 0,
                "batch_size": 0,
                "processed_papers": 0,
                "total_papers": 0,
                "current_titles": [],
                "batches": [],
            },
            "screening": {
                "current_batch": 0,
                "total_batches": 0,
                "batch_size": 0,
                "processed_papers": 0,
                "total_papers": 0,
                "selected_papers": 0,
                "current_titles": [],
                "batches": [],
            },
            "refinement": {
                "enabled": False,
                "current_round": 0,
                "total_rounds": 0,
                "rounds": [],
            },
            "raw_output": "",
            "final_output": "",
            "error": "",
        }
        self.write()

    def patch(self, **updates: Any) -> None:
        deep_update(self.data, updates)
        self.data["updated_at"] = now_iso()
        self.write()

    def stage(self, stage: str, message: str) -> None:
        self.patch(stage=stage, message=message)

    def finish(self, message: str, final_output: Path) -> None:
        self.patch(
            status="completed",
            stage="done",
            message=message,
            final_output=str(final_output),
        )

    def fail(self, message: str) -> None:
        self.patch(status="failed", message=message, error=message)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        temp_path.replace(self.path)


def deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return content_to_text(content.get("text"))
        if "content" in content:
            return content_to_text(content.get("content"))
        if "output_text" in content:
            return content_to_text(content.get("output_text"))
        if content.get("type") == "text" and "text" in content:
            return content_to_text(content.get("text"))
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def shorten_text(text: str, limit: int = 500) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def extract_json_from_text(text: Any) -> Any:
    """Parse JSON from a model response that may contain Markdown fences."""
    response = content_to_text(text).strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fence_match:
        response = fence_match.group(1).strip()

    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            for key in ("data", "result", "results", "papers", "items", "outputs"):
                if key in parsed and isinstance(parsed[key], (list, dict)):
                    return parsed[key]
        return parsed
    except json.JSONDecodeError:
        pass

    candidates: List[Tuple[int, int]] = []
    if "{" in response and "}" in response:
        candidates.append((response.find("{"), response.rfind("}") + 1))
    if "[" in response and "]" in response:
        candidates.append((response.find("["), response.rfind("]") + 1))

    for start, end in candidates:
        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError:
            continue

    raise ValueError("LLM response did not contain valid JSON.")


class LLMClient:
    def __init__(self, api_url: str, api_key: str, model: str, timeout: int = 90):
        self.api_url = api_url.rstrip("/")
        if not self.api_url.endswith("/chat/completions"):
            self.api_url = self.api_url + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Optional["LLMClient"]:
        api_key = (
            args.api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("ZAI_API_KEY")
            or os.getenv("BIGMODEL_API_KEY")
            or os.getenv("ZHIPUAI_API_KEY")
            or os.getenv("GLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            return None

        api_url = (
            args.api_url
            or os.getenv("LLM_API_URL")
            or os.getenv("ZAI_API_URL")
            or os.getenv("BIGMODEL_API_URL")
            or os.getenv("GLM_API_URL")
            or os.getenv("OPENAI_BASE_URL")
            or ("https://open.bigmodel.cn/api/paas/v4" if (
                os.getenv("ZAI_API_KEY")
                or os.getenv("BIGMODEL_API_KEY")
                or os.getenv("ZHIPUAI_API_KEY")
                or os.getenv("GLM_API_KEY")
            ) else None)
            or ("https://api.deepseek.com/v1" if os.getenv("DEEPSEEK_API_KEY") else "https://api.openai.com/v1")
        )
        model = (
            args.model
            or os.getenv("LLM_MODEL")
            or os.getenv("ZAI_MODEL")
            or os.getenv("BIGMODEL_MODEL")
            or os.getenv("GLM_MODEL")
            or (
                "glm-5.1" if (
                    os.getenv("ZAI_API_KEY")
                    or os.getenv("BIGMODEL_API_KEY")
                    or os.getenv("ZHIPUAI_API_KEY")
                    or os.getenv("GLM_API_KEY")
                ) else None
            )
            or ("deepseek-chat" if os.getenv("DEEPSEEK_API_KEY") else "gpt-4o-mini")
        )
        return cls(api_url=api_url, api_key=api_key, model=model, timeout=args.llm_timeout)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
        retries: int = 3,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(retries):
            request = urllib.request.Request(self.api_url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                choice = result["choices"][0]
                message = choice.get("message", {})
                content = message.get("content")
                if content is None and "text" in choice:
                    content = choice.get("text")
                if content is None:
                    raise KeyError("message.content")
                return content_to_text(content)
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""
                detail = body
                try:
                    error_json = json.loads(body) if body else {}
                    if isinstance(error_json, dict):
                        error_obj = error_json.get("error", error_json)
                        if isinstance(error_obj, dict):
                            detail = error_obj.get("message") or error_obj.get("msg") or body
                except json.JSONDecodeError:
                    detail = body or str(exc)
                last_error = RuntimeError(f"HTTP {exc.code} {exc.reason}: {shorten_text(detail, 800)}")
                wait_seconds = min(20, 2 ** attempt)
                print(f"LLM call failed on attempt {attempt + 1}/{retries}: {last_error}")
                if attempt < retries - 1:
                    time.sleep(wait_seconds)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
                wait_seconds = min(20, 2 ** attempt)
                print(f"LLM call failed on attempt {attempt + 1}/{retries}: {exc}")
                if attempt < retries - 1:
                    time.sleep(wait_seconds)

        raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def fallback_search_plan(interest: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
    text = interest.lower()
    mapped_terms: List[str] = []
    theme_specs = [
        (
            ["视觉slam", "slam", "定位", "建图"],
            "视觉 SLAM / 视觉里程计",
            ["visual SLAM", "visual odometry", "camera localization", "simultaneous localization and mapping"],
        ),
        (
            ["3dgs", "gaussian", "高斯", "splatting"],
            "3DGS 地图表示与 SLAM",
            ["3D Gaussian Splatting", "Gaussian Splatting SLAM", "3DGS SLAM", "dense visual SLAM"],
        ),
        (
            ["mesh", "具身", "视频", "图片", "场景生成", "三维重建", "3d"],
            "图像/视频到 3D 场景",
            [
                "image to 3D",
                "video to 3D",
                "3D reconstruction",
                "mesh reconstruction",
                "scene generation",
                "embodied AI",
            ],
        ),
        (
            ["动态", "人体姿势", "人体姿态", "pose", "motion"],
            "动态物体与人体姿态",
            [
                "dynamic scene reconstruction",
                "dynamic object reconstruction",
                "human pose estimation",
                "human motion estimation",
                "video motion estimation",
            ],
        ),
    ]

    search_queries = []
    for triggers, name, terms in theme_specs:
        if any(trigger in text for trigger in triggers):
            mapped_terms.extend(terms)
            search_queries.append(
                {
                    "name_cn": name,
                    "terms": terms,
                    "required_terms": [],
                    "notes_cn": "未配置 LLM 时按内置主题映射生成。",
                }
            )

    rough_terms = re.split(r"[,，;；、\n]+", interest)
    mapped_terms.extend([clean_text(item) for item in rough_terms if clean_text(item)])

    # Preserve order while removing duplicates.
    deduped_terms = list(dict.fromkeys(mapped_terms))[:12]
    if not deduped_terms:
        deduped_terms = [interest]

    if not search_queries:
        search_queries = [
            {
                "name_cn": "用户原始兴趣的宽搜版本",
                "terms": deduped_terms[:8],
                "required_terms": [],
                "notes_cn": "未配置 LLM 时生成的保守搜索词。",
            }
        ]

    return {
        "search_intent_cn": interest,
        "search_queries": search_queries[:4],
        "positive_keywords": deduped_terms,
        "negative_keywords": [],
        "categories": categories or [],
        "rationale_cn": "未配置 LLM，使用内置关键词映射和用户原始输入生成搜索计划。",
        "source": "fallback",
    }


def optimize_search_plan(
    interest: str,
    llm: Optional[LLMClient],
    categories: Optional[List[str]],
) -> Dict[str, Any]:
    if llm is None:
        return fallback_search_plan(interest, categories)

    category_hint = ", ".join(categories) if categories else "由你推荐；如果不确定可留空"
    messages = [
        {
            "role": "system",
            "content": (
                "你是熟悉 arXiv 检索语法的科研助手。你的任务是把用户可能不够准确的中文研究兴趣，"
                "改写成适合 arXiv 的英文检索词。只输出 JSON object，不要输出 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": f"""
用户研究兴趣：
{interest}

用户指定类别提示：
{category_hint}

请生成 JSON，schema 如下：
{{
  "search_intent_cn": "用中文重述你理解的检索目标",
  "search_queries": [
    {{
      "name_cn": "这一组检索词的用途",
      "terms": ["英文短语或术语，优先 arXiv 标题/摘要中会出现的表达"],
      "required_terms": ["可选，必须同时出现的英文术语"],
      "notes_cn": "为什么这样搜"
    }}
  ],
  "positive_keywords": ["用于本地相关性打分的中英文关键词"],
  "negative_keywords": ["明显不想要方向的关键词，比如 lidar、survey、navigation 等，如果用户没有排除项可以少填"],
  "categories": ["推荐 arXiv 类别，例如 cs.CV, cs.RO, cs.AI, eess.SY；不确定可留空"],
  "rationale_cn": "简短说明你如何修正了用户输入"
}}

约束：
- search_queries 最多 4 组。
- 每组 terms 最多 8 个，尽量包含同义词和常用缩写。
- 不要只给 robotics、deep learning 这类过宽词。
- 类别只能给 arXiv category 字符串。
""".strip(),
        },
    ]

    try:
        response = llm.chat(messages, temperature=0.1, max_tokens=2500)
        plan = extract_json_from_text(response)
        if not isinstance(plan, dict):
            raise ValueError("Search plan is not a JSON object.")
        plan.setdefault("source", "llm")
        if categories is not None:
            plan["categories"] = categories
        return normalize_search_plan(plan, interest)
    except Exception as exc:
        print(f"Search-plan LLM step failed, using fallback plan: {exc}")
        return fallback_search_plan(interest, categories)


def normalize_search_plan(plan: Dict[str, Any], interest: str) -> Dict[str, Any]:
    queries = plan.get("search_queries") or []
    normalized_queries = []
    for query in queries:
        if isinstance(query, str):
            normalized_queries.append({"name_cn": query, "terms": [query], "required_terms": [], "notes_cn": ""})
        elif isinstance(query, dict):
            terms = query.get("terms") or []
            if isinstance(terms, str):
                terms = [terms]
            required_terms = query.get("required_terms") or []
            if isinstance(required_terms, str):
                required_terms = [required_terms]
            normalized_queries.append(
                {
                    "name_cn": clean_text(query.get("name_cn") or query.get("name") or "检索组"),
                    "terms": [clean_text(str(item)) for item in terms if clean_text(str(item))][:8],
                    "required_terms": [clean_text(str(item)) for item in required_terms if clean_text(str(item))][:4],
                    "notes_cn": clean_text(query.get("notes_cn") or query.get("notes") or ""),
                }
            )

    if not normalized_queries:
        normalized_queries = fallback_search_plan(interest).get("search_queries", [])

    plan["search_queries"] = normalized_queries[:4]
    for key in ("positive_keywords", "negative_keywords", "categories"):
        value = plan.get(key) or []
        if isinstance(value, str):
            value = parse_csv(value)
        plan[key] = [clean_text(str(item)) for item in value if clean_text(str(item))]
    plan.setdefault("search_intent_cn", interest)
    plan.setdefault("rationale_cn", "")
    return plan


def arxiv_term_clause(term: str, field: str = "all") -> str:
    term = term.strip().replace('"', "")
    if not term:
        return ""
    if re.match(r"^(all|ti|abs|au|cat|id|jr|co):", term):
        return term
    if re.search(r"\s", term):
        return f'{field}:"{term}"'
    return f"{field}:{term}"


def build_arxiv_query(
    query: Dict[str, Any],
    categories: List[str],
    from_date: Optional[str],
    to_date: Optional[str],
) -> str:
    terms = [arxiv_term_clause(term) for term in query.get("terms", [])]
    terms = [term for term in terms if term]
    required_terms = [arxiv_term_clause(term) for term in query.get("required_terms", [])]
    required_terms = [term for term in required_terms if term]

    if not terms:
        terms = [arxiv_term_clause(query.get("name_cn", ""))]

    parts = [f"({' OR '.join(terms)})"]
    for required in required_terms:
        parts.append(required)

    if categories:
        cat_expr = " OR ".join(f"cat:{cat}" for cat in categories)
        parts.append(f"({cat_expr})")

    if from_date or to_date:
        start = date_to_arxiv_bound(from_date, "0000") if from_date else "190001010000"
        end = date_to_arxiv_bound(to_date, "2359") if to_date else "299912312359"
        parts.append(f"submittedDate:[{start} TO {end}]")

    return " AND ".join(parts)


def date_to_arxiv_bound(value: str, hhmm: str) -> str:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        raise ValueError(f"Date must be YYYY-MM-DD: {value}")
    return value.replace("-", "") + hhmm


def fetch_arxiv_page(
    search_query: str,
    start: int,
    max_results: int,
    sort_by: str,
    sort_order: str,
    timeout: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "arxiv-paper-radar/1.0 (local research helper)",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        xml_text = response.read().decode("utf-8")

    root = ET.fromstring(xml_text)
    total_text = root.findtext("opensearch:totalResults", default="0", namespaces=ATOM_NS)
    try:
        total = int(total_text.strip())
    except ValueError:
        total = 0

    entries = [parse_arxiv_entry(entry) for entry in root.findall("atom:entry", ATOM_NS)]
    return total, entries


def parse_arxiv_entry(entry: ET.Element) -> Dict[str, Any]:
    id_url = clean_text(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
    raw_arxiv_id = id_url.rsplit("/abs/", 1)[-1] if "/abs/" in id_url else id_url
    version_match = re.search(r"v(\d+)$", raw_arxiv_id)
    version = version_match.group(1) if version_match else ""
    arxiv_id = re.sub(r"v\d+$", "", raw_arxiv_id)

    authors = [
        clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    authors = [author for author in authors if author]

    links = {"abs": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else id_url, "pdf": ""}
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.attrib.get("href", "")
        title = link.attrib.get("title", "")
        link_type = link.attrib.get("type", "")
        if title == "pdf" or link_type == "application/pdf":
            links["pdf"] = href
        elif link.attrib.get("rel") == "alternate" and href:
            links["abs"] = href

    categories = [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ATOM_NS)]
    categories = [cat for cat in categories if cat]
    primary_category_node = entry.find("arxiv:primary_category", ATOM_NS)
    primary_category = primary_category_node.attrib.get("term", "") if primary_category_node is not None else ""

    paper = {
        "arxiv_id": arxiv_id,
        "arxiv_version": version,
        "title": clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS)),
        "authors": authors,
        "published": clean_text(entry.findtext("atom:published", default="", namespaces=ATOM_NS)),
        "updated": clean_text(entry.findtext("atom:updated", default="", namespaces=ATOM_NS)),
        "primary_category": primary_category,
        "categories": categories,
        "abstract": clean_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS)),
        "comment": clean_text(entry.findtext("arxiv:comment", default="", namespaces=ATOM_NS)),
        "journal_ref": clean_text(entry.findtext("arxiv:journal_ref", default="", namespaces=ATOM_NS)),
        "doi": clean_text(entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS)),
        "links": links,
    }
    paper["publication_venue"] = infer_publication_venue(paper)
    return paper


def infer_publication_venue(paper: Dict[str, Any]) -> str:
    journal_ref = clean_text(paper.get("journal_ref", ""))
    if journal_ref:
        return journal_ref

    comment = clean_text(paper.get("comment", ""))
    if not comment:
        return ""

    venue_patterns = [
        r"(?:accepted|published|to appear)\s+(?:at|in|to)\s+([^.;,\n]+)",
        r"(?:proceedings of|appears in)\s+([^.;,\n]+)",
    ]
    for pattern in venue_patterns:
        match = re.search(pattern, comment, re.IGNORECASE)
        if match:
            return clean_text(match.group(1))

    known_venues = [
        "CVPR", "ICCV", "ECCV", "NeurIPS", "ICML", "ICLR", "AAAI", "IJCAI",
        "ICRA", "IROS", "RSS", "CoRL", "WACV", "BMVC", "SIGGRAPH", "TPAMI",
        "TRO", "RA-L", "RAL", "IJRR", "TITS", "IV", "3DV", "ICCVW", "CVPRW",
    ]
    upper_comment = comment.upper()
    for venue in known_venues:
        if venue.upper() in upper_comment:
            return venue

    return ""


def score_paper(paper: Dict[str, Any], positive_keywords: Iterable[str], negative_keywords: Iterable[str]) -> int:
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    haystack = f"{title} {abstract}"
    score = 5

    for keyword in positive_keywords:
        key = keyword.lower()
        if not key:
            continue
        if key in title:
            score += 12
        elif key in abstract:
            score += 5

    for keyword in negative_keywords:
        key = keyword.lower()
        if key and key in haystack:
            score -= 15

    return max(0, min(100, score))


def search_arxiv(
    plan: Dict[str, Any],
    args: argparse.Namespace,
    progress: Optional[ProgressReporter] = None,
    existing_papers: Optional[Dict[str, Dict[str, Any]]] = None,
    round_label: str = "初始检索",
    round_index: int = 1,
) -> List[Dict[str, Any]]:
    categories = plan.get("categories", [])
    positive_keywords = plan.get("positive_keywords", [])
    negative_keywords = plan.get("negative_keywords", [])
    deduped: Dict[str, Dict[str, Any]] = dict(existing_papers or {})
    search_queries = plan.get("search_queries", [])
    candidate_results = resolve_candidate_results(args)
    per_query_limit_default = resolve_per_query_limit(args, len(search_queries))
    query_progress = [
        {
            "index": idx,
            "round": round_index,
            "round_label": round_label,
            "name": query.get("name_cn", "检索组"),
            "arxiv_query": "",
            "status": "pending",
            "fetched": 0,
            "limit": per_query_limit_default,
            "total": 0,
            "unique_papers_after_query": 0,
        }
        for idx, query in enumerate(search_queries, start=1)
    ]

    if progress is not None:
        progress.patch(
            search={
                "total_queries": len(search_queries),
                "queries": query_progress,
                "candidate_target": candidate_results,
                "per_query_limit": per_query_limit_default,
            }
        )

    for query_idx, query in enumerate(search_queries, start=1):
        search_query = build_arxiv_query(query, categories, args.from_date, args.to_date)
        print(f"\n[{query_idx}] {query.get('name_cn', '检索组')}")
        print(f"arXiv query: {search_query}")

        fetched_for_query = 0
        start = 0
        per_query_limit = per_query_limit_default
        query_progress[query_idx - 1].update(
            {
                "arxiv_query": search_query,
                "status": "running",
                "fetched": fetched_for_query,
                "limit": per_query_limit,
            }
        )
        if progress is not None:
            progress.patch(
                message=f"{round_label}：正在检索第 {query_idx}/{len(search_queries)} 组：{query.get('name_cn', '检索组')}",
                search={
                    "current_query_index": query_idx,
                    "current_query_name": query.get("name_cn", "检索组"),
                    "current_query_arxiv_query": search_query,
                    "current_query_fetched": fetched_for_query,
                    "current_query_limit": per_query_limit,
                    "current_query_total": 0,
                    "total_unique_papers": len(deduped),
                    "queries": query_progress,
                },
            )

        while fetched_for_query < per_query_limit:
            page_size = min(100, per_query_limit - fetched_for_query)
            try:
                total, entries = fetch_arxiv_page(
                    search_query=search_query,
                    start=start,
                    max_results=page_size,
                    sort_by=args.sort_by,
                    sort_order=args.sort_order,
                    timeout=args.arxiv_timeout,
                )
            except urllib.error.URLError as exc:
                print(f"arXiv request failed: {exc}")
                query_progress[query_idx - 1]["status"] = "failed"
                if progress is not None:
                    progress.patch(
                        message=f"第 {query_idx} 组 arXiv 请求失败：{exc}",
                        search={"queries": query_progress},
                    )
                break

            if not entries:
                print("No more arXiv entries returned for this query.")
                query_progress[query_idx - 1]["status"] = "completed"
                if progress is not None:
                    progress.patch(
                        message=f"第 {query_idx} 组没有更多结果",
                        search={"queries": query_progress},
                    )
                break

            for paper in entries:
                paper["matched_query"] = query.get("name_cn", "")
                paper["heuristic_score"] = score_paper(paper, positive_keywords, negative_keywords)
                key = paper.get("arxiv_id") or paper.get("title", "")
                if key and key not in deduped:
                    deduped[key] = paper
                elif key:
                    deduped[key]["heuristic_score"] = max(
                        deduped[key].get("heuristic_score", 0),
                        paper.get("heuristic_score", 0),
                    )

            fetched_for_query += len(entries)
            start += len(entries)
            print(f"Fetched {fetched_for_query}/{min(total, per_query_limit)} for this query.")
            query_progress[query_idx - 1].update(
                {
                    "fetched": fetched_for_query,
                    "total": total,
                    "unique_papers_after_query": len(deduped),
                }
            )
            if progress is not None:
                progress.patch(
                    message=f"{round_label}：第 {query_idx}/{len(search_queries)} 组已抓取 {fetched_for_query}/{min(total, per_query_limit)}，当前去重后 {len(deduped)} 篇",
                    search={
                        "current_query_fetched": fetched_for_query,
                        "current_query_total": total,
                        "total_unique_papers": len(deduped),
                        "queries": query_progress,
                    },
                )

            if fetched_for_query >= total or len(entries) < page_size:
                break
            time.sleep(args.request_delay)

        if query_progress[query_idx - 1]["status"] == "running":
            query_progress[query_idx - 1]["status"] = "completed"
            if progress is not None:
                progress.patch(search={"queries": query_progress})

        if query_idx < len(search_queries):
            time.sleep(args.request_delay)

    papers = list(deduped.values())
    papers.sort(key=lambda item: (item.get("heuristic_score", 0), item.get("published", "")), reverse=True)
    selected_papers = papers[:candidate_results]
    if progress is not None:
        progress.patch(
            message=f"{round_label}完成，候选池保留 {len(selected_papers)} 篇，最终将输出 {args.max_results} 篇",
            search={
                "total_unique_papers": len(deduped),
                "queries": query_progress,
            },
        )
    return selected_papers


def batched(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fallback_analysis(papers: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    positive_keywords = plan.get("positive_keywords", [])
    negative_keywords = plan.get("negative_keywords", [])
    analyzed = []
    for paper in papers:
        score = paper.get("heuristic_score", score_paper(paper, positive_keywords, negative_keywords))
        matched = []
        haystack = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        for keyword in positive_keywords:
            if keyword.lower() in haystack:
                matched.append(keyword)
        priority = "high" if score >= 70 else "medium" if score >= 35 else "low"
        enriched = dict(paper)
        enriched.update(
            {
                "relevance_score": score,
                "priority": priority,
                "summary_cn": "未配置 LLM，暂保留 arXiv 原始摘要；配置 LLM 后会自动生成中文总结。",
                "key_points_cn": [
                    f"标题匹配方向：{paper.get('title', '')}",
                    f"arXiv 类别：{', '.join(paper.get('categories', [])) or 'N/A'}",
                ],
                "innovation_ideas_cn": [
                    "建议重点阅读方法设计、实验设定和开源资源，判断能否迁移到你的研究问题。",
                    "如果该论文与目标方向高度相关，可把摘要和方法部分再次交给 LLM 做细粒度拆解。",
                ],
                "why_relevant_cn": "根据标题和摘要关键词初筛得到。" if matched else "当前只完成 arXiv 检索，尚未进行 LLM 语义判断。",
                "limitations_cn": ["未进行 LLM 语义分析，相关性可能不够准确。"],
                "matched_topics": matched[:8],
            }
        )
        analyzed.append(enriched)
    return analyzed


def split_batch_in_half(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    middle = max(1, len(items) // 2)
    return items[:middle], items[middle:]


def score_screen_batch(
    batch: List[Dict[str, Any]],
    plan: Dict[str, Any],
    interest: str,
    llm: LLMClient,
    max_tokens: int = 5000,
) -> List[Dict[str, Any]]:
    compact_papers = [
        {
            "arxiv_id": paper.get("arxiv_id"),
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "categories": paper.get("categories"),
            "published": paper.get("published"),
            "publication_venue": paper.get("publication_venue"),
            "journal_ref": paper.get("journal_ref"),
            "comment": paper.get("comment"),
            "heuristic_score": paper.get("heuristic_score"),
        }
        for paper in batch
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "你是论文检索筛选助手。请只做相关性打分和简短理由，不要做详细总结。"
                "只输出 JSON array，不要输出 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": f"""
用户真正想找的工作：
{interest}

检索计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

候选论文：
{json.dumps(compact_papers, ensure_ascii=False, indent=2)}

请为每篇论文返回一个 JSON object，数组长度必须等于候选论文数量：
[
  {{
    "arxiv_id": "必须与输入一致",
    "screen_score": 0,
    "screen_reason_cn": "一句话说明为什么相关或不相关",
    "matched_topics": ["匹配主题词"],
    "should_keep": true
  }}
]

打分规则：
- 90-100：直接命中用户主题，值得优先读。
- 70-89：强相关，但可能缺少 3DGS 或 occupancy 中的一个关键元素。
- 40-69：弱相关或背景相关。
- 0-39：明显偏离。
- 如果是 survey、纯 SLAM、纯 navigation、LiDAR-only 且不涉及用户主题，降低分数。
""".strip(),
        },
    ]

    response = llm.chat(messages, temperature=0.1, max_tokens=max_tokens)
    parsed = extract_json_from_text(response)
    if not isinstance(parsed, list):
        raise ValueError(f"Screening response is not a JSON array: {shorten_text(content_to_text(response), 400)}")
    return [item for item in parsed if isinstance(item, dict)]


def analyze_summary_batch(
    batch: List[Dict[str, Any]],
    plan: Dict[str, Any],
    interest: str,
    llm: LLMClient,
    max_tokens: int = 5000,
) -> List[Dict[str, Any]]:
    compact_papers = [
        {
            "arxiv_id": paper.get("arxiv_id"),
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "categories": paper.get("categories"),
            "published": paper.get("published"),
            "publication_venue": paper.get("publication_venue"),
            "journal_ref": paper.get("journal_ref"),
            "comment": paper.get("comment"),
            "heuristic_score": paper.get("heuristic_score"),
        }
        for paper in batch
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "你是科研论文阅读助手。请根据用户研究兴趣，用中文总结 arXiv 论文的核心贡献、"
                "关键要点和可借鉴创新点。只输出 JSON array，不要输出 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": f"""
用户研究兴趣：
{interest}

LLM 优化后的检索计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

待分析论文：
{json.dumps(compact_papers, ensure_ascii=False, indent=2)}

请为每篇论文返回一个 JSON object，数组长度必须等于论文数量。schema：
[
  {{
    "arxiv_id": "必须和输入一致",
    "relevance_score": 0,
    "priority": "high | medium | low",
    "summary_cn": "用中文 2-4 句话总结论文做了什么",
    "key_points_cn": ["关键要点1", "关键要点2", "关键要点3"],
    "innovation_ideas_cn": ["这个工作可供用户参考的创新点/可迁移启发1", "启发2"],
    "why_relevant_cn": "说明它和用户兴趣的关系，如果不相关也直接说清楚",
    "limitations_cn": ["从摘要中能看出的局限或需要进一步查全文确认的点"],
    "matched_topics": ["匹配到的主题词"]
  }}
]

要求：
- relevance_score 根据用户兴趣打 0-100 分，不要全部高分。
- 不要编造摘要里没有的信息。
- innovation_ideas_cn 要写成用户可以参考的研究启发，而不是泛泛夸奖。
- 如果论文不相关，也要保留条目并给低分，方便后续网页筛选。
""".strip(),
        },
    ]
    response = llm.chat(messages, temperature=0.1, max_tokens=max_tokens)
    parsed = extract_json_from_text(response)
    if not isinstance(parsed, list):
        raise ValueError(f"Analysis response is not a JSON array: {shorten_text(content_to_text(response), 400)}")
    return [item for item in parsed if isinstance(item, dict)]


def screen_candidates_with_llm(
    papers: List[Dict[str, Any]],
    plan: Dict[str, Any],
    interest: str,
    llm: Optional[LLMClient],
    keep_count: int,
    batch_size: int,
    progress: Optional[ProgressReporter] = None,
) -> List[Dict[str, Any]]:
    if len(papers) <= keep_count:
        if progress is not None:
            progress.patch(
                message=f"候选论文 {len(papers)} 篇，不超过最终数量 {keep_count}，跳过轻量筛选",
                screening={
                    "current_batch": 0,
                    "total_batches": 0,
                    "batch_size": batch_size,
                    "processed_papers": len(papers),
                    "total_papers": len(papers),
                    "selected_papers": len(papers),
                    "current_titles": [],
                    "batches": [],
                },
            )
        return papers

    if llm is None:
        selected = sorted(
            papers,
            key=lambda item: (item.get("heuristic_score", 0), item.get("published", "")),
            reverse=True,
        )[:keep_count]
        if progress is not None:
            progress.patch(
                message=f"未启用 LLM，已按关键词分数从 {len(papers)} 篇候选中选出 {len(selected)} 篇",
                screening={
                    "current_batch": 1,
                    "total_batches": 1,
                    "batch_size": len(papers),
                    "processed_papers": len(papers),
                    "total_papers": len(papers),
                    "selected_papers": len(selected),
                    "current_titles": [],
                    "batches": [
                        {
                            "index": 1,
                            "status": "completed",
                            "paper_count": len(papers),
                            "titles": [paper.get("title", "") for paper in papers[:8]],
                        }
                    ],
                },
            )
        return selected

    scored_by_id: Dict[str, Dict[str, Any]] = {}
    total_batches = (len(papers) + batch_size - 1) // batch_size
    batch_progress = [
        {
            "index": idx,
            "status": "pending",
            "paper_count": len(batch),
            "titles": [paper.get("title", "") for paper in batch[:6]],
        }
        for idx, batch in enumerate(batched(papers, batch_size), start=1)
    ]
    if progress is not None:
        progress.patch(
            message=f"准备从 {len(papers)} 篇候选中轻量筛选出 {keep_count} 篇",
            screening={
                "current_batch": 0,
                "total_batches": total_batches,
                "batch_size": batch_size,
                "processed_papers": 0,
                "total_papers": len(papers),
                "selected_papers": 0,
                "current_titles": [],
                "batches": batch_progress,
            },
        )

    for batch_idx, batch in enumerate(batched(papers, batch_size), start=1):
        batch_progress[batch_idx - 1]["status"] = "running"
        if progress is not None:
            progress.patch(
                message=f"正在轻量筛选第 {batch_idx}/{total_batches} 批候选",
                screening={
                    "current_batch": batch_idx,
                    "processed_papers": min((batch_idx - 1) * batch_size, len(papers)),
                    "current_titles": [paper.get("title", "") for paper in batch],
                    "batches": batch_progress,
                },
            )

        try:
            parsed = score_screen_batch(batch, plan, interest, llm, max_tokens=5000)
            for item in parsed:
                if isinstance(item, dict) and item.get("arxiv_id"):
                    scored_by_id[str(item["arxiv_id"])] = item
            batch_progress[batch_idx - 1]["status"] = "completed"
        except Exception as exc:
            print(f"LLM screening failed for batch {batch_idx} ({len(batch)} papers): {exc}")
            if len(batch) > 1:
                left_batch, right_batch = split_batch_in_half(batch)
                print(f"Retrying screening batch {batch_idx} by splitting into {len(left_batch)} + {len(right_batch)} papers.")
                try:
                    for sub_batch in (left_batch, right_batch):
                        parsed = score_screen_batch(sub_batch, plan, interest, llm, max_tokens=3000)
                        for item in parsed:
                            if isinstance(item, dict) and item.get("arxiv_id"):
                                scored_by_id[str(item["arxiv_id"])] = item
                    batch_progress[batch_idx - 1]["status"] = "split"
                except Exception as split_exc:
                    print(f"Split retry also failed for screening batch {batch_idx}: {split_exc}")
                    for paper in batch:
                        scored_by_id[str(paper.get("arxiv_id", paper.get("title", "")))] = {
                            "arxiv_id": paper.get("arxiv_id"),
                            "screen_score": paper.get("heuristic_score", 0),
                            "screen_reason_cn": f"LLM 轻量筛选失败，使用关键词初筛分数。错误：{shorten_text(str(split_exc), 180)}",
                            "matched_topics": [],
                            "should_keep": True,
                        }
                    batch_progress[batch_idx - 1]["status"] = "fallback"
            else:
                paper = batch[0]
                scored_by_id[str(paper.get("arxiv_id", paper.get("title", "")))] = {
                    "arxiv_id": paper.get("arxiv_id"),
                    "screen_score": paper.get("heuristic_score", 0),
                    "screen_reason_cn": f"LLM 轻量筛选失败，使用关键词初筛分数。错误：{shorten_text(str(exc), 180)}",
                    "matched_topics": [],
                    "should_keep": True,
                }
                batch_progress[batch_idx - 1]["status"] = "fallback"

        if progress is not None:
            progress.patch(
                message=f"第 {batch_idx}/{total_batches} 批候选筛选完成",
                screening={
                    "processed_papers": min(batch_idx * batch_size, len(papers)),
                    "current_titles": [],
                    "batches": batch_progress,
                },
            )

    enriched_candidates = []
    for paper in papers:
        arxiv_id = str(paper.get("arxiv_id", ""))
        score_item = scored_by_id.get(arxiv_id, {})
        enriched = dict(paper)
        enriched["screen_score"] = int(score_item.get("screen_score", paper.get("heuristic_score", 0)) or 0)
        enriched["screen_reason_cn"] = score_item.get("screen_reason_cn", "")
        enriched["screen_should_keep"] = bool(score_item.get("should_keep", True))
        if score_item.get("matched_topics"):
            enriched["matched_topics"] = ensure_list(score_item.get("matched_topics"))
        enriched_candidates.append(enriched)

    enriched_candidates.sort(
        key=lambda item: (
            item.get("screen_should_keep", True),
            item.get("screen_score", 0),
            item.get("heuristic_score", 0),
            item.get("published", ""),
        ),
        reverse=True,
    )
    selected = enriched_candidates[:keep_count]
    if progress is not None:
        progress.patch(
            message=f"轻量筛选完成：从 {len(papers)} 篇候选中选出 {len(selected)} 篇进入详细分析",
            screening={
                "processed_papers": len(papers),
                "selected_papers": len(selected),
                "current_titles": [],
                "batches": batch_progress,
            },
        )
    return selected


def build_refined_search_plan(
    interest: str,
    base_plan: Dict[str, Any],
    seed_papers: List[Dict[str, Any]],
    llm: Optional[LLMClient],
    categories: Optional[List[str]],
    round_index: int,
) -> Optional[Dict[str, Any]]:
    if llm is None or not seed_papers:
        return None

    compact_papers = [
        {
            "arxiv_id": paper.get("arxiv_id"),
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "screen_score": paper.get("screen_score", paper.get("heuristic_score")),
            "screen_reason_cn": paper.get("screen_reason_cn", ""),
            "matched_topics": paper.get("matched_topics", []),
        }
        for paper in seed_papers
    ]
    category_hint = ", ".join(categories or base_plan.get("categories", [])) or "由你推荐；如果不确定可留空"
    messages = [
        {
            "role": "system",
            "content": (
                "你是 arXiv 迭代检索助手。请根据已经找到的高相关论文，生成下一轮更精准、"
                "更能召回相似工作的英文检索词。只输出 JSON object，不要输出 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": f"""
用户目标：
{interest}

上一轮检索计划：
{json.dumps(base_plan, ensure_ascii=False, indent=2)}

上一轮高相关论文：
{json.dumps(compact_papers, ensure_ascii=False, indent=2)}

请生成下一轮检索 JSON：
{{
  "search_intent_cn": "下一轮检索目标",
  "search_queries": [
    {{
      "name_cn": "检索组名称",
      "terms": ["英文术语或短语"],
      "required_terms": ["可选，必须出现的术语"],
      "notes_cn": "为什么这样扩展"
    }}
  ],
  "positive_keywords": ["用于相关性打分的关键词"],
  "negative_keywords": ["要降权的关键词"],
  "categories": ["{category_hint}"],
  "rationale_cn": "你从高相关论文中提炼出了哪些新搜索方向"
}}

约束：
- 最多 3 组 search_queries，每组 terms 最多 8 个。
- 要尽量补充同义词、任务名、数据表示名、自动驾驶常用表达，例如 occupancy prediction、occupancy forecasting、occupancy scene completion、semantic occupancy、BEV occupancy、Gaussian Splatting、3DGS、scene reconstruction。
- 避免只重复上一轮完全相同的 query。
""".strip(),
        },
    ]

    try:
        response = llm.chat(messages, temperature=0.1, max_tokens=2500)
        plan = extract_json_from_text(response)
        if not isinstance(plan, dict):
            raise ValueError("Refined search plan is not a JSON object.")
        plan.setdefault("source", f"llm_refine_round_{round_index}")
        if categories is not None:
            plan["categories"] = categories
        return normalize_search_plan(plan, interest)
    except Exception as exc:
        print(f"Refined search-plan LLM step failed: {exc}")
        return None


def analyze_papers_with_llm(
    papers: List[Dict[str, Any]],
    plan: Dict[str, Any],
    interest: str,
    llm: Optional[LLMClient],
    batch_size: int,
    min_score: int,
    progress: Optional[ProgressReporter] = None,
) -> List[Dict[str, Any]]:
    if llm is None:
        analyzed = [paper for paper in fallback_analysis(papers, plan) if paper.get("relevance_score", 0) >= min_score]
        if progress is not None:
            progress.patch(
                message=f"未启用 LLM，已完成关键词初筛，共 {len(analyzed)} 篇",
                analysis={
                    "current_batch": 1 if papers else 0,
                    "total_batches": 1 if papers else 0,
                    "batch_size": batch_size,
                    "processed_papers": len(papers),
                    "total_papers": len(papers),
                    "current_titles": [],
                    "batches": [
                        {
                            "index": 1,
                            "status": "completed",
                            "paper_count": len(papers),
                            "titles": [paper.get("title", "") for paper in papers[:8]],
                        }
                    ] if papers else [],
                },
            )
        return analyzed

    analyzed_by_id: Dict[str, Dict[str, Any]] = {}
    total_batches = (len(papers) + batch_size - 1) // batch_size
    batch_progress = [
        {
            "index": idx,
            "status": "pending",
            "paper_count": len(batch),
            "titles": [paper.get("title", "") for paper in batch],
        }
        for idx, batch in enumerate(batched(papers, batch_size), start=1)
    ]
    if progress is not None:
        progress.patch(
            message=f"准备分 {total_batches} 批进行 LLM 分析",
            analysis={
                "current_batch": 0,
                "total_batches": total_batches,
                "batch_size": batch_size,
                "processed_papers": 0,
                "total_papers": len(papers),
                "current_titles": [],
                "batches": batch_progress,
            },
        )

    for batch_idx, batch in enumerate(batched(papers, batch_size), start=1):
        batch_progress[batch_idx - 1]["status"] = "running"
        if progress is not None:
            progress.patch(
                message=f"正在用 LLM 分析第 {batch_idx}/{total_batches} 批",
                analysis={
                    "current_batch": batch_idx,
                    "processed_papers": min((batch_idx - 1) * batch_size, len(papers)),
                    "current_titles": [paper.get("title", "") for paper in batch],
                    "batches": batch_progress,
                },
            )
        print(f"Analyzing batch {batch_idx}/{total_batches} with LLM...")
        try:
            parsed = analyze_summary_batch(batch, plan, interest, llm, max_tokens=5000)
            for item in parsed:
                if isinstance(item, dict) and item.get("arxiv_id"):
                    analyzed_by_id[str(item["arxiv_id"])] = item
            batch_progress[batch_idx - 1]["status"] = "completed"
        except Exception as exc:
            print(f"LLM analysis failed for batch {batch_idx} ({len(batch)} papers): {exc}")
            if len(batch) > 1:
                left_batch, right_batch = split_batch_in_half(batch)
                print(f"Retrying analysis batch {batch_idx} by splitting into {len(left_batch)} + {len(right_batch)} papers.")
                try:
                    for sub_batch in (left_batch, right_batch):
                        parsed = analyze_summary_batch(sub_batch, plan, interest, llm, max_tokens=3000)
                        for item in parsed:
                            if isinstance(item, dict) and item.get("arxiv_id"):
                                analyzed_by_id[str(item["arxiv_id"])] = item
                    batch_progress[batch_idx - 1]["status"] = "split"
                except Exception as split_exc:
                    print(f"Split retry also failed for analysis batch {batch_idx}: {split_exc}")
                    for item in fallback_analysis(batch, plan):
                        item["limitations_cn"] = ensure_list(item.get("limitations_cn")) + [
                            f"LLM 详细分析失败，已回退到摘要初筛。错误：{shorten_text(str(split_exc), 180)}"
                        ]
                        analyzed_by_id[item.get("arxiv_id", item.get("title", ""))] = item
                    batch_progress[batch_idx - 1]["status"] = "fallback"
            else:
                for item in fallback_analysis(batch, plan):
                    item["limitations_cn"] = ensure_list(item.get("limitations_cn")) + [
                        f"LLM 详细分析失败，已回退到摘要初筛。错误：{shorten_text(str(exc), 180)}"
                    ]
                    analyzed_by_id[item.get("arxiv_id", item.get("title", ""))] = item
                batch_progress[batch_idx - 1]["status"] = "fallback"

        if progress is not None:
            progress.patch(
                message=f"第 {batch_idx}/{total_batches} 批分析完成",
                analysis={
                    "processed_papers": min(batch_idx * batch_size, len(papers)),
                    "current_titles": [],
                    "batches": batch_progress,
                },
            )

    result = []
    for paper in papers:
        arxiv_id = str(paper.get("arxiv_id", ""))
        analysis = analyzed_by_id.get(arxiv_id, {})
        enriched = dict(paper)
        enriched.update(
            {
                "relevance_score": int(analysis.get("relevance_score", paper.get("heuristic_score", 0)) or 0),
                "priority": analysis.get("priority") or "low",
                "summary_cn": analysis.get("summary_cn") or "",
                "key_points_cn": ensure_list(analysis.get("key_points_cn")),
                "innovation_ideas_cn": ensure_list(analysis.get("innovation_ideas_cn")),
                "why_relevant_cn": analysis.get("why_relevant_cn") or "",
                "limitations_cn": ensure_list(analysis.get("limitations_cn")),
                "matched_topics": ensure_list(analysis.get("matched_topics")),
            }
        )
        if enriched["relevance_score"] >= min_score:
            result.append(enriched)

    result.sort(key=lambda item: (item.get("relevance_score", 0), item.get("published", "")), reverse=True)
    if progress is not None:
        progress.patch(
            message=f"LLM 分析完成，保留 {len(result)} 篇",
            analysis={
                "processed_papers": len(papers),
                "current_titles": [],
                "batches": batch_progress,
            },
        )
    return result


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(str(item)) for item in value if clean_text(str(item))]
    return [clean_text(str(value))] if clean_text(str(value)) else []


def build_output(
    interest: str,
    plan: Dict[str, Any],
    papers: List[Dict[str, Any]],
    candidate_papers: List[Dict[str, Any]],
    llm: Optional[LLMClient],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "source": "arXiv",
            "user_interest": interest,
            "generated_at": now_iso(),
            "total_papers": len(papers),
            "llm_enabled": llm is not None,
            "llm_model": llm.model if llm else "",
            "sort_by": args.sort_by,
            "sort_order": args.sort_order,
            "candidate_results": resolve_candidate_results(args),
            "candidate_papers_collected": len(candidate_papers),
            "max_results": args.max_results,
            "from_date": args.from_date or "",
            "to_date": args.to_date or "",
            "min_score": args.min_score,
            "refine_rounds": args.refine_rounds,
            "viewer": "../arxiv_papers_viewer.html",
        },
        "search_plan": plan,
        "candidate_papers": candidate_papers,
        "papers": papers,
    }


def default_output_path(filename: str) -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search arXiv papers and summarize them in Chinese.")
    parser.add_argument("--interest", required=True, help="你的研究兴趣、关键词或自然语言问题。")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS, help="最终保留的论文数量。")
    parser.add_argument("--candidate-results", type=int, default=None, help="候选池规模，例如从 300 篇候选里筛出 --max-results 篇。默认等于 --max-results。")
    parser.add_argument("--per-query-max", type=int, default=None, help="每组优化检索词最多从 arXiv 拉取多少篇；不填时根据候选池自动估算。")
    parser.add_argument("--categories", default=None, help="逗号分隔的 arXiv 类别；传空字符串表示不限制类别。")
    parser.add_argument("--from-date", default=None, help="只检索此日期之后提交的论文，格式 YYYY-MM-DD。")
    parser.add_argument("--to-date", default=None, help="只检索此日期之前提交的论文，格式 YYYY-MM-DD。")
    parser.add_argument("--sort-by", default="relevance", choices=["relevance", "lastUpdatedDate", "submittedDate"])
    parser.add_argument("--sort-order", default="descending", choices=["ascending", "descending"])
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY, help="arXiv 请求间隔，建议不少于 3 秒。")
    parser.add_argument("--arxiv-timeout", type=int, default=60, help="arXiv 请求超时时间。")
    parser.add_argument("--screen-batch-size", type=int, default=20, help="候选论文轻量相关性筛选时每批交给 LLM 的论文数量。")
    parser.add_argument("--analysis-batch-size", type=int, default=6, help="每次交给 LLM 分析的论文数量。")
    parser.add_argument("--min-score", type=int, default=0, help="只保留 LLM 相关性评分不低于该值的论文。")
    parser.add_argument("--refine-rounds", type=int, default=0, help="迭代扩展检索轮数；0 表示只按初始 query 检索。")
    parser.add_argument("--refine-seed-size", type=int, default=20, help="每轮用多少篇高相关候选论文提炼下一轮检索词。")
    parser.add_argument("--no-llm", action="store_true", help="不调用 LLM，只抓 arXiv 并做关键词初筛。")
    parser.add_argument("--api-url", default=None, help="OpenAI-compatible base URL，例如 https://api.deepseek.com/v1")
    parser.add_argument("--api-key", default=None, help="LLM API key；更推荐用环境变量。")
    parser.add_argument("--model", default=None, help="LLM 模型名。")
    parser.add_argument("--llm-timeout", type=int, default=90, help="LLM 请求超时时间。")
    parser.add_argument("--output", default=None, help="最终 JSON 输出路径。")
    parser.add_argument("--raw-output", default=None, help="原始 arXiv 结果输出路径。")
    parser.add_argument("--progress-output", default=None, help="运行进度 JSON 输出路径，默认 arxiv_progress.json。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_results <= 0:
        raise ValueError("--max-results must be positive.")
    if args.candidate_results is not None and args.candidate_results <= 0:
        raise ValueError("--candidate-results must be positive.")
    if resolve_candidate_results(args) < args.max_results:
        raise ValueError("--candidate-results must be greater than or equal to --max-results.")
    if args.per_query_max is not None and args.per_query_max <= 0:
        raise ValueError("--per-query-max must be positive.")
    if args.screen_batch_size <= 0:
        raise ValueError("--screen-batch-size must be positive.")
    if args.analysis_batch_size <= 0:
        raise ValueError("--analysis-batch-size must be positive.")
    if args.refine_rounds < 0:
        raise ValueError("--refine-rounds must be zero or positive.")
    if args.refine_seed_size <= 0:
        raise ValueError("--refine-seed-size must be positive.")

    progress_output = Path(args.progress_output) if args.progress_output else default_output_path("arxiv_progress.json")
    progress = ProgressReporter(progress_output, args.interest)

    try:
        categories = None if args.categories is None else parse_csv(args.categories)
        llm = None if args.no_llm else LLMClient.from_args(args)
        if llm is None:
            print("未检测到 LLM 配置，将使用关键词初筛；中文总结和创新点会以占位说明形式输出。")
        else:
            print(f"Using LLM model: {llm.model}")

        progress.patch(
            message="任务已启动",
            llm_enabled=llm is not None,
            llm_model=llm.model if llm else "",
            candidate_results=resolve_candidate_results(args),
            max_results=args.max_results,
        )

        print("\nOptimizing search plan...")
        progress.stage("optimize", "正在让 LLM 优化 arXiv 检索词")
        plan = optimize_search_plan(args.interest, llm, categories)
        progress.patch(
            message=f"检索计划已生成，共 {len(plan.get('search_queries', []))} 组 query",
            search_plan=plan,
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2))

        print("\nSearching arXiv...")
        progress.stage("search", "正在按检索计划逐组搜索 arXiv")
        raw_papers = search_arxiv(plan, args, progress=progress)
        print(f"\nCollected {len(raw_papers)} candidate arXiv papers.")

        candidate_papers = raw_papers
        progress.patch(
            refinement={
                "enabled": args.refine_rounds > 0 and llm is not None,
                "current_round": 0,
                "total_rounds": args.refine_rounds,
                "rounds": [],
            }
        )

        if args.refine_rounds > 0 and llm is None:
            print("Refine rounds requested but LLM is disabled or not configured; skipping refinement.")

        for refine_round in range(1, args.refine_rounds + 1):
            if llm is None:
                break
            progress.stage("screen", f"第 {refine_round} 轮扩展检索前，正在筛选高相关种子论文")
            seed_keep = max(args.max_results, min(args.refine_seed_size, len(candidate_papers)))
            seed_papers = screen_candidates_with_llm(
                candidate_papers,
                plan,
                args.interest,
                llm,
                keep_count=seed_keep,
                batch_size=args.screen_batch_size,
                progress=progress,
            )[: args.refine_seed_size]
            progress.stage("refine", f"正在生成第 {refine_round}/{args.refine_rounds} 轮扩展检索词")
            refined_plan = build_refined_search_plan(
                args.interest,
                plan,
                seed_papers,
                llm,
                categories,
                round_index=refine_round,
            )
            if refined_plan is None:
                progress.patch(
                    message=f"第 {refine_round} 轮扩展检索词生成失败，停止迭代",
                    refinement={
                        "current_round": refine_round,
                        "rounds": [
                            {
                                "index": refine_round,
                                "status": "failed",
                                "seed_count": len(seed_papers),
                            }
                        ],
                    },
                )
                break

            existing = {
                paper.get("arxiv_id") or paper.get("title", ""): paper
                for paper in candidate_papers
                if paper.get("arxiv_id") or paper.get("title", "")
            }
            progress.patch(
                message=f"第 {refine_round}/{args.refine_rounds} 轮扩展检索词已生成，继续搜索 arXiv",
                refinement={
                    "current_round": refine_round,
                    "rounds": [
                        {
                            "index": refine_round,
                            "status": "running",
                            "seed_count": len(seed_papers),
                            "search_plan": refined_plan,
                        }
                    ],
                },
            )
            progress.stage("search", f"正在执行第 {refine_round}/{args.refine_rounds} 轮扩展检索")
            candidate_papers = search_arxiv(
                refined_plan,
                args,
                progress=progress,
                existing_papers=existing,
                round_label=f"扩展检索第 {refine_round} 轮",
                round_index=refine_round + 1,
            )
            # Expand positive/negative keywords for later ranking and analysis.
            plan["positive_keywords"] = list(dict.fromkeys(plan.get("positive_keywords", []) + refined_plan.get("positive_keywords", [])))
            plan["negative_keywords"] = list(dict.fromkeys(plan.get("negative_keywords", []) + refined_plan.get("negative_keywords", [])))
            plan.setdefault("refined_search_plans", []).append(refined_plan)
            progress.patch(
                message=f"第 {refine_round}/{args.refine_rounds} 轮扩展检索完成，候选池 {len(candidate_papers)} 篇",
                refinement={
                    "current_round": refine_round,
                    "rounds": [
                        {
                            "index": refine_round,
                            "status": "completed",
                            "seed_count": len(seed_papers),
                            "candidate_count": len(candidate_papers),
                            "search_plan": refined_plan,
                        }
                    ],
                },
            )

        raw_output = Path(args.raw_output) if args.raw_output else default_output_path("arxiv_raw_papers.json")
        save_json(
            raw_output,
            {
                "metadata": {
                    "source": "arXiv",
                    "user_interest": args.interest,
                    "generated_at": now_iso(),
                    "total_papers": len(candidate_papers),
                    "candidate_results": resolve_candidate_results(args),
                    "max_results": args.max_results,
                },
                "search_plan": plan,
                "papers": candidate_papers,
            },
        )
        progress.patch(raw_output=str(raw_output))
        print(f"Raw arXiv data saved to: {raw_output}")

        print("\nScreening candidates...")
        progress.stage("screen", f"正在从 {len(candidate_papers)} 篇候选中筛选最相关的 {args.max_results} 篇")
        selected_papers = screen_candidates_with_llm(
            candidate_papers,
            plan,
            args.interest,
            llm,
            keep_count=args.max_results,
            batch_size=args.screen_batch_size,
            progress=progress,
        )

        print("\nAnalyzing papers...")
        progress.stage("analyze", f"正在用 LLM 详细分析筛选出的 {len(selected_papers)} 篇论文摘要")
        analyzed_papers = analyze_papers_with_llm(
            selected_papers,
            plan,
            args.interest,
            llm,
            batch_size=args.analysis_batch_size,
            min_score=args.min_score,
            progress=progress,
        )

        output = Path(args.output) if args.output else default_output_path("arxiv_research_results.json")
        progress.stage("save", "正在保存最终 JSON")
        save_json(output, build_output(args.interest, plan, analyzed_papers, candidate_papers, llm, args))
        progress.finish(f"任务完成，最终保留 {len(analyzed_papers)} 篇论文", output)
        print(f"\nFinal JSON saved to: {output}")
        print(f"Viewer file: {output.parent / 'arxiv_papers_viewer.html'}")
        print(f"Progress file: {progress_output}")
        return 0
    except KeyboardInterrupt:
        progress.patch(status="interrupted", stage="interrupted", message="用户中断了任务")
        raise
    except Exception as exc:
        progress.fail(str(exc))
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
