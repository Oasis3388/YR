# 储罐管理 select_skill
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import date
from difflib import SequenceMatcher, get_close_matches
from html import escape
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from skills.base_skill import BaseSkill, SkillResult
from tools.http.tank_area_api import TankAreaAPI
from tools.websocket.common_api import (
    _gen_clear_session,
    gen_jump_data,
    gen_ws_list_data,
)


# ─────────────────────────────────────────────────────────────
# Skill 基础配置
# ─────────────────────────────────────────────────────────────
_SKILL_FILE = os.path.abspath(__file__)

# 前端真实配置
# 单条精确查询：跳转详情页 /system/tankAreaDetail?id=xxx
# 多条结果/候选结果：仿照维保计划查询，跳转管理列表页 /system/tankAreaManage，
# 并携带 listTimer 刷新列表缓存。
#
# 前端依赖：
# 1. 已有详情页：/system/tankAreaDetail，接收 query.id
# 2. 已有管理列表页：/system/tankAreaManage，读取 fsTankAreaDelOrSearchList
# 当前查询 Skill 不新增前端页面。
_DETAIL_JUMP_ROUTE = "/system/tankAreaDetail"
_LIST_JUMP_ROUTE = "/system/tankAreaManage"
_JUMP_ROUTE = _LIST_JUMP_ROUTE

_URL = "/system/tankAreaManage"
_CACHE_NAME = "fsTankAreaDelOrSearchList"

_QUERY_PAGE_SIZE = 10000
_CHAT_DISPLAY_LIMIT = 20
_CANDIDATE_LIMIT = 10
_LAST_LIST_TIMER = 0


# ─────────────────────────────────────────────────────────────
# 字段定义：字段名与后端 /light/sysTankArea/list 返回字段保持一致
# ─────────────────────────────────────────────────────────────
_TANK_AREA_FIELDS = {
    "id",
    "areaName",
    "areaLocation",
    "excludeAreas",
    "createBy",
    "createTime",
    "updateTime",
    "updateBy",
    "tankDetailList",
    "excludeAreasName",
    "tankCodesStr",
}

_FIELD_LABELS = {
    "id": "区域ID",
    "areaName": "区域名称",
    "areaLocation": "区域位置",
    "excludeAreas": "排斥区域ID",
    "createBy": "创建人",
    "createTime": "创建时间",
    "updateTime": "更新时间",
    "updateBy": "更新人",
    "tankDetailList": "储罐详情列表",
    "excludeAreasName": "排斥区域",
    "tankCodesStr": "储罐编号",
}

# 兼容历史 slots / 口语字段名，最终统一成后端字段名。
_FIELD_ALIASES = {
    "name": "areaName",
    "area_name": "areaName",
    "storage_tank_area_name": "areaName",
    "storage_tank_area_creator": "createBy",
    "storage_tank_area_create_time": "createTime",
    "storage_tank_area_update_time": "updateTime",
    "storage_tank_area_position": "areaLocation",
    "exclude_storage_tank_area": "excludeAreasName",
    "excludeAreaName": "excludeAreasName",
    "excludeAreaNames": "excludeAreasName",
    "excludeAreasNames": "excludeAreasName",
    "tankCode": "tankCodesStr",
    "tankCodes": "tankCodesStr",
}

_TEXT_MATCH_FIELDS = {
    "id",
    "areaName",
    "areaLocation",
    "excludeAreas",
    "createBy",
    "updateBy",
    "excludeAreasName",
    "tankCodesStr",
}
_DATE_FIELDS = {"createTime", "updateTime"}
_FILTER_OPS = {"contains", "eq", "ne", "empty", "not_empty", "range"}

_DEFAULT_DISPLAY_FIELDS = [
    "areaName",
    "areaLocation",
    "excludeAreasName",
    "tankCodesStr",
    "createBy",
    "createTime",
    "updateTime",
]

# 模糊候选优先匹配这些字段，确保用户输入错字或意图不清晰时，页面能看到可能要查询的信息。
_CANDIDATE_MATCH_FIELDS = [
    "areaName",
    "excludeAreasName",
    "tankCodesStr",
    "createBy",
    "updateBy",
    "areaLocation",
    "id",
]

_GENERIC_QUERY_WORDS = {
    "查",
    "查询",
    "查看",
    "帮我查",
    "帮我查询",
    "储罐区域",
    "储罐管理",
    "罐区",
    "区域",
    "信息",
    "列表",
    "内容",
}


# ─────────────────────────────────────────────────────────────
# LLM 工具：与维保计划查询 Skill 保持同一套调用方式
# ─────────────────────────────────────────────────────────────
def _get_llm():
    """获取 LLM 实例，与 main.py 走同一套配置。"""
    from openai import OpenAI

    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def _call_llm_sync(system: str, user: str) -> str:
    """同步 LLM 调用，返回纯文本。"""
    client = _get_llm()
    model = os.getenv("OPENAI_INTENT_MODEL") or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    extra = {}
    if os.getenv("DISABLE_THINKING", "false").lower() == "true":
        extra = {"chat_template_kwargs": {"enable_thinking": False}}

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        extra_body=extra or None,
    )
    return (resp.choices[0].message.content or "").strip()


def _strip_json_fence(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


# ─────────────────────────────────────────────────────────────
# 通用格式化与匹配工具
# ─────────────────────────────────────────────────────────────
def _short_label(field: str) -> str:
    return _FIELD_LABELS.get(field, field)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _clean_text(value: Any) -> str:
    text = _safe_text(value).strip()
    if text.lower() in {"none", "null", "undefined"}:
        return ""
    return text


def _display_text(value: Any, max_len: int = 90) -> str:
    text = _clean_text(value)
    if not text:
        return "-"
    text = text.replace("\r", " ").replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _normalize_for_match(value: Any) -> str:
    """用于候选匹配的轻量归一化，降低错字、符号、空格对匹配的影响。"""
    text = _clean_text(value).lower()
    if not text:
        return ""
    text = re.sub(r"[\s,，。.;；:：/\\|_\-—()（）\[\]【】{}<>《》'\"“”]+", "", text)
    for word in [
        "帮我", "请", "麻烦", "查询", "查看", "查一下", "查", "看一下", "储罐管理",
        "储罐区域", "储罐", "罐区", "区域", "信息", "列表", "内容", "这个", "一下",
        "的", "是", "吗", "呢", "在哪", "哪里", "哪个", "哪些", "所有", "全部",
    ]:
        text = text.replace(word, "")
    return text


def _build_today_context() -> Dict[str, str]:
    from datetime import timedelta

    today = date.today()
    yesterday = today - timedelta(days=1)
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    return {
        "today": today.isoformat(),
        "yesterday": yesterday.isoformat(),
        "week_start": week_start.isoformat(),
        "month_start": month_start.isoformat(),
    }


def _is_generic_query(text: str) -> bool:
    cleaned = _normalize_for_match(text)
    if not cleaned:
        return True
    raw = _clean_text(text)
    # 只包含泛化词时认为参数缺失；只要带具体名称/编号/人员/日期，就不认为泛化。
    tmp = raw
    for word in _GENERIC_QUERY_WORDS:
        tmp = tmp.replace(word, "")
    tmp = re.sub(r"[\s,，。.!！?？的了呢一下]+", "", tmp)
    return len(tmp) == 0


def _direct_text_match(current: Any, expected: Any, op: str) -> bool:
    current_text = _clean_text(current)
    expected_text = _clean_text(expected)

    if op == "empty":
        return current in (None, "", [], {}) or current_text == ""
    if op == "not_empty":
        return current not in (None, "", [], {}) and current_text != ""
    if op == "eq":
        return current_text == expected_text
    if op == "ne":
        return current_text != expected_text
    if not expected_text:
        return True

    current_lower = current_text.lower()
    expected_lower = expected_text.lower()
    if expected_lower in current_lower or current_lower in expected_lower:
        return True
    return SequenceMatcher(None, current_lower, expected_lower).ratio() >= 0.72


def _is_strict_text_match(current: Any, expected: Any) -> bool:
    current_text = _clean_text(current).lower()
    expected_text = _clean_text(expected).lower()
    if not current_text or not expected_text:
        return False
    return expected_text in current_text or current_text in expected_text


def _value_in_range(current: Any, start: Any, end: Any) -> bool:
    current_text = _clean_text(current)[:10]
    start_text = _clean_text(start)[:10]
    end_text = _clean_text(end)[:10]

    if not current_text:
        return False
    if start_text and current_text < start_text:
        return False
    if end_text and current_text > end_text:
        return False
    return True


def _semantic_match_values(field: str, expected: Any, candidates: List[Any]) -> set[str]:
    """对错别字、简称、口语化表达做候选值匹配；LLM 只从真实候选中选择，不允许编造。"""
    expected_text = _clean_text(expected)
    values = []
    for item in candidates:
        text = _clean_text(item)
        if text and text not in values:
            values.append(text)

    if not expected_text or not values:
        return set()

    expected_norm = _normalize_for_match(expected_text)
    direct = set()
    for value in values:
        value_norm = _normalize_for_match(value)
        if not value_norm:
            continue
        if expected_norm and (expected_norm in value_norm or value_norm in expected_norm):
            direct.add(value)
            continue
        ratio = SequenceMatcher(None, value_norm, expected_norm).ratio() if expected_norm else 0
        if ratio >= 0.66:
            direct.add(value)
    if direct:
        return direct

    close = set(get_close_matches(expected_text, values, n=8, cutoff=0.55))
    if close:
        return close

    try:
        system = """
你是储罐区域查询候选匹配助手。用户输入可能有错别字、简称或模糊语义。
请只从候选值中选择语义上匹配用户输入的值，不要编造。
只输出 JSON：{"matchedValues": []}
""".strip()
        user = (
            f"字段：{_short_label(field)}\n"
            f"用户输入：{expected_text}\n"
            f"候选值：{json.dumps(values[:80], ensure_ascii=False)}"
        )
        raw = _call_llm_sync(system, user)
        data = json.loads(_strip_json_fence(raw))
        matched = {
            str(item).strip()
            for item in data.get("matchedValues", [])
            if str(item).strip() in values
        }
        if matched:
            logger.info(
                f"[StorageTankAreaSelectSkill] LLM 模糊匹配字段={field}, "
                f"expected={expected_text}, matched={matched}"
            )
        return matched
    except Exception as e:
        logger.warning(f"[StorageTankAreaSelectSkill] LLM 模糊匹配失败 field={field}: {e}")
        return set()


def _record_matches_filter(
    record: Dict[str, Any],
    flt: Dict[str, Any],
    semantic_values: Optional[set[str]] = None,
) -> bool:
    field = flt.get("field")
    op = flt.get("op") or "contains"

    if field == "__any__":
        expected = flt.get("value")
        expected_norm = _normalize_for_match(expected)
        haystack = " ".join(_safe_text(record.get(k)) for k in _CANDIDATE_MATCH_FIELDS)
        haystack_norm = _normalize_for_match(haystack)
        if expected_norm and (expected_norm in haystack_norm or haystack_norm in expected_norm):
            return True
        return _direct_text_match(haystack, expected, "contains")

    if op == "range":
        return _value_in_range(record.get(field), flt.get("start"), flt.get("end"))

    if semantic_values is not None and op in {"contains", "eq"}:
        return _clean_text(record.get(field)) in semantic_values

    return _direct_text_match(record.get(field), flt.get("value"), op)


def _apply_filters(records: List[Dict[str, Any]], filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not filters:
        return records

    filtered = records
    for flt in filters:
        before = len(filtered)
        field = flt.get("field")
        op = flt.get("op") or "contains"
        semantic_values = None

        if field in _TEXT_MATCH_FIELDS and op in {"contains", "eq"}:
            semantic_values = _semantic_match_values(
                field,
                flt.get("value"),
                [r.get(field) for r in filtered],
            )

        filtered = [
            record for record in filtered
            if _record_matches_filter(record, flt, semantic_values)
        ]
        logger.info(f"[StorageTankAreaSelectSkill] 后过滤 {before} -> {len(filtered)}, filter={flt}")

    return filtered


def _dedupe_by_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for idx, record in enumerate(records or []):
        key = record.get("id") or f"idx-{idx}"
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def _field_name(field: Any) -> str:
    if not field:
        return ""
    field = str(field).strip()
    return _FIELD_ALIASES.get(field, field)


# ─────────────────────────────────────────────────────────────
# 查询计划生成：先由 LLM 生成查询计划，再 run 真实查接口
# ─────────────────────────────────────────────────────────────
def _build_query_plan_prompt(context: Optional[Dict[str, Any]] = None) -> str:
    ctx = _build_today_context()
    runtime = context or {}
    fields = sorted(_TANK_AREA_FIELDS)
    field_labels = {field: _short_label(field) for field in fields}

    return f"""
你是储罐区域查询编排助手。你的任务不是回答用户，而是把用户的话转换成可执行查询计划。

【当前运行上下文】
- 当前日期：{ctx["today"]}
- 昨天：{ctx["yesterday"]}
- 本周开始：{ctx["week_start"]}
- 本月开始：{ctx["month_start"]}
- 当前用户：{runtime.get("user_name") or runtime.get("username") or "未知"}

【后端接口】
GET /light/sysTankArea/list

【字段列表】
字段名必须使用后端返回字段名：
{json.dumps(fields, ensure_ascii=False)}

【字段含义】
{json.dumps(field_labels, ensure_ascii=False)}

【输出格式】
只能输出纯 JSON，不要 markdown，不要解释：
{{
  "filters": [],
  "query_mode": "targeted",
  "display_fields": [],
  "reply_hint": ""
}}

【filters 格式】
filters 是数组，每个过滤条件格式如下：
{{"field":"areaName","op":"contains","value":"罐区一"}}

规则：
- field 只能取字段列表中的字段，或者 "__any__"。
- op 只能是 contains、eq、ne、empty、not_empty、range。
- 文本类字段默认使用 contains。
- 日期时间范围使用 range，例如：{{"field":"createTime","op":"range","start":"2026-05-25","end":"2026-05-25"}}。
- display_fields 放用户关心的字段，最多 6 个；为空时系统自动选择。
- “全部/所有/全量查询/查看储罐区域列表/查看所有储罐区域”且没有条件时，query_mode="all_explicit"。
- 只说“查询储罐区域”“查罐区信息”等完全泛泛的请求，且没有明确说查全部时，query_mode="need_clarify"。
- 用户输入有错字、简称、不完整名称、模糊描述，但仍包含可能的名称/编号/人员/日期时，不要 need_clarify；尽量生成最可能字段过滤条件。
- 如果不确定字段，但用户确实给了关键词，使用 field="__any__"、op="contains"、value=用户关键词，让系统做候选匹配。
- 有任何有效条件时，query_mode="targeted"。

【业务理解】
1. “罐区/储罐区域/区域名称/叫 xxx 的区域” -> areaName。
2. “区域位置/坐标/地图位置” -> areaLocation。
3. “排斥区域/互斥区域/不能同时出现的区域” -> 优先 excludeAreasName；如果用户给的是长数字 ID，则用 excludeAreas。
4. “创建人/谁创建的” -> createBy。
5. “创建时间/哪天创建/今天创建/昨天创建” -> createTime。
6. “更新人/谁更新的” -> updateBy。
7. “更新时间/哪天更新/最近更新” -> updateTime。
8. “储罐编号/罐号/包含哪个储罐/关联储罐” -> tankCodesStr。
9. 用户问“罐区一的排斥区域是什么”，filters 用 areaName，display_fields 放 excludeAreasName。
10. 用户问“罐区一在哪”，filters 用 areaName，display_fields 放 areaLocation。
11. 用户问“哪些区域排斥区域是罐区3”，filters 用 excludeAreasName。
12. 用户问“哪些区域包含 zy test 或 888888”，filters 用 tankCodesStr。
13. 不要创造字段；不确定字段时用 "__any__" 做模糊查询。

【示例】
用户：帮我查罐区一
输出：{{"filters":[{{"field":"areaName","op":"contains","value":"罐区一"}}],"query_mode":"targeted","display_fields":[],"reply_hint":""}}

用户：罐区3的排斥区域是什么
输出：{{"filters":[{{"field":"areaName","op":"contains","value":"罐区3"}}],"query_mode":"targeted","display_fields":["excludeAreasName"],"reply_hint":""}}

用户：查询创建人是 sysadmin 的储罐区域
输出：{{"filters":[{{"field":"createBy","op":"contains","value":"sysadmin"}}],"query_mode":"targeted","display_fields":["createBy"],"reply_hint":""}}

用户：查 2026-05-25 创建的储罐区域
输出：{{"filters":[{{"field":"createTime","op":"range","start":"2026-05-25","end":"2026-05-25"}}],"query_mode":"targeted","display_fields":["createTime"],"reply_hint":""}}

用户：查看全部储罐区域
输出：{{"filters":[],"query_mode":"all_explicit","display_fields":[],"reply_hint":""}}

用户：查一下储罐区域
输出：{{"filters":[],"query_mode":"need_clarify","display_fields":[],"reply_hint":""}}

用户：帮我看下 zy测式
输出：{{"filters":[{{"field":"__any__","op":"contains","value":"zy测式"}}],"query_mode":"targeted","display_fields":[],"reply_hint":""}}
""".strip()


def _normalize_filter_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    field = _field_name(item.get("field"))
    if not field:
        return None
    if field != "__any__" and field not in _TANK_AREA_FIELDS:
        return None

    op = item.get("op") or "contains"
    if op not in _FILTER_OPS:
        op = "contains"

    clean: Dict[str, Any] = {"field": field, "op": op}
    for key in ("value", "start", "end"):
        value = item.get(key)
        if value is not None and value != "":
            clean[key] = value

    if op == "range":
        if not clean.get("start") and clean.get("value"):
            clean["start"] = clean["value"]
        if not clean.get("end") and clean.get("value"):
            clean["end"] = clean["value"]
        if not clean.get("start") and not clean.get("end"):
            return None
    elif op not in {"empty", "not_empty"} and clean.get("value") in (None, ""):
        return None

    return clean


def _normalize_query_plan(plan: Dict[str, Any], latest_input: str = "") -> Dict[str, Any]:
    """兜底清洗 LLM 查询计划，保证后续只处理允许字段和允许操作符。"""
    if not isinstance(plan, dict):
        plan = {}

    raw_filters = plan.get("filters") or []
    if isinstance(raw_filters, dict):
        raw_filters = [
            {"field": _field_name(field), "op": "contains", "value": value}
            for field, value in raw_filters.items()
        ]

    filters: List[Dict[str, Any]] = []
    for item in raw_filters:
        clean = _normalize_filter_item(item)
        if clean:
            filters.append(clean)

    query_mode = str(plan.get("query_mode") or "").strip().lower()
    if query_mode not in {"targeted", "all_explicit", "need_clarify"}:
        query_mode = "targeted" if filters else "need_clarify"

    # LLM 没抽出字段，但用户不是泛泛请求时，用 __any__ 兜底匹配，保证左侧能显示可能要查询的信息。
    if not filters and query_mode != "all_explicit" and latest_input and not _is_generic_query(latest_input):
        filters = [{"field": "__any__", "op": "contains", "value": latest_input}]
        query_mode = "targeted"

    display_fields = []
    for field in plan.get("display_fields") or []:
        field = _field_name(field)
        if field in _TANK_AREA_FIELDS and field != "tankDetailList" and field not in display_fields:
            display_fields.append(field)

    return {
        "filters": filters,
        "query_mode": query_mode,
        "display_fields": display_fields[:6],
        "reply_hint": plan.get("reply_hint") or "",
    }


def _build_query_plan(user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    latest_input = (user_input or "").split("用户补充：")[-1].strip()
    try:
        raw = _call_llm_sync(_build_query_plan_prompt(context), latest_input)
        plan = json.loads(_strip_json_fence(raw))
        normalized = _normalize_query_plan(plan, latest_input)
        logger.info(f"[StorageTankAreaSelectSkill] 查询计划：{normalized}")
        return normalized
    except Exception as e:
        logger.warning(f"[StorageTankAreaSelectSkill] 生成查询计划失败：{e}")
        return _normalize_query_plan({}, latest_input)


# ─────────────────────────────────────────────────────────────
# 后端接口调用：统一通过 tools/http/tank_area_api.py，不在 Skill 中直接写 requests
# ─────────────────────────────────────────────────────────────
def _fetch_tank_area_page() -> Dict[str, Any]:
    api = TankAreaAPI()
    api.login()

    if hasattr(api, "get_tank_area_page"):
        return api.get_tank_area_page(pageNo=1, pageSize=_QUERY_PAGE_SIZE)

    # 兼容旧版 tank_area_api.py
    records = api.get_tank_area_records(pageNo=1, pageSize=_QUERY_PAGE_SIZE)
    return {
        "records": records,
        "total": len(records),
        "size": _QUERY_PAGE_SIZE,
        "current": 1,
        "pages": 1 if records else 0,
    }


def _fetch_all_tank_area_records() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    page_result = _fetch_tank_area_page()
    records = page_result.get("records") or []
    if not isinstance(records, list):
        records = []

    records = [r for r in records if isinstance(r, dict)]
    logger.info(
        f"[StorageTankAreaSelectSkill] 已加载储罐区域 records={len(records)}, "
        f"backend_total={page_result.get('total')}"
    )
    return records, page_result


# ─────────────────────────────────────────────────────────────
# 候选匹配：没有精确结果时，仍返回可能要查询的信息
# ─────────────────────────────────────────────────────────────
def _filter_query_terms(plan: Dict[str, Any]) -> List[Tuple[str, str]]:
    terms: List[Tuple[str, str]] = []
    for flt in plan.get("filters") or []:
        field = flt.get("field") or "__any__"
        if flt.get("op") == "range":
            start = _clean_text(flt.get("start"))
            end = _clean_text(flt.get("end"))
            if start:
                terms.append((field, start))
            if end and end != start:
                terms.append((field, end))
        else:
            value = _clean_text(flt.get("value"))
            if value:
                terms.append((field, value))
    return terms


def _similarity_score(query: str, value: Any) -> float:
    q = _normalize_for_match(query)
    v = _normalize_for_match(value)
    if not q or not v:
        return 0.0
    if q == v:
        return 1.0
    if q in v or v in q:
        return 0.92
    ratio = SequenceMatcher(None, q, v).ratio()
    q_chars = set(q)
    v_chars = set(v)
    overlap = len(q_chars & v_chars) / max(len(q_chars), 1)
    return max(ratio, overlap * 0.75)


def _find_candidate_records(records: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    terms = _filter_query_terms(plan)
    if not terms:
        return []

    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx, record in enumerate(records or []):
        best = 0.0
        for field, query in terms:
            fields = _CANDIDATE_MATCH_FIELDS if field == "__any__" else [field]
            # 字段抽错时，也允许扫一遍关键字段。
            if field != "__any__":
                fields = list(dict.fromkeys(fields + ["areaName", "excludeAreasName", "tankCodesStr"]))
            for candidate_field in fields:
                best = max(best, _similarity_score(query, record.get(candidate_field)))
        if best >= 0.46:
            scored.append((best, idx, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [record for _, _, record in scored[:_CANDIDATE_LIMIT]]
    logger.info(f"[StorageTankAreaSelectSkill] 候选匹配数量={len(candidates)}, terms={terms}")
    return candidates


def _build_match_notes(records: List[Dict[str, Any]], plan: Dict[str, Any], candidate_mode: bool) -> List[str]:
    notes: List[str] = []

    if candidate_mode:
        notes.append("没有找到完全一致的记录，以下是系统根据您的描述匹配到的可能查询信息，请您核对一下。")

    for flt in plan.get("filters") or []:
        field = flt.get("field")
        op = flt.get("op")
        if field in (None, "__any__", "tankDetailList") or op not in {"contains", "eq"}:
            continue
        expected = _clean_text(flt.get("value"))
        if not expected:
            continue
        for record in records[:3]:
            actual = _clean_text(record.get(field))
            if not actual:
                continue
            if _is_strict_text_match(actual, expected):
                continue
            if _similarity_score(expected, actual) >= 0.46:
                note = f"系统里原有的{_short_label(field)}是{actual}，不是{expected}，您记得核对一下。"
                if note not in notes:
                    notes.append(note)
                break

    return notes[:4]


def _execute_query_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    records, page_result = _fetch_all_tank_area_records()
    records = _dedupe_by_id(records)

    filters = plan.get("filters") or []
    if plan.get("query_mode") == "all_explicit":
        exact_records = records
    else:
        exact_records = _apply_filters(records, filters)

    candidate_records: List[Dict[str, Any]] = []
    candidate_mode = False
    display_records = exact_records
    if not exact_records and filters:
        candidate_records = _find_candidate_records(records, plan)
        if candidate_records:
            candidate_mode = True
            display_records = candidate_records

    match_notes = _build_match_notes(display_records, plan, candidate_mode)

    logger.info(
        f"[StorageTankAreaSelectSkill] 查询执行完成 records={len(records)}, "
        f"exact_records={len(exact_records)}, candidate_records={len(candidate_records)}"
    )

    return {
        "success": True,
        "records": display_records,
        "exact_records": exact_records,
        "candidate_records": candidate_records,
        "candidate_mode": candidate_mode,
        "match_notes": match_notes,
        "total": len(display_records),
        "exact_total": len(exact_records),
        "backend_total": page_result.get("total", len(records)),
        "page": page_result,
    }


# ─────────────────────────────────────────────────────────────
# 结果展示与前端同步
# ─────────────────────────────────────────────────────────────
def _next_list_timer() -> int:
    global _LAST_LIST_TIMER
    now = int(time.time() * 1000)
    _LAST_LIST_TIMER = max(now, _LAST_LIST_TIMER + 1)
    return _LAST_LIST_TIMER


def _query_extra_fields(filters: List[Dict[str, Any]], display_fields: List[str], candidate_mode: bool = False) -> List[str]:
    """
    计算“待查询字段”中要展示的字段。

    固定表格列只保留：
    - 单条：区域名称、区域位置、排斥区域、待查询字段
    - 多条：序号、区域名称、区域位置、排斥区域、待查询字段

    其中“待查询字段”列把用户关心的字段合并展示，避免 H5 窄屏下表头和字段值被挤成竖排。
    """
    base_fields = {"areaName", "areaLocation", "excludeAreasName", "excludeAreas", "id", "tankDetailList"}
    extras: List[str] = []

    for field in display_fields or []:
        field = _field_name(field)
        if field in _TANK_AREA_FIELDS and field not in base_fields and field not in extras:
            extras.append(field)

    for flt in filters or []:
        field = _field_name(flt.get("field"))
        if field and field != "__any__" and field in _TANK_AREA_FIELDS and field not in base_fields and field not in extras:
            extras.append(field)

    # 如果用户没有明确问具体字段，则给一个简洁默认字段，避免“待查询字段”为空。
    if not extras:
        if candidate_mode:
            extras = ["tankCodesStr", "createBy"]
        else:
            extras = ["tankCodesStr"]

    return extras[:4]


def _filters_to_display_keys(filters: List[Dict[str, Any]], display_fields: List[str], candidate_mode: bool = False) -> List[str]:
    """
    兼容旧调用：实际 H5 表格采用固定列结构。
    """
    return ["areaName", "areaLocation", "excludeAreasName"] + _query_extra_fields(filters, display_fields, candidate_mode)


def _table_cell_style(is_header: bool = False, align_center: bool = False) -> str:
    if is_header:
        return (
            "padding:6px 8px;"
            "border:0.5px solid rgba(65,109,241,0.20);"
            "background:#416DF1;"
            "color:#ffffff;"
            "font-size:12px;"
            "font-weight:600;"
            "line-height:1.4;"
            "text-align:center;"
            "white-space:nowrap;"
        )
    return (
        "padding:6px 8px;"
        "border:0.5px solid rgba(65,109,241,0.20);"
        "background:#ffffff;"
        "color:#2f3440;"
        "font-size:12px;"
        "line-height:1.5;"
        f"text-align:{'center' if align_center else 'left'};"
        "vertical-align:top;"
        "word-break:break-all;"
    )


def _build_pending_field_text(record: Dict[str, Any], extra_fields: List[str]) -> str:
    """
    生成“待查询字段”列内容。
    多个字段使用中文分隔，避免增加太多列导致 H5 表格挤压。
    """
    parts: List[str] = []
    for field in extra_fields or []:
        if field not in _TANK_AREA_FIELDS or field == "tankDetailList":
            continue
        value = _display_text(record.get(field), 70)
        parts.append(f"{_short_label(field)}：{value}")

    if not parts:
        return "-"

    return "；".join(parts)


def _h5_table_cell_style(is_header: bool = False, align_center: bool = False, width: str = "") -> str:
    base = [
        "box-sizing:border-box",
        "border:0.5px solid rgba(65,109,241,0.20)",
        "padding:7px 8px",
        "font-size:12px",
        "line-height:1.55",
        "vertical-align:top",
        "word-break:break-all",
        "white-space:normal",
    ]
    if width:
        base.append(f"width:{width}")
    if align_center:
        base.append("text-align:center")
    else:
        base.append("text-align:left")

    if is_header:
        base.extend([
            "background:#416DF1",
            "color:#ffffff",
            "font-weight:600",
            "white-space:nowrap",
        ])
    else:
        base.extend([
            "background:#ffffff",
            "color:#1F2937",
            "font-weight:400",
        ])
    return ";".join(base) + ";"


def _build_query_table_html(
    records: List[Dict[str, Any]],
    query_keywords: List[str],
    show_index: bool = True,
) -> str:
    """
    H5 横向表格展示，格式对齐 SKILL.md 模板：

    单条：
    | 区域名称 | 区域位置 | 排斥区域 | 待查询字段 |

    多条：
    | 序号 | 区域名称 | 区域位置 | 排斥区域 | 待查询字段 |
    """
    if not records:
        return "<div>无数据</div>"

    # 从 query_keywords 里还原“待查询字段”要合并展示的字段。
    base_fields = {"areaName", "areaLocation", "excludeAreasName", "excludeAreas"}
    extra_fields = [field for field in query_keywords if field not in base_fields and field in _TANK_AREA_FIELDS]

    html: List[str] = []
    html.append(
        '<div style="margin-top:10px;'
        'border:0.5px solid rgba(65,109,241,0.20);'
        'border-radius:8px;'
        'overflow:hidden;'
        'background:#ffffff;'
        'box-shadow:0 1px 3px rgba(15,23,42,0.06);'
        '">'
    )
    html.append('<table style="width:100%;border-collapse:collapse;table-layout:fixed;">')

    html.append("<thead><tr>")
    if show_index:
        html.append(f'<th style="{_h5_table_cell_style(True, True, "42px")}">序号</th>')
        area_width = "70px"
        loc_width = "34%"
        exclude_width = "82px"
        pending_width = "26%"
    else:
        area_width = "76px"
        loc_width = "42%"
        exclude_width = "82px"
        pending_width = "28%"

    html.append(f'<th style="{_h5_table_cell_style(True, False, area_width)}">区域名称</th>')
    html.append(f'<th style="{_h5_table_cell_style(True, False, loc_width)}">区域位置</th>')
    html.append(f'<th style="{_h5_table_cell_style(True, False, exclude_width)}">排斥区域</th>')
    html.append(f'<th style="{_h5_table_cell_style(True, False, pending_width)}">待查询字段</th>')
    html.append("</tr></thead>")

    html.append("<tbody>")
    for idx, record in enumerate(records[:_CHAT_DISPLAY_LIMIT], start=1):
        row_bg = "#ffffff" if idx % 2 else "#F7F9FF"
        html.append(f'<tr style="background:{row_bg};">')

        if show_index:
            html.append(
                f'<td style="{_h5_table_cell_style(False, True)}background:{row_bg};">{idx}</td>'
            )

        area_name = escape(_display_text(record.get("areaName"), 45))
        area_location = escape(_display_text(record.get("areaLocation"), 120))
        exclude_name = escape(_display_text(record.get("excludeAreasName"), 70))
        pending_text = escape(_build_pending_field_text(record, extra_fields))

        html.append(f'<td style="{_h5_table_cell_style(False)}background:{row_bg};">{area_name}</td>')
        html.append(f'<td style="{_h5_table_cell_style(False)}background:{row_bg};">{area_location}</td>')
        html.append(f'<td style="{_h5_table_cell_style(False)}background:{row_bg};">{exclude_name}</td>')
        html.append(f'<td style="{_h5_table_cell_style(False)}background:{row_bg};">{pending_text}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


def _build_detail_section(match_notes: List[str]) -> str:
    if not match_notes:
        return ""

    items = "".join(
        '<div style="'
        'display:flex;'
        'align-items:flex-start;'
        'gap:6px;'
        'margin-top:4px;'
        'color:#4B5563;'
        'font-size:13px;'
        'line-height:1.7;'
        '">'
        '<span style="color:#416DF1;font-weight:700;">·</span>'
        f'<span style="flex:1;">{escape(note)}</span>'
        '</div>'
        for note in match_notes
    )

    return (
        '<div style="margin-top:12px;">'
        '<div style="font-size:14px;font-weight:600;color:#1F2937;">跟您说一下小细节：</div>'
        f'{items}'
        '</div>'
    )


def _build_field_brief(record: Dict[str, Any], display_fields: List[str]) -> str:
    fields = [field for field in display_fields if field in _TANK_AREA_FIELDS and field != "tankDetailList"]
    if not fields:
        fields = ["areaName", "areaLocation", "excludeAreasName"]
    parts = []
    for field in fields[:3]:
        parts.append(f"{_short_label(field)}是{_display_text(record.get(field), 45)}")
    return "、".join(parts)


def _build_result_summary(
    records: List[Dict[str, Any]],
    plan: Dict[str, Any],
    candidate_mode: bool = False,
    match_notes: Optional[List[str]] = None,
) -> str:
    match_notes = match_notes or []
    if not records:
        return "没有找到符合条件的相关信息哦~"

    query_keywords = _filters_to_display_keys(
        plan.get("filters") or [],
        plan.get("display_fields") or [],
        candidate_mode=candidate_mode,
    )
    shown = min(len(records), _CHAT_DISPLAY_LIMIT)

    if candidate_mode:
        title = (
            "<div style='margin-top:10px;'>没有找到完全一致的储罐区域信息。"
            "不过根据您的描述，系统匹配到以下可能要查询的信息：</div>"
        )
    elif len(records) == 1:
        # 单条结果不输出“它的区域名称是...、区域位置是...”这类长句，
        # 只保留提示语 + H5 卡片，避免坐标过长影响可读性。
        title = "<div style='margin-top:10px;'>好的，已为您查询到储罐区域如下：</div>"
    else:
        title = f"<div style='margin-top:10px;'>好的，已为您查询到符合条件的以下 {len(records)} 个储罐区域：</div>"
        if len(records) > shown:
            title += f"<div style='margin-top:4px;color:#6B7280;'>本次结果较多，右侧仅展示前 {shown} 条。</div>"

    # 单个储罐区域表格不显示“序号”；多个储罐区域/候选结果显示“序号”。
    show_index = candidate_mode or len(records) != 1
    table_html = _build_query_table_html(records[:shown], query_keywords, show_index=show_index)
    detail_html = _build_detail_section(match_notes)
    if not candidate_mode and len(records) == 1 and _clean_text(records[0].get("id")):
        footer = "<div style='margin-top:8px;'>左侧已同步打开该储罐区域详情页面，您可以直接查看哦～</div>"
    else:
        footer = "<div style='margin-top:8px;'>左侧已同步打开本次查询结果的储罐区域列表，您可以直接查看哦～</div>"
    return title + table_html + detail_html + footer


class StorageTankAreaSelectSkill(BaseSkill):
    md_path = "skills/storage_tank_area_select_skill/SKILL.md"

    @property
    def execution_mode(self) -> str:
        """查询类 skill 走直通：先生成查询计划，再由 run() 真实查接口。"""
        return "direct"

    @property
    def reply_template(self) -> str:
        """
        查询结果由本 skill 的 _build_result_summary() 直接生成 H5/Markdown 内容。

        这里故意返回空字符串，避免 main.py 的 _handle_media_direct 再把
        SKILL.md 里的 reply_template 注入大模型，导致二次生成：
        - “它的区域名称是...、区域位置是...”
        - 普通 Markdown 表格
        - 模板占位符内容
        """
        return ""

    def jump_and_clear_session(self, session_id, question_id, jump_route):
        if not session_id:
            return
        ws_data = _gen_clear_session(session_id=session_id)
        logger.info(f"[StorageTankAreaSelectSkill] clear_session ws_data={ws_data}")
        jump_data = gen_jump_data(session_id=session_id, question_id=question_id, route=jump_route)
        self.set_cached_jump_data(session_id=session_id, data=jump_data, ttl=60)
        self.set_cached_data(session_id=session_id, data=ws_data, ttl=60)

    def jump_and_push_data(
        self,
        session_id,
        question_id,
        filtered_records,
        jump_route,
        update_url,
        cacheName,
        detail_id: Optional[str] = None,
    ):
        if not session_id:
            return

        if detail_id:
            # gen_jump_data 当前只内置 listTimer/addStatusTimer，这里手动补 query.id，
            # 前端实际地址形如：/#/system/tankAreaDetail?id=2059477819503775746
            jump_data = gen_jump_data(
                session_id=session_id,
                question_id=question_id,
                route=_DETAIL_JUMP_ROUTE,
            )
            jump_data.setdefault("data", {}).setdefault("query", {})["id"] = str(detail_id)
        else:
            # 多条结果/候选结果跳转列表页，仿照维保计划查询 Skill：
            # route=/system/tankAreaManage，并携带不会重复的 listTimer，
            # 让前端能稳定刷新列表缓存。
            jump_data = gen_jump_data(
                session_id=session_id,
                question_id=question_id,
                route=jump_route,
                listTimer=_next_list_timer(),
            )

        jump_payload = jump_data.get("data", {}) if isinstance(jump_data, dict) else {}
        real_route = jump_payload.get("route")
        real_query = jump_payload.get("query") or {}
        preview_url = f"/#/{real_route.lstrip('/')}" if real_route else ""
        if real_query:
            preview_url += "?" + "&".join(f"{k}={v}" for k, v in real_query.items())

        route_type = "详情页" if real_route == _DETAIL_JUMP_ROUTE else ("列表页" if real_route == _LIST_JUMP_ROUTE else "未知页面")
        jump_mode = "single_detail" if detail_id else "multi_or_candidate_list"
        logger.info(
            f"[StorageTankAreaSelectSkill] 即将跳转页面：mode={jump_mode}, "
            f"route={real_route}, route_type={route_type}, "
            f"query={real_query}, preview_url={preview_url}"
        )
        logger.info(
            f"[StorageTankAreaSelectSkill] jump_data={jump_data}, "
            f"url={update_url}, cacheName={cacheName}, "
            f"cacheDataSize={len(filtered_records) if isinstance(filtered_records, list) else filtered_records}, "
            f"detail_id={detail_id}"
        )
        self.set_cached_jump_data(session_id=session_id, data=jump_data)

        # 多条结果/候选结果需要列表缓存；单条跳详情时也同步当前命中记录，
        # 这样前端如果需要从缓存读取当前记录，也能拿到。
        ws_data = gen_ws_list_data(
            url=update_url,
            cacheName=cacheName,
            cacheData=filtered_records,
            question_id=question_id,
            session_id=session_id,
        )
        logger.info(f"[StorageTankAreaSelectSkill] ws_data={ws_data}")
        self.set_cached_data(session_id=session_id, data=ws_data)

    async def check_slots(
        self,
        question: str,
        collected: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        session_id=None,
        question_id=None,
    ) -> Tuple[Any, Dict[str, Any], str]:
        collected = dict(collected or {})
        context = context or {}
        logger.info(f"[StorageTankAreaSelectSkill] check_slots context={context}")
        logger.info(f"[StorageTankAreaSelectSkill] check_slots collected={collected}")
        return await self._collect_slots(question, collected, context, session_id, question_id)

    async def _collect_slots(self, question, collected, context, session_id=None, question_id=None):
        latest_question = (question or "").split("用户补充：")[-1].strip()

        if latest_question in {"取消", "退出", "不用查了", "不查了"}:
            self.jump_and_clear_session(
                session_id=session_id,
                question_id=question_id,
                jump_route=_JUMP_ROUTE,
            )
            return "cancel", {}, "好的，已退出储罐区域查询状态。"

        plan = await asyncio.to_thread(_build_query_plan, question, context)
        filters = plan.get("filters") or []
        query_mode = plan.get("query_mode")

        if not hasattr(self, "_query_plan_store"):
            self._query_plan_store = {}
        self._query_plan_store[session_id] = plan

        for flt in filters:
            field = flt.get("field")
            value = flt.get("value") or flt.get("start") or ""
            if field and field != "__any__" and value != "":
                collected[field] = value

        if not filters and query_mode != "all_explicit":
            return (
                False,
                collected,
                "好的！请问要查询什么信息？可以按区域名称、排斥区域、创建人、创建时间、更新时间或储罐编号查询。"
                "如果您要查全部，请明确说“查询全部储罐区域”。",
            )

        logger.info(f"[StorageTankAreaSelectSkill] 已生成查询计划：{plan}")
        return True, collected, "槽位已齐全，请立即调用 storage_tank_area_select_skill 执行真实查询，不要直接回答。"

    def run(self, session_id=None, question_id=None, **kwargs) -> Dict[str, Any]:
        logger.info("[StorageTankAreaSelectSkill] run 已执行")
        logger.info(f"[StorageTankAreaSelectSkill] run kwargs={kwargs}")

        plan = getattr(self, "_query_plan_store", {}).pop(session_id, None)
        if not plan:
            fallback_filters = []
            raw_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ""}
            for raw_field, value in raw_kwargs.items():
                field = _field_name(raw_field)
                if field not in _TANK_AREA_FIELDS:
                    continue
                op = "contains" if field in _TEXT_MATCH_FIELDS else "eq"
                clean = _normalize_filter_item({"field": field, "op": op, "value": value})
                if clean:
                    fallback_filters.append(clean)

            plan = {
                "filters": fallback_filters,
                "query_mode": "targeted" if fallback_filters else "all_explicit",
                "display_fields": [],
                "reply_hint": "",
            }
            logger.warning(f"[StorageTankAreaSelectSkill] 未取到实例查询计划，使用 kwargs 兼容计划：{plan}")

        try:
            result = _execute_query_plan(plan)
        except Exception as e:
            logger.exception(f"[StorageTankAreaSelectSkill] 查询异常：{e}")
            self.jump_and_push_data(
                session_id=session_id,
                question_id=question_id,
                filtered_records=[],
                jump_route=_JUMP_ROUTE,
                update_url=_URL,
                cacheName=_CACHE_NAME,
            )
            return SkillResult.fail(
                "储罐区域查询暂时没成功，麻烦您稍后再试一下哦。如果还是一直失败的话，"
                "就联系下系统管理员，说明一下操作时间就可以帮忙排查啦。"
            )

        records = result.get("records") or []
        candidate_mode = bool(result.get("candidate_mode"))
        detail_id = None
        if not candidate_mode and len(records) == 1:
            detail_id = _clean_text(records[0].get("id")) or None

        self.jump_and_push_data(
            session_id=session_id,
            question_id=question_id,
            filtered_records=records,
            jump_route=_JUMP_ROUTE,
            update_url=_URL,
            cacheName=_CACHE_NAME,
            detail_id=detail_id,
        )

        summary = _build_result_summary(
            records=records,
            plan=plan,
            candidate_mode=candidate_mode,
            match_notes=result.get("match_notes") or [],
        )
        logger.info(f"[StorageTankAreaSelectSkill] 最终直出 summary 预览={summary[:300]!r}")

        # 关键：这里不要返回 SkillResult.ok(...)
        # 原因：main.py 的 _handle_media_direct 在 status == "success" 时会再次调用大模型，
        # 并把 SKILL.md 的 reply_template 注入 prompt，导致输出又变成：
        # “它的区域名称是...、区域位置是...” 和 Markdown 表格。
        #
        # 这里使用自定义状态 direct_summary，让 main.py 走 else 分支：
        # full_answer = skill_result.get("summary", ...)
        # 这样可以把本 skill 生成的 H5 summary 原样推给前端。
        # 注意：jump_and_push_data() 已经在上面执行，main.py 后续仍会 pop cached_jump_data/cached_data，
        # 所以左侧详情页/列表页跳转不会受影响。
        logger.info(
            "[StorageTankAreaSelectSkill] 返回 direct_summary，强制右侧对话直出 summary，避免大模型二次改写。"
        )
        return {
            "status": "direct_summary",
            "data": {},
            "summary": summary,
        }


storage_tank_area_select_skill = StorageTankAreaSelectSkill()
