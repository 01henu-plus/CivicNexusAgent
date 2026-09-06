from __future__ import annotations

import re

from civicnexus.agents.base import Agent
from civicnexus.agents.decision import DecisionEngine
from civicnexus.domain.context import CompiledContext
from civicnexus.domain.enums import TaskStatus
from civicnexus.domain.results import AgentResult

INTAKE_LLM_PROMPT = """
你负责 CivicNexus 的自然语言意图识别和信息抽取。
请将识别结果写入 patch.extracted，至少考虑 category、location、urgency，必要时补充
subcategory 或 intent。category 优先使用以下标准分类：垃圾处理、资源回收、污水与下水道、
排水、积水与洪涝、道路维护、街道与交通、树木和杂草维护、环境卫生与失管物业、建筑规范与分区违规、
环境问题、社区与小区问题。像“漏水”“水管破裂”“冒水”等表达已经说明了问题类型，应归入合适的排水或
污水类分类，不要再追问事项类型。
location 可以先保留用户自然表达的相对地点，例如“我家附近”“我家这条路”“附近这条路”“小区门口”，
但相对地点不能作为受理完成条件。此时只追问具体道路、小区、楼栋或门牌号，不要重复追问事项类型。
结合 recent_messages 和 active_facts 理解多轮补充；例如上一轮已经说明漏水，本轮只说“我家附近”，应合并
为同一个漏水事项并只追问实际地址。
只有信息真正缺失时才填写 missing_fields；category 和 location 都足够时将 next_status 设为 CASE_ANALYSIS，
但 location 是相对地点时仍视为缺少具体地址。否则设为 WAITING_FOR_USER，并在 message_zh 中只追问真正缺失的字段。
不要创建工具请求。
""".strip()


class IntakeAgent(Agent):
    name = "intake"

    def __init__(self, decisions: DecisionEngine) -> None:
        self.decisions = decisions

    def run(self, context: CompiledContext) -> AgentResult:
        facts = dict(context.active_facts)
        message = context.latest_message
        category = _category(message) or facts.get("category")
        location = _location(message) or facts.get("location")
        urgency = (
            "紧急"
            if any(word in message for word in ("紧急", "严重", "危险", "冒水", "影响通行"))
            else facts.get("urgency", "普通")
        )
        patch = {
            "extracted": {**facts, "category": category, "location": location, "urgency": urgency}
        }
        missing = _missing_fields(category, location)
        local = AgentResult(
            agent_name=self.name,
            patch=patch,
            missing_fields=missing,
            next_status=TaskStatus.WAITING_FOR_USER if missing else TaskStatus.CASE_ANALYSIS,
            confidence=0.95 if category else 0.78,
            message_zh=f"请补充{_missing_zh(missing)}。"
            if missing
            else "信息已整理，正在查询相关历史案例。",
            reason="deterministic slot extraction",
        )
        result = self.decisions.decide(
            agent_name=self.name,
            context=context,
            local=local,
            prompt=f"{INTAKE_LLM_PROMPT}\n{context.skill.get('prompt', '')}",
        )

        # A model may describe a relative location as complete.  Location
        # precision is a business gate, so normalize it after fusion and keep
        # the follow-up focused on the missing address only.
        extracted = result.patch.get("extracted")
        extracted = dict(extracted) if isinstance(extracted, dict) else {}
        if location and _is_relative_location(location):
            model_location = extracted.get("location")
            if not model_location or not _is_relative_location(str(model_location)):
                extracted["location"] = location
        result.patch["extracted"] = extracted
        result.missing_fields = _missing_fields(
            extracted.get("category"), extracted.get("location")
        )
        result.next_status = (
            TaskStatus.WAITING_FOR_USER if result.missing_fields else TaskStatus.CASE_ANALYSIS
        )
        if result.missing_fields:
            result.message_zh = f"请补充{_missing_zh(result.missing_fields)}。"
        return result


def _category(message: str) -> str | None:
    groups = {
        "垃圾处理": ("垃圾", "清运", "垃圾收集"),
        "资源回收": ("回收", "可回收物", "资源回收"),
        "污水与下水道": (
            "污水",
            "下水道",
            "排污",
            "污水回流",
            "漏水",
            "冒水",
            "水管",
            "管道破裂",
        ),
        "排水、积水与洪涝": ("排水", "积水", "内涝", "洪涝", "排水沟"),
        "道路维护": ("道路", "路面", "坑洞", "道路维护"),
        "街道与交通": ("街道", "交通", "信号灯", "标志", "路灯"),
        "树木和杂草维护": ("树木", "杂草", "修剪", "绿化"),
        "环境卫生与失管物业": ("失管物业", "废弃房屋", "环境卫生"),
        "建筑规范与分区违规": ("建筑规范", "分区", "违规建筑"),
        "环境问题": ("污染", "异味", "环境问题"),
        "社区与小区问题": ("社区", "小区", "邻里"),
    }
    return next(
        (name for name, words in groups.items() if any(word in message for word in words)), None
    )


def _location(message: str) -> str | None:
    suffix = (
        r"(?:"
        r"(?:路|街|道|巷)(?:\s*\d+\s*号)?(?:附近|门口|门前|路口)?"
        r"|(?:小区|社区)(?:门口|门前)?"
        r"|号|附近|楼下|门口|门前|路口"
        r")"
    )
    explicit = re.search(
        rf"(?:地址(?:是|为)?|位于|在)[:：\s]*([^，。；!?！？\n]{{2,30}}?{suffix})",
        message,
    )
    if explicit:
        return explicit.group(1).strip()
    match = re.search(
        rf"([^，。；!?！？\n]{{2,30}}?{suffix})",
        message,
    )
    return match.group(1).strip() if match else None


def _missing_fields(category: object, location: object) -> list[str]:
    missing: list[str] = []
    if not category:
        missing.append("category")
    if not location or _is_relative_location(str(location)):
        missing.append("location")
    return missing


def _is_relative_location(location: str) -> bool:
    normalized = re.sub(r"\s+", "", location)
    return any(
        marker in normalized
        for marker in (
            "附近",
            "这条路",
            "这附近",
            "这里",
            "那边",
            "楼下",
            "家门口",
            "门口",
            "门前",
        )
    )


def _missing_zh(fields: list[str]) -> str:
    labels = {
        "category": "事项类型",
        "location": "具体地址（道路、小区、楼栋或门牌号）",
    }
    return "和".join(labels[item] for item in fields)
