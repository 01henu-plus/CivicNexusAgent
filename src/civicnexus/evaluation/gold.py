"""The intentionally small, versioned gold set for the interview demo.
The messages are short Chinese paraphrases and do not contain the normalized
category labels.  Keeping this set in source control makes every reported
number reproducible without downloading the full city dataset or calling an
external model.
"""

from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field

GOLD_DATASET = "civicnexus-gold-v1"


class GoldCase(BaseModel):
    model_config = ConfigDict(frozen=True)
    case_id: str = Field(min_length=1)
    message: str = Field(min_length=2)
    category: str = Field(min_length=1)
    location: str = Field(min_length=1)
    skill: str = Field(min_length=1)
    department: str = Field(min_length=1)


# Six examples are enough to exercise the three declarative skills while
# keeping a candidate's walkthrough understandable.  Do not change labels
# without bumping GOLD_DATASET: the version is part of an evaluation report.
GOLD_CASES: tuple[GoldCase, ...] = (
    GoldCase(
        case_id="gold-001",
        message="地址：人民路1号，污水井堵塞并且路面积水。",
        category="污水与下水道",
        location="人民路1号",
        skill="drainage",
        department="排水部门",
    ),
    GoldCase(
        case_id="gold-002",
        message="地址：幸福小区，垃圾连续三天没人清运。",
        category="垃圾处理",
        location="幸福小区",
        skill="garbage",
        department="环卫部门",
    ),
    GoldCase(
        case_id="gold-003",
        message="地址：和平路，路面有坑洞影响通行。",
        category="道路维护",
        location="和平路",
        skill="road",
        department="道路交通部门",
    ),
    GoldCase(
        case_id="gold-004",
        message="地址：江滨街18号，下水道冒水很危险。",
        category="污水与下水道",
        location="江滨街18号",
        skill="drainage",
        department="排水部门",
    ),
    GoldCase(
        case_id="gold-005",
        message="地址：城北小区，垃圾回收点已经装满。",
        category="垃圾处理",
        location="城北小区",
        skill="garbage",
        department="环卫部门",
    ),
    GoldCase(
        case_id="gold-006",
        message="地址：建设路，路灯旁道路破损需要修复。",
        category="道路维护",
        location="建设路",
        skill="road",
        department="道路交通部门",
    ),
)
__all__ = ["GOLD_CASES", "GOLD_DATASET", "GoldCase"]
