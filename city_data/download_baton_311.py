#!/usr/bin/env python3
"""Download a reproducible, stratified sample of Baton Rouge 311 records.

The source layers are queried for category counts and IDs first. Only the
selected IDs are then downloaded, so the full 1.2M-record service is never
materialized locally.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_SERVICE_URL = (
    "https://services.arcgis.com/KYvXadMcgf0K1EzK/arcgis/rest/services/"
    "311_Citizen_Request_for_Service___All_Requests/FeatureServer"
)
OUTPUT_DIR = Path(__file__).resolve().parent
TARGET_TOTAL = 24_000
SAMPLE_SEED = "civicflow-baton-311-20260829-v1"
REQUEST_TIMEOUT_SECONDS = 120

LAYER_NAMES = {
    0: "Open or In-Progress Status",
    1: "Closed Status",
}

LAYER_FIELDS = {
    0: [
        "id",
        "ParentType",
        "typename",
        "StreetAddress",
        "cityname",
        "createdate",
        "Lastaction",
        "StatusDesc",
        "comments",
        "Department",
        "Division",
        "DeptDiv",
        "DeptDIVID",
    ],
    1: [
        "id",
        "ParentType",
        "typename",
        "StreetAddress",
        "cityname",
        "createdate",
        "Lastaction",
        "ClosedDate",
        "StatusDesc",
        "comments",
        "Department",
        "Division",
        "DeptDiv",
        "DeptDIVID",
    ],
}

CATEGORY_ZH = {
    "GARBAGE": "垃圾处理",
    "RECYCLING": "资源回收",
    "SEWER/WASTEWATER": "污水与下水道",
    "DRAINAGE, EROSION, FLOODING OR HOLES": "排水、积水与洪涝",
    "ROAD MAINTENANCE ISSUES": "道路维护",
    "STREET/TRAFFIC ISSUES": "街道与交通",
    "MOWING AND TREE ISSUES": "树木和杂草维护",
    "BLIGHTED PROPERTIES": "环境卫生与失管物业",
    "BUILDING CODE/ZONING VIOLATIONS": "建筑规范与分区违规",
    "ENVIRONMENTAL ISSUES": "环境问题",
    "NEIGHBORHOOD/SUBDIVISION ISSUES": "社区与小区问题",
}

CATEGORY_KEYWORDS_ZH = {
    "GARBAGE": ["垃圾", "垃圾收集", "垃圾清运"],
    "RECYCLING": ["回收", "资源回收", "可回收物"],
    "SEWER/WASTEWATER": ["污水", "下水道", "排污", "堵塞", "污水回流"],
    "DRAINAGE, EROSION, FLOODING OR HOLES": ["排水", "积水", "内涝", "洪涝", "排水沟"],
    "ROAD MAINTENANCE ISSUES": ["道路", "路面", "坑洞", "道路维护"],
    "STREET/TRAFFIC ISSUES": ["街道", "交通", "信号灯", "标志"],
    "MOWING AND TREE ISSUES": ["树木", "杂草", "修剪", "绿化"],
    "BLIGHTED PROPERTIES": ["失管物业", "环境卫生", "废弃房屋"],
    "BUILDING CODE/ZONING VIOLATIONS": ["建筑规范", "分区", "违规建筑"],
    "ENVIRONMENTAL ISSUES": ["环境", "污染", "异味"],
    "NEIGHBORHOOD/SUBDIVISION ISSUES": ["社区", "小区", "邻里"],
}


def arcgis_query(layer: int, params: dict[str, Any]) -> dict[str, Any]:
    """Run an ArcGIS REST query with a small retry policy."""

    url = f"{BASE_SERVICE_URL}/{layer}/query"
    encoded = urlencode({key: str(value) for key, value in params.items()}).encode("utf-8")
    request = Request(
        url,
        data=encoded,
        headers={"User-Agent": "CivicFlow-city-data/1.0"},
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.load(response)
            if "error" in payload:
                raise RuntimeError(json.dumps(payload["error"], ensure_ascii=False))
            return payload
        except (HTTPError, URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt == 2:
                break

    raise RuntimeError(f"ArcGIS query failed for layer {layer}: {last_error}")


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def allocate_quotas(total: int, strata: list[dict[str, Any]]) -> dict[tuple[int, str | None], int]:
    """Allocate an exact total with the largest-remainder method."""

    source_total = sum(int(item["count"]) for item in strata)
    if source_total == 0:
        raise RuntimeError("The source returned no records.")
    if total > source_total:
        raise RuntimeError(f"Requested {total} records but source has only {source_total}.")

    quotas: dict[tuple[int, str | None], int] = {}
    fractions: list[tuple[float, tuple[int, str | None]]] = []
    assigned = 0
    for item in strata:
        key = (int(item["layer"]), item["category"])
        exact = total * int(item["count"]) / source_total
        base = min(int(exact), int(item["count"]))
        quotas[key] = base
        fractions.append((exact - base, key))
        assigned += base

    for _, key in sorted(fractions, key=lambda pair: (-pair[0], pair[1]))[: total - assigned]:
        quotas[key] += 1
    return quotas


def stable_order(value: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}:{value}".encode("utf-8")).hexdigest()


def get_strata() -> list[dict[str, Any]]:
    strata: list[dict[str, Any]] = []
    for layer in LAYER_NAMES:
        payload = arcgis_query(
            layer,
            {
                "f": "json",
                "where": "1=1",
                "outStatistics": json.dumps(
                    [
                        {
                            "statisticType": "count",
                            "onStatisticField": "id",
                            "outStatisticFieldName": "total",
                        }
                    ]
                ),
                "groupByFieldsForStatistics": "ParentType",
                "orderByFields": "total DESC",
                "returnGeometry": "false",
            },
        )
        for feature in payload.get("features", []):
            attributes = feature.get("attributes", {})
            strata.append(
                {
                    "layer": layer,
                    "category": attributes.get("ParentType"),
                    "count": int(attributes.get("total") or 0),
                }
            )
    return [item for item in strata if item["count"] > 0]


def get_ids(layer: int, category: str | None) -> list[int]:
    if category is None:
        where = "ParentType IS NULL"
    else:
        escaped_category = category.replace("'", "''")
        where = f"ParentType = '{escaped_category}'"

    payload = arcgis_query(
        layer,
        {
            "f": "json",
            "where": where,
            "returnIdsOnly": "true",
            "returnGeometry": "false",
        },
    )
    return [int(value) for value in (payload.get("objectIds") or [])]


def fetch_selected_records(layer: int, ids: list[int]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for id_chunk in chunks(sorted(ids), 2_000):
        payload = arcgis_query(
            layer,
            {
                "f": "json",
                "objectIds": ",".join(str(value) for value in id_chunk),
                "outFields": ",".join(LAYER_FIELDS[layer]),
                "returnGeometry": "false",
            },
        )
        for feature in payload.get("features", []):
            attributes = feature.get("attributes", {})
            source_id = attributes.get("id")
            if source_id is not None:
                records[int(source_id)] = attributes
    missing = sorted(set(ids) - set(records))
    if missing:
        raise RuntimeError(f"Layer {layer}: {len(missing)} selected records were not returned.")
    return records


def iso_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
    return str(value)


def status_zh(status: Any, layer: int) -> str:
    text = str(status or "").strip().upper()
    if layer == 1 or "CLOSED" in text:
        return "已关闭"
    if "PROGRESS" in text:
        return "处理中"
    if "OPEN" in text or "NEW" in text:
        return "待处理"
    return "状态待确认"


def build_raw_record(layer: int, attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_layer": layer,
        "source_layer_name": LAYER_NAMES[layer],
        "source_id": int(attributes["id"]),
        "attributes": attributes,
    }


def build_normalized_record(layer: int, attributes: dict[str, Any], split: str) -> dict[str, Any]:
    parent_type = attributes.get("ParentType")
    category_zh = CATEGORY_ZH.get(parent_type, "其他城市公共服务")
    keywords = CATEGORY_KEYWORDS_ZH.get(parent_type, ["城市公共服务"])
    source_id = int(attributes["id"])
    type_name = str(attributes.get("typename") or "").strip() or None
    comments = str(attributes.get("comments") or "").strip() or None
    department = str(attributes.get("Department") or "").strip() or None
    division = str(attributes.get("Division") or "").strip() or None

    raw_text_parts = [
        str(parent_type or ""),
        type_name or "",
        comments or "",
        department or "",
        division or "",
    ]
    raw_text_en = " ".join(part for part in raw_text_parts if part).strip()
    retrieval_text_zh = (
        f"事项类别：{category_zh}；关键词：{'、'.join(keywords)}；"
        f"事项状态：{status_zh(attributes.get('StatusDesc'), layer)}"
    )

    return {
        "case_id": f"br311-{layer}-{source_id}",
        "source_layer": layer,
        "source_layer_name": LAYER_NAMES[layer],
        "source_id": source_id,
        "parent_type": parent_type,
        "type_name": type_name,
        "street_address": attributes.get("StreetAddress"),
        "city_name": attributes.get("cityname"),
        "created_at": iso_date(attributes.get("createdate")),
        "last_action_at": iso_date(attributes.get("Lastaction")),
        "closed_at": attributes.get("ClosedDate"),
        "status_desc": attributes.get("StatusDesc"),
        "comments": comments,
        "department": department,
        "division": division,
        "dept_div": attributes.get("DeptDiv"),
        "dept_div_id": attributes.get("DeptDIVID"),
        "category_zh": category_zh,
        "status_zh": status_zh(attributes.get("StatusDesc"), layer),
        "keywords_zh": keywords,
        "retrieval_text_zh": retrieval_text_zh,
        "raw_text_en": raw_text_en,
        "dataset_split": split,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        return path
    columns = list(rows[0].keys())
    output_path = path
    try:
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                csv_row["keywords_zh"] = "、".join(csv_row["keywords_zh"])
                writer.writerow(csv_row)
    except PermissionError:
        output_path = path.with_name(f"{path.stem}_updated{path.suffix}")
        print(f"      CSV 原文件被占用，改写入: {output_path.name}")
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                csv_row["keywords_zh"] = "、".join(csv_row["keywords_zh"])
                writer.writerow(csv_row)
    return output_path


def main() -> int:
    print("[1/5] 查询图层和事项类别统计...")
    strata = get_strata()
    source_total = sum(int(item["count"]) for item in strata)
    quotas = allocate_quotas(TARGET_TOTAL, strata)

    print(f"      数据源记录数: {source_total:,}")
    print(f"      目标抽样数: {TARGET_TOTAL:,}")

    selected_ids: dict[int, dict[str | None, list[int]]] = defaultdict(dict)
    selected_by_stratum: dict[tuple[int, str | None], list[int]] = {}
    print("[2/5] 获取各分层 ID 并执行稳定抽样...")
    for item in strata:
        layer = int(item["layer"])
        category = item["category"]
        quota = quotas[(layer, category)]
        ids = get_ids(layer, category)
        if len(ids) != int(item["count"]):
            print(
                f"      警告: 图层 {layer} / {category} 统计数与 ID 数不一致 "
                f"({item['count']} vs {len(ids)})"
            )
        ordered_ids = sorted(ids, key=lambda value: stable_order(f"{layer}:{value}"))
        selected = ordered_ids[:quota]
        selected_by_stratum[(layer, category)] = selected
        selected_ids[layer][category] = selected
        print(f"      图层 {layer} | {category or '未分类'} | {len(selected):,} 条")

    print("[3/5] 按 80/10/10 生成检索、验证和测试划分...")
    split_by_id: dict[tuple[int, int], str] = {}
    for (layer, category), ids in selected_by_stratum.items():
        ordered = sorted(ids, key=lambda value: stable_order(f"split:{layer}:{value}"))
        retrieval_end = int(len(ordered) * 0.8)
        validation_end = int(len(ordered) * 0.9)
        for index, source_id in enumerate(ordered):
            split = "retrieval" if index < retrieval_end else "validation" if index < validation_end else "test"
            split_by_id[(layer, source_id)] = split

    print("[4/5] 下载选中的属性字段...")
    raw_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for layer in LAYER_NAMES:
        layer_ids = [source_id for (selected_layer, source_id) in split_by_id if selected_layer == layer]
        attributes_by_id = fetch_selected_records(layer, layer_ids)
        for source_id in layer_ids:
            attributes = attributes_by_id[source_id]
            raw_rows.append(build_raw_record(layer, attributes))
            normalized_rows.append(
                build_normalized_record(layer, attributes, split_by_id[(layer, source_id)])
            )

    raw_rows.sort(key=lambda row: (row["source_layer"], row["source_id"]))
    normalized_rows.sort(key=lambda row: row["case_id"])
    if len(raw_rows) != TARGET_TOTAL or len(normalized_rows) != TARGET_TOTAL:
        raise RuntimeError(f"Expected {TARGET_TOTAL} records, got {len(raw_rows)}.")
    if len({row["case_id"] for row in normalized_rows}) != TARGET_TOTAL:
        raise RuntimeError("Duplicate case_id values detected.")

    print("[5/5] 写入 city_data 文件...")
    write_jsonl(OUTPUT_DIR / "cases_raw.jsonl", raw_rows)
    write_jsonl(OUTPUT_DIR / "cases_normalized.jsonl", normalized_rows)
    csv_path = write_csv(OUTPUT_DIR / "cases_normalized.csv", normalized_rows)

    split_counts = Counter(row["dataset_split"] for row in normalized_rows)
    category_counts = Counter(row["category_zh"] for row in normalized_rows)
    layer_counts = Counter(str(row["source_layer"]) for row in normalized_rows)
    manifest = {
        "source": {
            "service_url": BASE_SERVICE_URL,
            "layer_names": {str(key): value for key, value in LAYER_NAMES.items()},
            "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "sampling": {
            "seed": SAMPLE_SEED,
            "method": "按图层和 ParentType 分层，分层内使用 SHA-256 稳定抽样",
            "source_record_count": source_total,
            "target_record_count": TARGET_TOTAL,
            "sample_ratio": TARGET_TOTAL / source_total,
            "split_ratio": "retrieval 80%, validation 10%, test 10%（各分层取整）",
        },
        "output": {
            "raw_jsonl": "cases_raw.jsonl",
            "normalized_jsonl": "cases_normalized.jsonl",
            "normalized_csv": csv_path.name,
            "fields_selected": LAYER_FIELDS,
            "geometry_downloaded": False,
        },
        "counts": {
            "by_split": dict(sorted(split_counts.items())),
            "by_layer": dict(sorted(layer_counts.items())),
            "by_category_zh": dict(sorted(category_counts.items())),
        },
    }
    (OUTPUT_DIR / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("下载完成:")
    print(f"  原始记录: {OUTPUT_DIR / 'cases_raw.jsonl'}")
    print(f"  标准化记录: {OUTPUT_DIR / 'cases_normalized.jsonl'}")
    print(f"  CSV: {csv_path}")
    print(f"  检索/验证/测试: {dict(sorted(split_counts.items()))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        raise SystemExit(130)
