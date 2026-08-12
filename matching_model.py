"""芯片替代料匹配引擎：缺失感知相似度、覆盖率和关键参数规则。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "products" / "MCC" / "clean"
HARD_CATEGORICAL_COLUMNS = (
    "Number of Functions", "Configuration", "Polarity", "Channel", "ESD Diodes",
)
NON_FEATURE_COLUMNS = {
    "Manufacture", "Product", "Status", "Compliance", "Package Type",
    *HARD_CATEGORICAL_COLUMNS,
}
MISSING_MARKERS = {"", "-", "—", "nan", "none", "n/a", "na", "null"}
MIN_COVERAGE = 0.50


@dataclass(frozen=True)
class CriticalRule:
    """关键参数验证规则。mode: min/max/band/required。"""

    mode: str
    tolerance: float = 0.0
    condition_key: str | None = None


def clean_column(name: str) -> str:
    """合并 CSV 表头中的换行和不间断空格。"""
    return " ".join(str(name).replace("\xa0", " ").split())


def _is_missing(value: Any) -> bool:
    return pd.isna(value) or str(value).strip().casefold() in MISSING_MARKERS


def _number(value: Any) -> float:
    """提取规格数值；含逗号的多值参数视为无效，不能静默拼成一个数字。"""
    if _is_missing(value):
        return np.nan
    text = str(value).strip()
    if "," in text:
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return abs(float(match.group())) if match else np.nan


def _field_key(name: str) -> str:
    text = unicodedata.normalize("NFKC", clean_column(name)).casefold()
    text = text.replace("μ", "u").replace("µ", "u").replace("ω", "ohm").replace("Ω", "ohm")
    return re.sub(r"[^a-z0-9]+", "", text)


def _category_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def discover_catalogs(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    catalogs: dict[str, Path] = {}
    for path in sorted(data_dir.glob("*.csv")):
        match = re.search(r"MCC_DataExport_(.*?)\(MCC-", path.name)
        name = match.group(1) if match else path.stem
        catalogs[name.replace("-", " ").title()] = path
    return catalogs


def load_catalog(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    df = pd.read_csv(source, header=0, skiprows=[0, 2])
    df.columns = [clean_column(column) for column in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    if "Product" in df:
        df = df[df["Product"].notna()].drop_duplicates(subset=["Product"], keep="first")
    df = df.reset_index(drop=True)
    df.attrs["source_name"] = source.name
    return df


def infer_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """识别参数列；缺失很多但已有值确实为数值的列仍会显示在 GUI。"""
    numeric: list[str] = []
    for column in df.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        present = ~df[column].map(_is_missing)
        if not present.any():
            continue
        if df.loc[present, column].map(_number).notna().mean() >= 0.80:
            numeric.append(column)
    categorical = [column for column in HARD_CATEGORICAL_COLUMNS if column in df.columns]
    return numeric, categorical


def critical_features(df: pd.DataFrame) -> set[str]:
    """返回当前产品类别中受关键方向/窗口规则约束的数值字段。"""
    numeric, _ = infer_features(df)
    family = _family(list(df.columns))
    return {column for column in numeric if _critical_rule(column, family) is not None}


def options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df:
        return []
    return sorted({str(value).strip() for value in df[column].dropna() if str(value).strip()})


def product_values(df: pd.DataFrame, product: str) -> dict[str, Any] | None:
    if "Product" not in df:
        return None
    rows = df[df["Product"].astype(str).str.casefold() == product.strip().casefold()]
    return None if rows.empty else rows.iloc[0].to_dict()


def _family(columns: list[str]) -> str:
    keys = {_field_key(column) for column in columns}
    if any("vznom" in key for key in keys):
        return "zener"
    if any("drainsourcevoltagevds" in key for key in keys):
        return "mosfet"
    if any("vceo" in key for key in keys):
        return "bjt"
    if any(key.startswith("r1typ") for key in keys):
        return "prebiased"
    if any("junctioncapacitancecj" in key for key in keys):
        return "esd"
    if any("peakpulsepowerdissipationpppm" in key for key in keys):
        return "tvs"
    if any("maximumvoltagegatetoline" in key for key in keys):
        return "thyristor"
    if any(key.startswith("ifav") for key in keys):
        return "rectifier"
    return "generic"


def _critical_rule(column: str, family: str) -> CriticalRule | None:
    key = _field_key(column)

    # 保护器件的工作/击穿窗口不能简单理解成越高越好。
    if "vznom" in key:
        return CriticalRule("band", tolerance=0.10)
    if key.startswith("r1typ") or key.startswith("r2typ"):
        return CriticalRule("band", tolerance=0.10)
    if "breakdownvoltage" in key and ("vbr" in key):
        return CriticalRule("band", tolerance=0.20)
    if "vrwm" in key:
        return CriticalRule("min")

    # 额定能力：候选值至少不能低于客户器件。
    minimum_patterns = (
        "ifav", "ifsm", "drainsourcevoltagevds", "gatesourcevoltagevgs",
        "draincurrentid", "vceo", "ioa", "iom", "vcc", "peakpulsecurrentipp",
        "peakpulsepowerdissipation", "peakplusepowerdissipation", "singlepulsedavalancheenergyeas",
        "pulseddraincurrentidm", "junctiontemperaturetj", "vesdiec",
    )
    if any(pattern in key for pattern in minimum_patterns) and "threshold" not in key:
        return CriticalRule("min")
    if key in {"ica", "pdw", "pdmw"} or key.startswith("pd"):
        return CriticalRule("min")
    if key.startswith("tj") and "max" in key:
        return CriticalRule("min")
    if family == "thyristor" and ("maximumvoltagegatetoline" in key or "peakpulsecurrentipp" in key):
        return CriticalRule("min")

    # 损耗、泄漏、钳位和恢复时间：候选上限不能更差。
    if "rdson" in key or "vcesat" in key or key.startswith("trr"):
        return CriticalRule("max")
    if key.startswith("vf") and "max" in key:
        return CriticalRule("max", condition_key="ifa")
    if (key.startswith("ir") or "reverseleakage" in key) and ("max" in key or family == "tvs"):
        compact_name = re.sub(r"\s+", "", clean_column(column).casefold())
        return CriticalRule("max", condition_key="vrv" if "@vr" in compact_name else None)
    if "clampingvoltagevc" in key:
        return CriticalRule("max", condition_key="peakpulsecurrentipp")
    if "junctioncapacitancecj" in key:
        return CriticalRule("max")
    if key.startswith("zzt"):
        return CriticalRule("max", condition_key="izt")
    if key.startswith("zzk"):
        return CriticalRule("max", condition_key="izk")
    if "hfemin" in key:
        return CriticalRule("min")

    # 以下是关键窗口值，但没有普适的单向“更好”关系。
    if family in {"esd", "tvs"} and "breakdownvoltage" in key:
        return CriticalRule("required")
    return None


def _find_condition_column(condition_key: str, numeric: list[str]) -> str | None:
    for column in numeric:
        key = _field_key(column)
        if condition_key == "ifa" and key == "ifa":
            return column
        if condition_key == "vrv" and key == "vrv":
            return column
        if condition_key == "izt" and key.startswith("izt"):
            return column
        if condition_key == "izk" and key.startswith("izk"):
            return column
        if condition_key == "peakpulsecurrentipp" and "peakpulsecurrentipp" in key:
            return column
    return None


def _similarity(query_value: float, candidate_value: float) -> float:
    """对正值规格使用对称相对相似度；2 倍差异得到 0.5。"""
    if np.isclose(query_value, candidate_value, rtol=1e-12, atol=1e-12):
        return 1.0
    largest = max(abs(query_value), abs(candidate_value))
    if largest <= 1e-15 or query_value * candidate_value < 0:
        return 0.0
    return max(0.0, 1.0 - abs(query_value - candidate_value) / largest)


def _passes_rule(rule: CriticalRule, query_value: float, candidate_value: float) -> bool:
    tolerance = 1e-9 * max(1.0, abs(query_value), abs(candidate_value))
    if rule.mode == "min":
        return candidate_value + tolerance >= query_value
    if rule.mode == "max":
        return candidate_value <= query_value + tolerance
    if rule.mode == "band":
        denominator = max(abs(query_value), 1e-12)
        return abs(candidate_value - query_value) / denominator <= rule.tolerance + 1e-12
    return True


def _condition_is_comparable(
    condition_column: str | None,
    query: dict[str, Any],
    candidate: pd.Series,
) -> bool:
    if not condition_column:
        return True
    query_condition = _number(query.get(condition_column))
    candidate_condition = _number(candidate.get(condition_column))
    if not np.isfinite(query_condition) or not np.isfinite(candidate_condition):
        return False
    denominator = max(abs(query_condition), abs(candidate_condition), 1e-12)
    return abs(query_condition - candidate_condition) / denominator <= 0.10


def _is_multi_function(value: Any) -> bool:
    normalized = _category_value(value)
    return bool(normalized) and normalized not in {"single", "1", "1.0"}


def recommend(
    inventory: pd.DataFrame,
    query: dict[str, Any],
    top_k: int = 10,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """硬规则过滤后，只用 query/candidate 共同已知的参数计算相似度。"""
    numeric, categorical = infer_features(inventory)
    family = _family(list(inventory.columns))
    source_name = str(inventory.attrs.get("source_name", "")).casefold()
    work = inventory.copy()
    for column in numeric:
        work[column] = work[column].map(_number)

    package = _category_value(query.get("Package Type"))
    if not package:
        raise ValueError("封装类型为必填项。")
    candidates = work[work["Package Type"].map(_category_value) == package].copy()

    # 拓扑、功能数、极性和沟道在用户填写时都是硬约束。
    for column in categorical:
        query_value = _category_value(query.get(column))
        if query_value:
            candidates = candidates[candidates[column].map(_category_value) == query_value]
    if candidates.empty:
        return pd.DataFrame()

    active = [column for column in numeric if np.isfinite(_number(query.get(column)))]
    if not active:
        raise ValueError("请至少填写一个电气参数。")

    rules = {column: _critical_rule(column, family) for column in active}
    feature_weights = {
        column: float((weights or {}).get(column, 2.0 if rules[column] else 1.0))
        for column in active
    }
    if any(value <= 0 for value in feature_weights.values()):
        raise ValueError("参数权重必须大于 0。")
    total_weight = sum(feature_weights.values())
    minimum_shared = 1 if len(active) == 1 else 2
    result_rows: list[dict[str, Any]] = []

    for _, candidate in candidates.iterrows():
        shared = [column for column in active if np.isfinite(candidate[column])]
        missing = [column for column in active if column not in shared]
        if len(shared) < minimum_shared:
            continue

        risks: list[str] = []
        pending_reasons = [f"{column}缺失" for column in missing if rules[column] is not None]
        failed_reasons: list[str] = []
        for column in shared:
            rule = rules[column]
            if rule is None or rule.mode == "required":
                continue
            condition_column = _find_condition_column(rule.condition_key, numeric) if rule.condition_key else None
            if rule.condition_key and not _condition_is_comparable(condition_column, query, candidate):
                pending_reasons.append(f"{column}测试条件未验证")
                continue
            if not _passes_rule(rule, _number(query[column]), float(candidate[column])):
                symbol = {"min": ">=", "max": "<=", "band": f"±{rule.tolerance:.0%}"}.get(rule.mode, "")
                failed_reasons.append(
                    f"{column}={float(candidate[column]):g}，要求{symbol}{_number(query[column]):g}"
                )

        shared_weight = sum(feature_weights[column] for column in shared)
        coverage = shared_weight / total_weight
        if coverage + 1e-12 < MIN_COVERAGE:
            continue
        raw_similarity = sum(
            feature_weights[column] * _similarity(_number(query[column]), float(candidate[column]))
            for column in shared
        ) / shared_weight
        base_score = raw_similarity * (0.70 + 0.30 * coverage)

        topology_pending = False
        if _is_multi_function(query.get("Number of Functions")):
            if "Configuration" not in inventory.columns:
                pending_reasons.append("多功能器件拓扑未验证")
                topology_pending = True
            elif not _category_value(query.get("Configuration")):
                pending_reasons.append("客户未提供拓扑配置")
                topology_pending = True
            elif not _category_value(candidate.get("Configuration")):
                pending_reasons.append("候选拓扑缺失")
                topology_pending = True

        if failed_reasons:
            critical_status = "不满足"
            rule_rank = 0
            rule_factor = 0.60
        elif pending_reasons or topology_pending:
            critical_status = "待人工确认"
            rule_rank = 1
            rule_factor = 0.85
        else:
            critical_status = "通过"
            rule_rank = 2
            rule_factor = 1.0
        final_score = base_score * rule_factor

        status = _category_value(candidate.get("Status"))
        if status and status != "active":
            risks.append(f"状态为 {candidate.get('Status')}")
        if family == "mosfet" and "wide soa" in source_name:
            risks.append("SOA 曲线未验证")

        if critical_status == "通过" and coverage >= 0.90 and not risks:
            confidence = "高"
        elif critical_status != "不满足" and coverage >= 0.70:
            confidence = "中"
        else:
            confidence = "低"

        critical_issues = failed_reasons or pending_reasons
        if failed_reasons:
            risks.append("存在不满足的关键参数")
        elif pending_reasons:
            risks.append("关键参数需要人工确认")

        row = candidate.to_dict()
        row.update({
            "综合得分": final_score * 100,
            "已知参数匹配度": raw_similarity * 100,
            "参数覆盖率": coverage * 100,
            "已比较参数": f"{len(shared)}/{len(active)}",
            "可信度": confidence,
            "关键参数检查": critical_status,
            "关键参数问题": "；".join(critical_issues) if critical_issues else "无",
            "缺失参数": "；".join(missing) if missing else "无",
            "风险提示": "；".join(dict.fromkeys(risks)) if risks else "无",
            "核对提示": "封装名称相同，Pinout/尺寸/原厂规格书仍需人工核对",
            "车规等级": "车规级" if "A" in str(candidate.get("Compliance", "")).upper().split() else "非车规级",
            "_规则等级": rule_rank,
        })
        result_rows.append(row)

    if not result_rows:
        return pd.DataFrame()
    results = pd.DataFrame(result_rows)
    source_product = str(query.get("Product", "")).strip().casefold()
    if source_product:
        results = results[results["Product"].astype(str).str.casefold() != source_product]

    leading = [
        "Product", "Manufacture", "综合得分", "已知参数匹配度", "参数覆盖率", "已比较参数",
        "可信度", "关键参数检查", "关键参数问题", "缺失参数", "风险提示", "核对提示", "车规等级",
        "Package Type", *categorical,
    ]
    columns = [column for column in leading if column in results]
    columns += [column for column in active if column not in columns]
    return (
        results.sort_values(["_规则等级", "综合得分", "参数覆盖率", "已知参数匹配度"], ascending=False)
        .head(top_k)[columns]
        .reset_index(drop=True)
    )
