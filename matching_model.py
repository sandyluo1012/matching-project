"""芯片替代料匹配引擎：缺失感知相似度、覆盖率和关键参数规则。"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matching_rules import (
    SOFT_DIRECTION_INFLUENCE,
    get_rule,
    passes_rule,
    preference_score,
    requirement_text,
)


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
PLUS_MINUS_DEFAULT_RATIO = 0.90
NUMBER_TOKEN = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
# MCC 当前只有 ESD IEC 数值会在单元格中重复写入 kV；其他单位已经在列名中。
# 使用白名单而不是任意字母后缀，避免把 ``24abc`` 静默当成 24。
OPTIONAL_UNIT = r"(?:\s*(?i:kv))?"
PLUS_MINUS_COMPONENT = re.compile(
    rf"\s*±\s*{NUMBER_TOKEN}{OPTIONAL_UNIT}\s*"
)
EXPLICIT_SIGNED_RANGE = re.compile(
    rf"\s*\+\s*{NUMBER_TOKEN}{OPTIONAL_UNIT}\s*/"
    rf"\s*[-−]\s*{NUMBER_TOKEN}{OPTIONAL_UNIT}\s*"
)
NUMERIC_COMPONENT = re.compile(
    rf"\s*(?:±|[+\-−])?\s*(?P<number>{NUMBER_TOKEN}){OPTIONAL_UNIT}\s*"
)
BOUNDED_NUMERIC_COMPONENT = re.compile(
    rf"\s*(?:<=|>=|≤|≥|<|>)\s*(?:[+\-−])?\s*"
    rf"(?P<number>{NUMBER_TOKEN}){OPTIONAL_UNIT}\s*"
)
OUTER_FORMAT_CHARACTERS = {"\u200b", "\ufeff"}

# Only these device selections give a slash-separated value component meaning.
# Keeping this explicit prevents unrelated slash values (for example ESD
# Air/Contact ratings) from being interpreted as two devices.
PAIR_PROFILES: dict[str, tuple[str, str, tuple[str, str], str]] = {
    "mosfet": ("Channel", "n+p", ("N", "P"), "N/P"),
    "transistor": ("Polarity", "npn+pnp", ("NPN", "PNP"), "NPN/PNP"),
    "prebiased": ("Polarity", "npn+pnp", ("NPN", "PNP"), "NPN/PNP"),
}

def clean_column(name: str) -> str:
    """合并 CSV 表头中的换行和不间断空格。"""
    return " ".join(str(name).replace("\xa0", " ").split())


def _strip_outer_spacing(value: Any) -> str:
    """移除首尾 Unicode 空白及常见不可见格式字符，绝不拼接内部数字。"""
    text = unicodedata.normalize("NFKC", str(value))
    start = 0
    end = len(text)
    while start < end and (
        text[start].isspace() or text[start] in OUTER_FORMAT_CHARACTERS
    ):
        start += 1
    while end > start and (
        text[end - 1].isspace() or text[end - 1] in OUTER_FORMAT_CHARACTERS
    ):
        end -= 1
    return text[start:end]


def _is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    text = _strip_outer_spacing(value)
    if text.casefold() in MISSING_MARKERS:
        return True
    # GUI 中的裸 ± 只是输入提示；N+P 双框会形成 ±/±，同样不能算作参数。
    return bool(text) and all(_strip_outer_spacing(part) == "±" for part in text.split("/"))


def parse_numeric_values(value: Any) -> tuple[float, ...]:
    """严格解析一个或多个规格值；任一分量含未知字符时拒绝整个值。"""
    if _is_missing(value):
        return ()
    text = _strip_outer_spacing(value)
    if "," in text:
        return ()
    parts = text.split("/")
    if len(parts) not in {1, 2}:
        return ()
    values: list[float] = []
    for part in parts:
        match = NUMERIC_COMPONENT.fullmatch(part)
        if not match:
            return ()
        values.append(abs(float(match.group("number"))))
    return tuple(values)


def is_bounded_numeric_value(value: Any) -> bool:
    """识别 `<1.2`、`>=5` 等有界规格；它们不能冒充精确数值。"""
    if _is_missing(value):
        return False
    parts = _strip_outer_spacing(value).split("/")
    return bool(parts) and all(
        BOUNDED_NUMERIC_COMPONENT.fullmatch(part) is not None
        for part in parts
    )


def _numbers(value: Any) -> tuple[float, ...]:
    """内部兼容别名；公开的数据清理流程复用 parse_numeric_values。"""
    return parse_numeric_values(value)


def _number(value: Any) -> float:
    """提取单值规格；多分量字段仅用于数值列识别，匹配时由 _numbers 成对处理。"""
    values = _numbers(value)
    return values[0] if values else np.nan


def _is_plus_minus_value(value: Any) -> bool:
    if _is_missing(value):
        return False
    return all(
        PLUS_MINUS_COMPONENT.fullmatch(part)
        for part in _strip_outer_spacing(value).split("/")
    )


def _is_explicit_signed_range(value: Any) -> bool:
    """识别 +正向额定值/-负向额定值，例如 +20/-16。"""
    if _is_missing(value):
        return False
    return EXPLICIT_SIGNED_RANGE.fullmatch(_strip_outer_spacing(value)) is not None


def _field_key(name: str) -> str:
    text = unicodedata.normalize("NFKC", clean_column(name)).casefold()
    text = text.replace("μ", "u").replace("µ", "u").replace("ω", "ohm").replace("Ω", "ohm")
    return re.sub(r"[^a-z0-9]+", "", text)


def _category_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _pair_profile(family: str) -> tuple[str, str, tuple[str, str], str] | None:
    return PAIR_PROFILES.get(family)


def _is_pair_selection(family: str, values: Any) -> bool:
    profile = _pair_profile(family)
    if profile is None:
        return False
    selector, expected, _, _ = profile
    value = values.get(selector) if hasattr(values, "get") else None
    # Treat optional spaces around '+' as cosmetic differences.
    return _category_value(value).replace(" ", "") == expected


def discover_catalogs(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    catalogs: dict[str, Path] = {}
    for path in sorted(data_dir.glob("*.csv")):
        match = re.search(r"MCC_DataExport_(.*?)\(MCC-", path.name)
        name = match.group(1) if match else path.stem
        catalogs[name.replace("-", " ").title()] = path
    return catalogs


def load_catalog(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    df = pd.read_csv(source)
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
    family = _family(list(df.columns), str(df.attrs.get("source_name", "")))
    return {
        column for column in numeric
        if (rule := get_rule(family, _field_key(column))) is not None and rule.critical
    }


def preference_features(df: pd.DataFrame) -> set[str]:
    """返回当前产品类别中配置为非关键方向偏好的字段。"""
    numeric, _ = infer_features(df)
    family = _family(list(df.columns), str(df.attrs.get("source_name", "")))
    return {
        column for column in numeric
        if (rule := get_rule(family, _field_key(column))) is not None and not rule.critical
    }


def plus_minus_features(df: pd.DataFrame) -> set[str]:
    """返回适合默认 ± 的字段：主流为 ±，例外仅允许明确的 +正/-负范围。"""
    numeric, _ = infer_features(df)
    result: set[str] = set()
    for column in numeric:
        values = [
            str(value).strip()
            for value in df[column]
            if not _is_missing(value)
        ]
        plus_minus_count = sum(_is_plus_minus_value(value) for value in values)
        if (
            values
            and plus_minus_count / len(values) >= PLUS_MINUS_DEFAULT_RATIO
            and all(
                _is_plus_minus_value(value) or _is_explicit_signed_range(value)
                for value in values
            )
        ):
            result.add(column)
    return result


def paired_features(df: pd.DataFrame) -> set[str]:
    """返回当前目录中存在器件成对斜杠值的数值字段。"""
    family = _family(list(df.columns), str(df.attrs.get("source_name", "")))
    profile = _pair_profile(family)
    if profile is None:
        return set()
    selector, expected, _, _ = profile
    if selector not in df.columns:
        return set()
    numeric, _ = infer_features(df)
    pair_rows = df[
        df[selector].map(lambda value: _category_value(value).replace(" ", ""))
        == expected
    ]
    if pair_rows.empty:
        return set()
    return {
        column
        for column in numeric
        if pair_rows[column].map(
            lambda value: len(_numbers(value)) == 2
        ).any()
    }


def is_shared_np_rating(column: str) -> bool:
    """VGS 的单个绝对额定值可同时用于 N、P 两侧；其他参数不能广播。"""
    return _field_key(column) == "gatesourcevoltagevgsv"


def _is_asymmetric_signed_rating(column: str, value: Any) -> bool:
    """识别单器件 VGS 的 +正向/-负向额定值，避免误判成 N/P 双通道。"""
    return is_shared_np_rating(column) and _is_explicit_signed_range(value)


def options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df:
        return []
    return sorted({str(value).strip() for value in df[column].dropna() if str(value).strip()})


def product_values(df: pd.DataFrame, product: str) -> dict[str, Any] | None:
    if "Product" not in df:
        return None
    rows = df[df["Product"].astype(str).str.casefold() == product.strip().casefold()]
    return None if rows.empty else rows.iloc[0].to_dict()


def _family(columns: list[str], source_name: str = "") -> str:
    source = source_name.casefold().replace("_", "-")
    if "switching-diodes" in source:
        return "switching_diode"
    if "schottky-barrier" in source or "small-signal-schottky" in source:
        return "sbd"
    if "fast-recovery-rectifiers" in source:
        return "fred"
    if "darlington" in source or "bipolar-transistors" in source:
        return "transistor"
    if "zener" in source:
        return "zener"
    if "tvs" in source:
        return "tvs"
    if "esd-protection" in source:
        return "esd"
    if "mosfet" in source:
        return "mosfet"

    keys = {_field_key(column) for column in columns}
    if any("vznom" in key for key in keys):
        return "zener"
    if any("drainsourcevoltagevds" in key for key in keys):
        return "mosfet"
    if any("vceo" in key for key in keys):
        return "transistor"
    if any(key.startswith("r1typ") for key in keys):
        return "prebiased"
    if any("junctioncapacitancecj" in key for key in keys):
        return "esd"
    if any("peakpulsepowerdissipationpppm" in key for key in keys):
        return "tvs"
    if any("maximumvoltagegatetoline" in key for key in keys):
        return "thyristor"
    return "generic"


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


def _numeric_values(value: Any, pair_mode: bool) -> tuple[float, ...]:
    if pair_mode:
        return _numbers(value)
    number = _number(value)
    return (float(number),) if np.isfinite(number) else ()


def _component_similarity(query_values: tuple[float, ...], candidate_values: tuple[float, ...]) -> float:
    return sum(
        _similarity(query_value, candidate_value)
        for query_value, candidate_value in zip(query_values, candidate_values)
    ) / len(query_values)


def _align_numeric_values(
    query_values: tuple[float, ...],
    candidate_values: tuple[float, ...],
    allow_scalar_broadcast: bool,
    required_components: int | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if required_components is not None and (
        len(query_values) != required_components
        or len(candidate_values) != required_components
    ):
        return None
    if len(query_values) == len(candidate_values):
        return query_values, candidate_values
    if allow_scalar_broadcast and {len(query_values), len(candidate_values)} == {1, 2}:
        if len(query_values) == 1:
            query_values = query_values * 2
        if len(candidate_values) == 1:
            candidate_values = candidate_values * 2
        return query_values, candidate_values
    return None


def _component_preference(
    rule: Any,
    query_values: tuple[float, ...],
    candidate_values: tuple[float, ...],
) -> float:
    return sum(
        preference_score(rule, query_value, candidate_value)
        for query_value, candidate_value in zip(query_values, candidate_values)
    ) / len(query_values)


def _rule_failure_text(
    column: str,
    rule: Any,
    query_values: tuple[float, ...],
    candidate_values: tuple[float, ...],
    component_labels: tuple[str, ...] | None = None,
) -> str:
    labels = component_labels or tuple(str(index + 1) for index in range(len(query_values)))
    failures = [
        f"{label}={candidate_value:g}，要求{requirement_text(rule, query_value)}"
        for label, query_value, candidate_value in zip(labels, query_values, candidate_values)
        if not passes_rule(rule, query_value, candidate_value)
    ]
    if len(query_values) == 1:
        return f"{column}={candidate_values[0]:g}，要求{requirement_text(rule, query_values[0])}"
    return f"{column}：" + "；".join(failures)


def recommend(
    inventory: pd.DataFrame,
    query: dict[str, Any],
    top_k: int = 10,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """硬规则过滤后，只用 query/candidate 共同已知的参数计算相似度。"""
    numeric, categorical = infer_features(inventory)
    source_name = str(inventory.attrs.get("source_name", "")).casefold()
    family = _family(list(inventory.columns), source_name)
    work = inventory.copy()

    package = _category_value(query.get("Package Type"))
    if not package:
        raise ValueError("封装类型为必填项。")
    candidates = work[work["Package Type"].map(_category_value) == package].copy()

    # 拓扑、功能数、极性和沟道在用户填写时都是硬约束。
    for column in categorical:
        query_value = _category_value(query.get(column))
        if query_value:
            candidates = candidates[candidates[column].map(_category_value) == query_value]

    pair_profile = _pair_profile(family)
    pair_selector = pair_profile[0] if pair_profile else ""
    pair_component_labels = pair_profile[2] if pair_profile else ()
    pair_display_name = pair_profile[3] if pair_profile else "成对"
    query_has_pair = any(
        not _is_missing(query.get(column))
        and "/" in str(query.get(column))
        and not _is_asymmetric_signed_rating(column, query.get(column))
        for column in numeric
    )
    query_pair_mode = _is_pair_selection(family, query)
    if pair_profile and query_has_pair and not query_pair_mode:
        expected_selection = "N+P" if family == "mosfet" else "NPN+PNP"
        raise ValueError(
            f"检测到 {pair_display_name} 双值参数，请先将 "
            f"{pair_selector} 选择为 {expected_selection}。"
        )
    query_values = {
        column: _numeric_values(
            query.get(column),
            query_pair_mode or _is_asymmetric_signed_rating(column, query.get(column)),
        )
        for column in numeric
    }
    invalid_query_values = sorted(
        column
        for column in numeric
        if not _is_missing(query.get(column)) and not query_values[column]
    )
    if invalid_query_values:
        raise ValueError(
            "这些参数不是有效的完整数值格式，请检查内部空白、比较符或未知字符："
            + "、".join(invalid_query_values)
        )
    paired = paired_features(inventory) if query_pair_mode else set()
    invalid_pairs = sorted([
        column
        for column in paired
        if query_pair_mode
        and not _is_missing(query.get(column))
        and len(query_values[column]) != 2
        and not (
            family == "mosfet"
            and is_shared_np_rating(column)
            and len(query_values[column]) == 1
        )
    ])
    invalid_shared = sorted([
        column
        for column in numeric
        if query_pair_mode
        and column not in paired
        and not _is_missing(query.get(column))
        and len(query_values[column]) != 1
    ])
    validation_messages: list[str] = []
    if invalid_pairs:
        if family == "mosfet":
            validation_messages.append(
                "这些 N+P 参数必须按“ N值/P值 ”填写（VGS 也可填一个共享值）："
                + "、".join(invalid_pairs)
            )
        else:
            validation_messages.append(
                "这些 NPN+PNP 参数必须按“ NPN值/PNP值 ”填写："
                + "、".join(invalid_pairs)
            )
    if invalid_shared:
        validation_messages.append(
            f"这些参数是 {pair_display_name} 共用规格，请只填写一个数值："
            + "、".join(invalid_shared)
        )
    if validation_messages:
        raise ValueError("；".join(validation_messages))
    active = [column for column in numeric if query_values[column]]
    if not active:
        raise ValueError("请至少填写一个电气参数。")
    # 即使硬约束没有候选，也要先完成客户输入校验，避免非法数值被空结果掩盖。
    if candidates.empty:
        return pd.DataFrame()
    rules = {column: get_rule(family, _field_key(column)) for column in active}

    feature_weights = {
        column: float(
            (weights or {}).get(column, 2.0 if rules[column] is not None and rules[column].critical else 1.0)
        )
        for column in active
    }
    if any(value <= 0 for value in feature_weights.values()):
        raise ValueError("参数权重必须大于 0。")
    total_weight = sum(feature_weights.values())
    # A BJT pair can legitimately contain an incomplete catalog value.  Keep
    # such candidates available for an explicit coverage/pending diagnosis;
    # MIN_COVERAGE still prevents recommendations based on too little data.
    has_active_pair = any(column in paired for column in active)
    minimum_shared = (
        1
        if len(active) == 1
        or (query_pair_mode and family != "mosfet" and has_active_pair)
        else 2
    )
    result_rows: list[dict[str, Any]] = []

    for _, candidate in candidates.iterrows():
        candidate_pair_mode = query_pair_mode or _is_pair_selection(family, candidate)
        candidate_values = {
            column: _numeric_values(
                candidate.get(column),
                candidate_pair_mode
                or _is_asymmetric_signed_rating(column, candidate.get(column)),
            )
            for column in active
        }
        invalid_candidate_values = [
            column
            for column in active
            if not _is_missing(candidate.get(column)) and not candidate_values[column]
        ]
        aligned_values = {
            column: _align_numeric_values(
                query_values[column],
                candidate_values[column],
                allow_scalar_broadcast=(
                    family == "mosfet"
                    and is_shared_np_rating(column)
                ),
                required_components=(
                    2
                    if query_pair_mode
                    and column in paired
                    and not (family == "mosfet" and is_shared_np_rating(column))
                    else None
                ),
            )
            if candidate_values[column]
            else None
            for column in active
        }
        incompatible = [
            column
            for column in active
            if candidate_values[column]
            and aligned_values[column] is None
        ]
        missing = [column for column in active if not candidate_values[column]]
        shared = [
            column
            for column in active
            if aligned_values[column] is not None
        ]
        if len(shared) < minimum_shared:
            continue

        risks: list[str] = []
        pending_reasons = [
            (
                f"{column}格式无法解析"
                if column in invalid_candidate_values
                else f"{column}缺失"
            )
            for column in missing
            if rules[column] is not None and rules[column].critical
        ]
        pending_reasons.extend(
            f"{column}的 {pair_display_name} 双值不完整"
            for column in incompatible
            if rules[column] is not None and rules[column].critical
        )
        failed_reasons: list[str] = []
        for column in shared:
            rule = rules[column]
            if rule is None:
                continue
            aligned_query, aligned_candidate = aligned_values[column]
            condition_column = _find_condition_column(rule.condition_key, numeric) if rule.condition_key else None
            if rule.condition_key and not _condition_is_comparable(condition_column, query, candidate):
                pending_reasons.append(f"{column}测试条件未验证")
                continue
            if not all(
                passes_rule(rule, query_value, candidate_value)
                for query_value, candidate_value in zip(aligned_query, aligned_candidate)
            ):
                problem = _rule_failure_text(
                    column,
                    rule,
                    aligned_query,
                    aligned_candidate,
                    pair_component_labels if len(aligned_query) == 2 else None,
                )
                if rule.critical:
                    failed_reasons.append(problem)

        shared_weight = sum(feature_weights[column] for column in shared)
        coverage = shared_weight / total_weight
        if coverage + 1e-12 < MIN_COVERAGE:
            continue
        raw_similarity = sum(
            feature_weights[column]
            * _component_similarity(*aligned_values[column])
            for column in shared
        ) / shared_weight
        soft_columns = [
            column for column in shared
            if rules[column] is not None and not rules[column].critical
        ]
        if soft_columns:
            soft_weight = sum(feature_weights[column] for column in soft_columns)
            direction_preference = sum(
                feature_weights[column]
                * _component_preference(rules[column], *aligned_values[column])
                for column in soft_columns
            ) / soft_weight
        else:
            direction_preference = 1.0
        direction_factor = (
            1.0 - SOFT_DIRECTION_INFLUENCE
            + SOFT_DIRECTION_INFLUENCE * direction_preference
        )
        base_score = raw_similarity * direction_factor * (0.70 + 0.30 * coverage)

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
        if invalid_candidate_values:
            risks.append(
                "候选参数格式无法解析：" + "、".join(invalid_candidate_values)
            )
        if incompatible:
            risks.append(f"{pair_display_name} 成对参数不完整，未参与比较")

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
            "方向偏好得分": direction_preference * 100,
            "已比较参数": f"{len(shared)}/{len(active)}",
            "可信度": confidence,
            "关键参数检查": critical_status,
            "关键参数问题": "；".join(critical_issues) if critical_issues else "无",
            "缺失参数": "；".join([
                *(
                    f"{column}（格式无法解析）"
                    if column in invalid_candidate_values
                    else column
                    for column in missing
                ),
                *(f"{column}（{pair_display_name} 双值不完整）" for column in incompatible),
            ]) if missing or incompatible else "无",
            "风险提示": "；".join(dict.fromkeys(risks)) if risks else "无",
            "核对提示": "封装名称相同，Pinout/尺寸/原厂规格书仍需人工核对",
            "车规等级": "车规级" if "A" in str(candidate.get("Compliance", "")).upper().split() else "非车规级",
            "_规则等级": rule_rank,
        })
        result_rows.append(row)

    if not result_rows:
        return pd.DataFrame()
    results = pd.DataFrame(result_rows)

    leading = [
        "Product", "Manufacture", "综合得分", "已知参数匹配度", "参数覆盖率", "方向偏好得分", "已比较参数",
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
