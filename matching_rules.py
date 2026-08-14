"""通用方向规则与各产品类别的规则配置。

扩展新产品时，只需在 PRODUCT_RULES 中增加配置，不需要修改相似度算法。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


AT_LEAST = "at_least"  # 替料 >= 原料
AT_MOST = "at_most"    # 替料 <= 原料
CLOSE = "close"        # 替料与原料的相对偏差不超过 tolerance
EXACT = "exact"        # 数值相等（允许浮点计算误差）
SOFT_DIRECTION_INFLUENCE = 0.15


@dataclass(frozen=True)
class ParameterRule:
    name: str
    patterns: tuple[str, ...]
    mode: str
    critical: bool = True
    tolerance: float = 0.0
    condition_key: str | None = None

    def matches(self, field_key: str) -> bool:
        return any(re.search(pattern, field_key) for pattern in self.patterns)


# 第一阶段只启用 MOSFET。Qg、Trr、Qrr 当前不在 MCC CSV 中；未来增加列后会自动匹配。
MOSFET_RULES = (
    ParameterRule("VDS", (r"^drainsourcevoltagevdsv?$",), AT_LEAST),
    ParameterRule("VGS(th)", (r"^gatethresholdvoltagevgsth(?:min|max)v?$",), CLOSE, tolerance=0.20),
    ParameterRule("RDS(on)", (r"rdson",), AT_MOST),
    ParameterRule("Qg", (r"(?:total)?gatechargeqg", r"^qg"), AT_MOST),
    ParameterRule("Trr", (r"^trr", r"reverserecoverytime"), AT_MOST),
    ParameterRule("Qrr", (r"^qrr", r"reverserecoverycharge"), AT_MOST),
    ParameterRule("ID", (r"^draincurrentida?$",), AT_LEAST),
)


SWITCHING_DIODE_RULES = (
    ParameterRule("Trr", (r"^trr", r"reverserecoverytime"), AT_MOST),
    ParameterRule("VF", (r"^vf.*max",), AT_MOST, critical=False),
    ParameterRule("IR", (r"^ir.*max", r"reverseleakage"), AT_MOST, critical=False),
    ParameterRule("IF", (r"^ifava?$",), AT_LEAST, critical=False),
    ParameterRule("IFSM", (r"^ifsma?$",), AT_LEAST, critical=False),
    ParameterRule("VR", (r"^vrwm",), AT_LEAST, critical=False),
)


SBD_RULES = (
    ParameterRule("VF", (r"^vf.*max",), AT_MOST),
    ParameterRule("IR", (r"^ir.*max", r"reverseleakage"), AT_MOST),
    ParameterRule("IF", (r"^ifava?$",), AT_LEAST, critical=False),
    ParameterRule("IFSM", (r"^ifsma?$",), AT_LEAST, critical=False),
    ParameterRule("VR", (r"^vrwm",), AT_LEAST, critical=False),
)


FRED_RULES = (
    ParameterRule("Trr", (r"^trr", r"reverserecoverytime"), AT_MOST),
    ParameterRule("VF", (r"^vf.*max",), AT_MOST, critical=False),
    ParameterRule("IR", (r"^ir.*max", r"reverseleakage"), AT_MOST, critical=False),
    ParameterRule("IF", (r"^ifava?$",), AT_LEAST, critical=False),
    ParameterRule("IFSM", (r"^ifsma?$",), AT_LEAST, critical=False),
    ParameterRule("VR", (r"^vrwm",), AT_LEAST, critical=False),
)


TRANSISTOR_RULES = (
    ParameterRule("fT", (r"^ft",), AT_LEAST),
    ParameterRule("HFE range", (r"^hfe(?:min|max)",), CLOSE, tolerance=0.20),
    ParameterRule("VCE(sat)", (r"^vcesat",), AT_MOST, critical=False),
    ParameterRule("VCE", (r"^vceo",), AT_LEAST, critical=False),
    ParameterRule("IC", (r"^ica?$",), AT_LEAST, critical=False),
    ParameterRule("PC", (r"^p[cd](?:w|mw)?$",), AT_LEAST, critical=False),
)


ZENER_RULES = (
    ParameterRule("VZ", (r"^vznom",), EXACT),
    ParameterRule("IZT", (r"^izt",), CLOSE, tolerance=0.20),
    ParameterRule("PD", (r"^pd(?:w|mw)?$",), AT_LEAST, critical=False),
)


TVS_RULES = (
    ParameterRule("VRWM", (r"^reversestandoffvoltagevrwm", r"^vrwm"), AT_MOST),
    ParameterRule("VC", (r"clampingvoltagevc",), AT_MOST),
    ParameterRule("IPP", (r"peakpulsecurrentipp",), AT_LEAST),
    ParameterRule("CJ", (r"junctioncapacitancecj",), AT_MOST),
    ParameterRule("IR", (r"^ir", r"reverseleakage"), AT_MOST, critical=False),
    ParameterRule("VBR", (r"breakdownvoltage.*vbr",), AT_MOST, critical=False),
)


ESD_RULES = (
    ParameterRule("VRWM", (r"^reversestandoffvoltagevrwm", r"^vrwm"), AT_MOST),
    ParameterRule("VC", (r"clampingvoltagevc",), AT_MOST),
    ParameterRule("IPP", (r"peakpulsecurrentipp",), AT_LEAST),
    ParameterRule("CJ", (r"junctioncapacitancecj",), AT_MOST),
    ParameterRule("VBR", (r"breakdownvoltage.*vbr",), AT_MOST, critical=False),
)


PRODUCT_RULES: dict[str, tuple[ParameterRule, ...]] = {
    "mosfet": MOSFET_RULES,
    "switching_diode": SWITCHING_DIODE_RULES,
    "sbd": SBD_RULES,
    "fred": FRED_RULES,
    "transistor": TRANSISTOR_RULES,
    "zener": ZENER_RULES,
    "tvs": TVS_RULES,
    "esd": ESD_RULES,
}


def get_rule(product_family: str, field_key: str) -> ParameterRule | None:
    for rule in PRODUCT_RULES.get(product_family, ()):
        if rule.matches(field_key):
            return rule
    return None


def passes_rule(rule: ParameterRule, query_value: float, candidate_value: float) -> bool:
    numerical_tolerance = 1e-9 * max(1.0, abs(query_value), abs(candidate_value))
    if rule.mode == AT_LEAST:
        return candidate_value + numerical_tolerance >= query_value
    if rule.mode == AT_MOST:
        return candidate_value <= query_value + numerical_tolerance
    if rule.mode == CLOSE:
        denominator = max(abs(query_value), 1e-12)
        return abs(candidate_value - query_value) / denominator <= rule.tolerance + 1e-12
    if rule.mode == EXACT:
        return math.isclose(candidate_value, query_value, rel_tol=1e-9, abs_tol=1e-12)
    raise ValueError(f"未知方向规则：{rule.mode}")


def preference_score(rule: ParameterRule, query_value: float, candidate_value: float) -> float:
    """软方向得分：期望方向为 1，反方向按相对偏差平滑下降。"""
    if rule.mode == AT_MOST and candidate_value <= query_value:
        return 1.0
    if rule.mode == AT_LEAST and candidate_value >= query_value:
        return 1.0
    if rule.mode == EXACT:
        return 1.0 if passes_rule(rule, query_value, candidate_value) else 0.0
    largest = max(abs(query_value), abs(candidate_value))
    if largest <= 1e-15 or query_value * candidate_value < 0:
        return 0.0
    return max(0.0, 1.0 - abs(candidate_value - query_value) / largest)


def requirement_text(rule: ParameterRule, query_value: float) -> str:
    if rule.mode == AT_LEAST:
        return f">={query_value:g}"
    if rule.mode == AT_MOST:
        return f"<={query_value:g}"
    if rule.mode == CLOSE:
        return f"接近 {query_value:g}（±{rule.tolerance:.0%}）"
    if rule.mode == EXACT:
        return f"={query_value:g}"
    return str(query_value)
