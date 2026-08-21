"""芯片替代料智能匹配桌面 GUI。"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import pandas as pd

from deepseek_service import classify_part_category, lookup_part_parameters
from matching_model import (
    critical_features,
    discover_catalogs,
    infer_features,
    is_shared_np_rating,
    load_catalog,
    options,
    paired_features,
    parse_numeric_values,
    plus_minus_features,
    preference_features,
    product_values,
    recommend,
)


CATALOG_CHINESE_NAMES = {
    "zener diodes": "稳压二极管",
    "switching diodes": "开关二极管",
    "small signal schottky diodes": "小信号肖特基二极管",
    "bridge rectifiers": "整流桥",
    "standard recovery rectifiers": "标准恢复整流二极管",
    "fast recovery rectifiers": "快恢复整流二极管",
    "super fast recovery rectifiers": "超快速恢复整流器",
    "schottky barrier rectifiers": "肖特基二极管",
    "small signal mosfets": "小信号MOSFETs",
    "power mosfets": "功率MOSFETs",
    "small signal bipolar transistors": "小信号晶体管",
    "medium power bipolar transistors": "中功率晶体管",
    "pre biased transistors": "数字晶体管",
    "voltage regulators": "电压调节器",
    "tvs": "瞬态抑制二极管",
    "esd protection devices": "静电保护器件",
    "darlington transistors": "达林顿晶体管",
    "programmable thyristor surge suppressor": "可编程晶闸管浪涌抑制器",
    "wide soa mosfets": "宽安全工作区MOSFETs",
}

DEEPSEEK_MANUFACTURERS = (
    "DIODES",
    "Infineon",
    "Nexperia",
    "ON",
    "st",
    "TI",
    "VISHAY",
    "强茂",
    "瞬雷",
    "扬杰",
    "长电科技",
    "长晶",
    "贝岭",
    "杰华特",
    "小华",
    "NXP",
    "聚鼎",
    "力特",
    "LRC",
    "槟城",
)

COMPACT_RESULT_LEADING_COLUMNS = (
    "Product", "Manufacture", "综合得分", "已知参数匹配度", "参数覆盖率",
    "可信度", "关键参数检查", "车规等级", "Package Type",
    "Number of Functions", "Configuration", "Polarity", "Channel", "ESD Diodes",
)
COMPACT_RESULT_TRAILING_COLUMNS = ("关键参数问题", "缺失参数", "风险提示")


def _localized_catalogs(catalogs: dict[str, Path]) -> dict[str, Path]:
    """只改变 GUI 显示名称，保留英文名称对应的原始 CSV 路径。"""
    localized: dict[str, Path] = {}
    for english_name, path in catalogs.items():
        chinese_name = CATALOG_CHINESE_NAMES.get(english_name.casefold())
        display_name = f"{english_name}（{chinese_name}）" if chinese_name else english_name
        localized[display_name] = path
    return localized


def _catalog_category_name(display_name: str) -> str:
    """从“英文（中文）”显示名中取出供分类服务使用的稳定英文类别名。"""
    return display_name.split("（", 1)[0].strip()


def _catalog_display_name(category: Any, catalogs: Mapping[str, Path]) -> str | None:
    """把服务返回的类别严格映射到唯一的 GUI 下拉选项。"""
    key = _option_key(category)
    if not key:
        return None

    matches: list[str] = []
    for display_name in catalogs:
        english_name = _catalog_category_name(display_name)
        chinese_name = CATALOG_CHINESE_NAMES.get(english_name.casefold(), "")
        aliases = (display_name, english_name, chinese_name)
        if any(alias and _option_key(alias) == key for alias in aliases):
            matches.append(display_name)
    return matches[0] if len(matches) == 1 else None


def _is_esd_discharge_field(column: str) -> bool:
    """识别 ESD 的 IEC 空气/接触放电规格列。"""
    normalized = "".join(column.casefold().split())
    return "iec61000-4-2" in normalized and "air/contact" in normalized


def _option_key(value: Any) -> str:
    """用于把模型返回值安全映射回当前目录中的真实下拉选项。"""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(text.split())


def _existing_option(value: Any, candidates: list[str]) -> str | None:
    """仅接受能与现有选项唯一对应的分类值，并返回目录中的原始写法。"""
    key = _option_key(value)
    matches = [candidate for candidate in candidates if _option_key(candidate) == key]
    return matches[0] if key and len(matches) == 1 else None


def _usable_text(value: Any) -> str:
    """把目录/API 标量转成 GUI 文本，同时排除 None、NaN 和空白。"""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value).strip()


def _lookup_payload(result: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    """兼容服务返回 dataclass 或字典，并提取参数、不确定字段和提示。"""
    if isinstance(result, Mapping):
        raw_parameters = result.get("parameters", result)
        raw_uncertain = result.get("uncertain_fields", [])
        raw_notes = result.get("warnings", result.get("notes", []))
    else:
        raw_parameters = getattr(result, "parameters", {})
        raw_uncertain = getattr(result, "uncertain_fields", [])
        raw_notes = getattr(result, "warnings", getattr(result, "notes", []))

    parameters = dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {}
    uncertain = _string_list(raw_uncertain)
    notes = _string_list(raw_notes)
    return parameters, uncertain, notes


def _classification_payload(result: Any) -> tuple[str, str, str, list[str]]:
    """兼容分类服务的 dataclass/字典结果，并拒绝不支持或有歧义的类别。"""
    if isinstance(result, Mapping):
        category = result.get("category")
        returned_part = result.get("part_number")
        manufacturer = result.get("manufacturer")
        notes = result.get("notes", result.get("warnings", []))
        supported = result.get("supported", True)
        ambiguous = result.get("ambiguous", False)
    else:
        category = getattr(result, "category", None)
        returned_part = getattr(result, "part_number", None)
        manufacturer = getattr(result, "manufacturer", None)
        notes = getattr(result, "notes", getattr(result, "warnings", []))
        supported = getattr(result, "supported", True)
        ambiguous = getattr(result, "ambiguous", False)

    if supported is False or ambiguous is True:
        category = None
    return (
        _usable_text(category),
        _usable_text(returned_part),
        _usable_text(manufacturer),
        _string_list(notes),
    )


def _classification_is_ambiguous(result: Any) -> bool:
    """单独保留分类的歧义信号，便于 GUI 引导用户选择制造商。"""
    if isinstance(result, Mapping):
        return result.get("ambiguous") is True
    return getattr(result, "ambiguous", False) is True


def _result_identity(result: Any) -> tuple[str, str]:
    """提取服务结果中的料号和制造商，用于跨阶段一致性校验。"""
    if isinstance(result, Mapping):
        part_number = result.get("part_number")
        manufacturer = result.get("manufacturer")
    else:
        part_number = getattr(result, "part_number", None)
        manufacturer = getattr(result, "manufacturer", None)
    return _usable_text(part_number), _usable_text(manufacturer)


def _manufacturer_key(value: Any) -> str:
    """规范化制造商名称；仅合并明确等价的 MCC 名称。"""
    normalized = unicodedata.normalize("NFKC", _usable_text(value)).casefold()
    compact = "".join(char for char in normalized if char.isalnum())
    if compact in {
        "mcc",
        "microcommercialcomponents",
        "mccmicrocommercialcomponents",
    }:
        return "microcommercialcomponents"
    return compact


def _same_manufacturer(first: Any, second: Any) -> bool:
    first_key = _manufacturer_key(first)
    return bool(first_key and first_key == _manufacturer_key(second))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _field_summary(fields: list[str], limit: int = 14) -> str:
    unique = list(dict.fromkeys(field for field in fields if field))
    if not unique:
        return "无"
    shown = "、".join(unique[:limit])
    remaining = len(unique) - limit
    return f"{shown}（另有 {remaining} 项）" if remaining > 0 else shown


def build_result_view(
    results: pd.DataFrame,
    *,
    compact: bool = False,
    automotive_only: bool = False,
    critical_columns: set[str] | None = None,
) -> pd.DataFrame:
    """根据 GUI 视图选项筛选行和列，不修改模型生成的完整结果。"""
    view = results
    if automotive_only:
        if "车规等级" not in view.columns:
            view = view.iloc[0:0]
        else:
            view = view[view["车规等级"] == "车规级"]
    if not compact:
        return view.copy()

    ordered = [
        column for column in COMPACT_RESULT_LEADING_COLUMNS
        if column in view.columns
    ]
    ordered.extend(
        column for column in view.columns
        if column in (critical_columns or set()) and column not in ordered
    )
    ordered.extend(
        column for column in COMPACT_RESULT_TRAILING_COLUMNS
        if column in view.columns and column not in ordered
    )
    return view.loc[:, ordered].copy()


class MatchingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("芯片替代料智能匹配")
        self.geometry("1280x820")
        self.minsize(1040, 680)
        self.catalogs = _localized_catalogs(discover_catalogs())
        self.df = pd.DataFrame()
        self.inputs: dict[str, tk.StringVar] = {}
        self.input_defaults: dict[str, str] = {}
        self.numeric_input_columns: set[str] = set()
        self.categorical_input_columns: set[str] = set()
        self.paired_inputs: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        self.single_input_widgets: dict[str, tk.Misc] = {}
        self.pair_input_widgets: dict[str, tk.Misc] = {}
        self.pair_selector_column = ""
        self.pair_selector_target = ""
        self.pair_labels = ("N沟道", "P沟道")
        self.pair_first_single_values: set[str] = set()
        self.pair_second_single_values: set[str] = set()
        self.previous_pair_selector_value = ""
        self.results = pd.DataFrame()
        self.has_run_match = False
        self.critical_result_columns: set[str] = set()
        self.compact_results_var = tk.BooleanVar(value=False)
        self.automotive_only_var = tk.BooleanVar(value=False)
        self.deepseek_api_key_var = tk.StringVar(
            value=os.environ.get("DEEPSEEK_API_KEY", "")
        )
        self.show_api_key_var = tk.BooleanVar(value=False)
        self.manufacturer_var = tk.StringVar(value="")
        self._autofill_generation = 0
        self._autofill_busy = False
        self._style()
        self._layout()
        if self.catalogs:
            self.catalog_var.set(next(iter(self.catalogs)))
            self.load_selected_catalog()
        else:
            messagebox.showerror("数据缺失", "未找到可用的候选 CSV 数据。")

    def _style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"), foreground="#17324d")
        style.configure("Sub.TLabel", font=("Microsoft YaHei UI", 10), foreground="#5c6f82")
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=(18, 9))
        style.configure("Treeview", rowheight=30, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

    def _layout(self) -> None:
        root = ttk.Frame(self, padding=20)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="芯片替代料智能匹配", style="Title.TLabel").pack(anchor="w")
        ttk.Label(root, text="硬约束过滤 · 缺失感知相似度 · 参数覆盖率与风险提示", style="Sub.TLabel").pack(anchor="w", pady=(2, 16))

        bar = ttk.Frame(root)
        bar.pack(fill="x", pady=(0, 12))
        ttk.Label(bar, text="物料类型").pack(side="left")
        self.catalog_var = tk.StringVar()
        self.catalog_box = ttk.Combobox(bar, textvariable=self.catalog_var, values=list(self.catalogs), state="readonly", width=62)
        self.catalog_box.pack(side="left", padx=(8, 20))
        self.catalog_box.bind("<<ComboboxSelected>>", lambda _e: self.load_selected_catalog())
        ttk.Label(bar, text="推荐数量").pack(side="left")
        self.top_var = tk.IntVar(value=10)
        ttk.Spinbox(bar, from_=1, to=50, textvariable=self.top_var, width=6).pack(side="left", padx=8)
        self.count_label = ttk.Label(bar, text="", style="Sub.TLabel")
        self.count_label.pack(side="right")

        panes = ttk.Panedwindow(root, orient="horizontal")
        panes.pack(fill="both", expand=True)
        left = ttk.LabelFrame(panes, text=" 客户芯片参数 ", padding=12)
        self.customer_panel = left
        right = ttk.LabelFrame(panes, text=" 推荐物料 ", padding=12)
        panes.add(left, weight=2)
        panes.add(right, weight=5)

        customer = ttk.Frame(left)
        customer.pack(fill="x", pady=(0, 8))
        ttk.Label(customer, text="DeepSeek API Key（仅本次运行）").pack(anchor="w")
        api_key_row = ttk.Frame(customer)
        api_key_row.pack(fill="x", pady=(4, 0))
        self.api_key_entry = ttk.Entry(
            api_key_row,
            textvariable=self.deepseek_api_key_var,
            show="*",
        )
        self.api_key_entry.pack(side="left", fill="x", expand=True)
        self.api_key_entry.bind("<KeyRelease>", self._api_key_edited)
        ttk.Checkbutton(
            api_key_row,
            text="显示",
            variable=self.show_api_key_var,
            command=self._toggle_api_key_visibility,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(
            customer,
            text="本程序不将 Key 写入磁盘；仅在本次运行中使用并经 HTTPS 发送给 DeepSeek。",
            style="Sub.TLabel",
            wraplength=330,
        ).pack(anchor="w", pady=(3, 7))

        ttk.Label(customer, text="制造商（可选）").pack(anchor="w")
        self.manufacturer_box = ttk.Combobox(
            customer,
            textvariable=self.manufacturer_var,
            values=("", *DEEPSEEK_MANUFACTURERS),
            state="readonly",
        )
        self.manufacturer_box.pack(fill="x", pady=(4, 0))
        self.manufacturer_box.bind("<<ComboboxSelected>>", self._manufacturer_edited)
        ttk.Label(
            customer,
            text="通常不需要选择；只有 DeepSeek 发现同型号重名/有歧义时再选。",
            style="Sub.TLabel",
            wraplength=330,
        ).pack(anchor="w", pady=(3, 7))

        ttk.Label(customer, text="客户当前使用的型号").pack(anchor="w")
        self.product_var = tk.StringVar()
        product_row = ttk.Frame(customer)
        product_row.pack(fill="x", pady=(4, 0))
        self.product_entry = ttk.Entry(product_row, textvariable=self.product_var)
        self.product_entry.pack(side="left", fill="x", expand=True)
        self.product_entry.bind("<KeyRelease>", self._product_number_edited)
        self.autofill_button = ttk.Button(
            product_row,
            text="识别类型并自动填参",
            command=self.autofill_from_part_number,
        )
        self.autofill_button.pack(side="left", padx=(8, 0))
        ttk.Label(
            customer,
            text="输入完整料号后可自动识别物料类型并填入参数；\n结果必须对照原厂规格书核对！！！",
            style="Sub.TLabel",
            wraplength=330,
        ).pack(anchor="w", pady=(4, 0))

        self.parameter_canvas = tk.Canvas(left, highlightthickness=0, width=350)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.parameter_canvas.yview)
        self.form = ttk.Frame(self.parameter_canvas)
        self.form.bind(
            "<Configure>",
            lambda _e: self.parameter_canvas.configure(scrollregion=self.parameter_canvas.bbox("all")),
        )
        self.parameter_canvas.create_window((0, 0), window=self.form, anchor="nw")
        self.parameter_canvas.configure(yscrollcommand=scrollbar.set)
        self.parameter_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        action = ttk.Frame(right)
        action.pack(fill="x", pady=(0, 8))
        ttk.Button(action, text="开始匹配", style="Accent.TButton", command=self.run_match).pack(side="left")
        ttk.Button(action, text="清空输入", command=self.clear_inputs).pack(side="left", padx=(8, 0))
        ttk.Button(action, text="导出结果 CSV", command=self.export_results).pack(side="left", padx=8)
        ttk.Checkbutton(
            action,
            text="精简显示",
            variable=self.compact_results_var,
            command=self._refresh_results_view,
        ).pack(side="left", padx=(2, 8))
        ttk.Checkbutton(
            action,
            text="只看车规级",
            variable=self.automotive_only_var,
            command=self._refresh_results_view,
        ).pack(side="left")
        self.status_var = tk.StringVar(value="请填写参数后开始匹配")
        ttk.Label(action, textvariable=self.status_var, style="Sub.TLabel").pack(side="right")
        table_frame = ttk.Frame(right)
        table_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table_frame, show="headings")
        ybar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        xbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.tree.tag_configure("passed", background="#edf8ef", foreground="#205c2e")
        self.tree.tag_configure("pending", background="#fff7df", foreground="#765b00")
        self.tree.tag_configure("failed", background="#ffe9e9", foreground="#8a2020")
        self.tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self._bind_customer_mousewheel(self.customer_panel)

    def _toggle_api_key_visibility(self) -> None:
        """只切换输入框的本地显示方式，不复制、记录或持久化 Key。"""
        self.api_key_entry.configure(show="" if self.show_api_key_var.get() else "*")

    def _cancel_autofill_for_context_change(self, status: str) -> None:
        """查询条件改变后取消旧响应的应用资格。"""
        if not self._autofill_busy:
            return
        self._autofill_generation += 1
        self._autofill_busy = False
        self.autofill_button.state(["!disabled"])
        self.status_var.set(status)

    def _api_key_edited(self, _event: tk.Event) -> None:
        self._cancel_autofill_for_context_change(
            "API Key 已修改，旧的 DeepSeek 查询结果将被丢弃"
        )

    def _manufacturer_edited(self, _event: tk.Event) -> None:
        self._cancel_autofill_for_context_change(
            "制造商已修改，旧的 DeepSeek 查询结果将被丢弃"
        )

    def _bind_customer_mousewheel(self, widget: tk.Misc) -> None:
        """让客户参数区域内的所有现有控件都能驱动参数画布滚动。"""
        widget.bind("<MouseWheel>", self._scroll_customer_form)
        widget.bind("<Button-4>", self._scroll_customer_form)
        widget.bind("<Button-5>", self._scroll_customer_form)
        for child in widget.winfo_children():
            self._bind_customer_mousewheel(child)

    def _scroll_customer_form(self, event: tk.Event) -> str:
        """兼容 Windows/macOS 滚轮与 Linux Button-4/5 事件。"""
        delta = int(getattr(event, "delta", 0))
        if delta:
            direction = -1 if delta > 0 else 1
            steps = direction * max(1, abs(delta) // 120)
        else:
            button = int(getattr(event, "num", 0))
            if button not in {4, 5}:
                return "break"
            steps = -1 if button == 4 else 1
        self.parameter_canvas.yview_scroll(steps, "units")
        return "break"

    def load_selected_catalog(
        self,
        *,
        preserve_part_number: str | None = None,
        autofill_token: int | None = None,
    ) -> None:
        """载入所选 CSV；自动分类切换时可延续同一条受 token 保护的请求。"""
        continuing_autofill = bool(
            autofill_token is not None
            and autofill_token == self._autofill_generation
            and self._autofill_busy
        )
        if not continuing_autofill:
            # 手动切换类别会让后台旧请求失效，避免结果进入错误的产品表单。
            self._autofill_generation += 1
            self._autofill_busy = False
            self.autofill_button.state(["!disabled"])
        else:
            self.autofill_button.state(["disabled"])
        self.df = load_catalog(self.catalogs[self.catalog_var.get()])
        self.product_var.set(preserve_part_number if continuing_autofill else "")
        for widget in self.form.winfo_children():
            widget.destroy()
        self.inputs.clear()
        self.input_defaults.clear()
        self.paired_inputs.clear()
        self.single_input_widgets.clear()
        self.pair_input_widgets.clear()
        self.pair_selector_column = ""
        self.pair_selector_target = ""
        self.pair_labels = ("N沟道", "P沟道")
        self.pair_first_single_values.clear()
        self.pair_second_single_values.clear()
        self.previous_pair_selector_value = ""
        numeric, categorical = infer_features(self.df)
        self.numeric_input_columns = set(numeric)
        self.categorical_input_columns = {"Package Type", *categorical}
        critical = critical_features(self.df)
        self.critical_result_columns = critical
        preferred = preference_features(self.df)
        paired = paired_features(self.df)
        if paired and "Polarity" in self.df.columns and any(
            str(value).strip().casefold() == "npn+pnp"
            for value in self.df["Polarity"].dropna()
        ):
            self.pair_selector_column = "Polarity"
            self.pair_selector_target = "npn+pnp"
            self.pair_labels = ("NPN", "PNP")
            self.pair_first_single_values = {"npn", "npn*2"}
            self.pair_second_single_values = {"pnp", "pnp*2"}
        elif paired and "Channel" in self.df.columns:
            self.pair_selector_column = "Channel"
            self.pair_selector_target = "n+p"
            self.pair_labels = ("N沟道", "P沟道")
            self.pair_first_single_values = {"n", "n+n"}
            self.pair_second_single_values = {"p", "p+p"}
        plus_minus = plus_minus_features(self.df)
        fields = ["Package Type", *categorical, *numeric]
        for index, column in enumerate(fields):
            is_esd_discharge = _is_esd_discharge_field(column)
            if column == "Package Type":
                suffix = "  * 必填"
            elif column in categorical:
                suffix = "  [硬约束]"
            elif column in critical:
                suffix = "  [关键]"
            elif column in preferred:
                suffix = "  [方向偏好]"
            else:
                suffix = ""
            if column in paired:
                pair_name = "NPN+PNP" if self.pair_selector_column == "Polarity" else "N+P"
                suffix += f"  [{pair_name}双值]"
            if column in plus_minus:
                suffix += "  [默认±，可修改]"
            if is_esd_discharge:
                suffix += "  [可选择/可输入]"
            label = column + suffix
            ttk.Label(self.form, text=label).grid(row=index * 2, column=0, sticky="w", pady=(7, 2))
            var = tk.StringVar()
            default_value = "±" if column in plus_minus else ""
            var.set(default_value)
            self.inputs[column] = var
            self.input_defaults[column] = default_value
            values = options(self.df, column) if column == "Package Type" or column in categorical else []
            if is_esd_discharge:
                widget = ttk.Combobox(
                    self.form,
                    textvariable=var,
                    values=options(self.df, column),
                    state="normal",
                    width=39,
                )
            elif values:
                widget = ttk.Combobox(self.form, textvariable=var, values=values, state="readonly", width=39)
            else:
                widget = ttk.Entry(self.form, textvariable=var, width=42)
            widget.grid(row=index * 2 + 1, column=0, sticky="ew")
            if column in paired:
                pair_frame = ttk.Frame(self.form)
                pair_default = self.input_defaults[column]
                n_var = tk.StringVar(value=pair_default)
                p_var = tk.StringVar(value=pair_default)
                first_label = ttk.Label(pair_frame, text=self.pair_labels[0])
                first_label.grid(row=0, column=0, padx=(0, 4))
                ttk.Entry(pair_frame, textvariable=n_var, width=14).grid(row=0, column=1, sticky="ew")
                second_label = ttk.Label(pair_frame, text=self.pair_labels[1])
                second_label.grid(row=0, column=2, padx=(10, 4))
                ttk.Entry(pair_frame, textvariable=p_var, width=14).grid(row=0, column=3, sticky="ew")
                pair_frame.columnconfigure(1, weight=1)
                pair_frame.columnconfigure(3, weight=1)
                pair_frame.grid(row=index * 2 + 1, column=0, sticky="ew")
                pair_frame.grid_remove()
                self.paired_inputs[column] = (n_var, p_var)
                self.single_input_widgets[column] = widget
                self.pair_input_widgets[column] = pair_frame
            if column == self.pair_selector_column:
                widget.bind("<<ComboboxSelected>>", lambda _event: self._update_np_input_mode())
        self.count_label.configure(text=f"候选数据（当前 CSV）：{len(self.df):,} 个物料 · {len(numeric)} 个电气参数")
        self._update_np_input_mode()
        # 产品类别切换后表单控件会重建，因此需要重新绑定新控件。
        self._bind_customer_mousewheel(self.customer_panel)
        self.results = pd.DataFrame()
        self.has_run_match = False
        if continuing_autofill:
            self.status_var.set(f"已切换到 {self.catalog_var.get()}，正在准备参数查询…")
        else:
            self.status_var.set("请填写参数后开始匹配")
        self._show(pd.DataFrame())

    def _update_np_input_mode(self) -> None:
        """按当前目录切换 MOSFET N/P 或 BJT NPN/PNP 双输入框。"""
        selector = self.inputs.get(self.pair_selector_column)
        selector_value = selector.get().strip().casefold() if selector else ""
        pair_mode = bool(self.pair_selector_target and selector_value == self.pair_selector_target)
        was_pair_mode = bool(
            self.pair_selector_target
            and self.previous_pair_selector_value == self.pair_selector_target
        )
        previous_was_first = self.previous_pair_selector_value in self.pair_first_single_values
        previous_was_second = self.previous_pair_selector_value in self.pair_second_single_values
        current_is_first = selector_value in self.pair_first_single_values
        current_is_second = selector_value in self.pair_second_single_values
        for column, (n_var, p_var) in self.paired_inputs.items():
            single_var = self.inputs[column]
            single_widget = self.single_input_widgets[column]
            pair_widget = self.pair_input_widgets[column]
            if pair_mode:
                # 仅在进入双值模式时迁移，避免重复选择事件覆盖双框中的编辑。
                if not was_pair_mode:
                    current = single_var.get().strip()
                    parts = [part.strip() for part in current.split("/")]
                    if len(parts) == 2 and all(parts):
                        n_var.set(parts[0])
                        p_var.set(parts[1])
                    elif previous_was_first:
                        n_var.set(current)
                    elif previous_was_second:
                        p_var.set(current)
                    elif current and current != "±":
                        n_var.set(current)
                        if self.pair_selector_column == "Channel" and is_shared_np_rating(column):
                            p_var.set(current)
                single_widget.grid_remove()
                pair_widget.grid()
            else:
                # 离开双值模式时把所选侧迁回单框；空值也同步，避免旧值复现。
                if was_pair_mode and current_is_first:
                    single_var.set(n_var.get().strip())
                elif was_pair_mode and current_is_second:
                    single_var.set(p_var.get().strip())
                pair_widget.grid_remove()
                single_widget.grid()
        self.previous_pair_selector_value = selector_value

    def _collect_query(self) -> dict[str, str]:
        selector = self.inputs.get(self.pair_selector_column)
        pair_mode = bool(
            selector
            and selector.get().strip().casefold() == self.pair_selector_target
        )
        query: dict[str, str] = {}
        for column, var in self.inputs.items():
            if pair_mode and column in self.paired_inputs:
                n_value = self.paired_inputs[column][0].get().strip()
                p_value = self.paired_inputs[column][1].get().strip()
                n_filled = bool(n_value and n_value != "±")
                p_filled = bool(p_value and p_value != "±")
                if n_filled != p_filled:
                    raise ValueError(
                        f"{column} 的 {self.pair_labels[0]} 和 {self.pair_labels[1]} 参数必须同时填写。"
                    )
                query[column] = f"{n_value}/{p_value}" if n_filled else ""
            else:
                query[column] = var.get().strip()
        query["Product"] = self.product_var.get().strip()
        return query

    def autofill_from_part_number(self) -> None:
        """先识别产品类型，再在对应 CSV 中核对或查询并填入参数。"""
        if self._autofill_busy:
            return
        api_key = self.deepseek_api_key_var.get().strip()
        if not api_key:
            messagebox.showinfo(
                "请输入 DeepSeek API Key",
                "请先在界面中输入 DeepSeek API Key，无需打开终端。\n\n"
                "Key 仅在本次程序运行期间保存在内存中。",
            )
            self.api_key_entry.focus_set()
            return
        part_number = self.product_var.get().strip()
        if not part_number:
            messagebox.showinfo("请输入物料号", "请先输入客户正在使用的完整物料号。")
            self.product_entry.focus_set()
            return
        manufacturer_hint = self.manufacturer_var.get().strip()

        categories = list(dict.fromkeys(
            _catalog_category_name(display_name)
            for display_name in self.catalogs
        ))
        self._autofill_generation += 1
        request_token = self._autofill_generation
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        self._autofill_busy = True
        self.autofill_button.state(["disabled"])
        self.status_var.set(f"正在通过 DeepSeek 识别 {part_number} 的物料类型，请稍候…")

        worker = threading.Thread(
            target=self._run_deepseek_classification,
            args=(result_queue, part_number, categories, api_key, manufacturer_hint),
            daemon=True,
        )
        worker.start()
        self.after(
            100,
            self._poll_classification_result,
            request_token,
            result_queue,
            part_number,
            api_key,
            manufacturer_hint,
        )

    def _product_number_edited(self, _event: tk.Event) -> None:
        """编辑料号即取消当前响应的应用资格；网络线程会自行安全结束。"""
        self._cancel_autofill_for_context_change(
            "料号已修改，旧的 DeepSeek 查询结果将被丢弃"
        )

    @staticmethod
    def _run_deepseek_classification(
        result_queue: queue.Queue[tuple[str, Any]],
        part_number: str,
        categories: list[str],
        api_key: str,
        manufacturer_hint: str,
    ) -> None:
        """后台识别物料类型；该线程不读取或修改任何 Tk 对象。"""
        try:
            result = classify_part_category(
                part_number=part_number,
                categories=categories,
                api_key=api_key,
                manufacturer_hint=manufacturer_hint or None,
            )
        except Exception as exc:
            result_queue.put(("error", str(exc)))
        else:
            result_queue.put(("success", result))

    def _poll_classification_result(
        self,
        request_token: int,
        result_queue: queue.Queue[tuple[str, Any]],
        part_number: str,
        api_key: str,
        manufacturer_hint: str,
    ) -> None:
        """在主线程应用类型识别结果，并继续同一 token 下的参数查询。"""
        if request_token != self._autofill_generation:
            return
        try:
            outcome, payload = result_queue.get_nowait()
        except queue.Empty:
            self.after(
                100,
                self._poll_classification_result,
                request_token,
                result_queue,
                part_number,
                api_key,
                manufacturer_hint,
            )
            return

        if self.product_var.get().strip().casefold() != part_number.casefold():
            self._finish_autofill(request_token)
            self.status_var.set("料号已改变，已丢弃旧的物料类型识别结果")
            return
        if outcome == "error":
            self._finish_autofill(request_token)
            self.status_var.set("DeepSeek 物料类型识别失败；现有输入未改变")
            messagebox.showerror("物料类型识别失败", str(payload))
            return

        category, returned_part, manufacturer, notes = _classification_payload(payload)
        if returned_part and returned_part.casefold() != part_number.casefold():
            self._finish_autofill(request_token)
            self.status_var.set("DeepSeek 返回的料号与当前输入不一致，结果已丢弃")
            messagebox.showwarning(
                "物料类型识别结果已丢弃",
                f"当前输入：{part_number}\nDeepSeek 返回：{returned_part}\n\n请检查料号后重试。",
            )
            return

        if _classification_is_ambiguous(payload):
            self._finish_autofill(request_token)
            detail = f"\n\n查询提示：{'；'.join(notes[:5])}" if notes else ""
            if not manufacturer_hint:
                self.status_var.set("发现同型号可能属于多个制造商；请选择制造商后重试")
                messagebox.showwarning(
                    "请选择制造商",
                    "DeepSeek 发现该型号存在重名或制造商歧义。\n\n"
                    "请在‘制造商（可选）’菜单中选择客户原料的制造商，然后重试。"
                    f"{detail}",
                )
                self.manufacturer_box.focus_set()
            else:
                self.status_var.set("所选制造商下仍无法唯一确定物料；现有输入未改变")
                messagebox.showwarning(
                    "制造商条件下仍有歧义",
                    f"DeepSeek 在制造商‘{manufacturer_hint}’下仍无法唯一确定该物料。"
                    "请核对完整型号，或手动选择物料类型并填写参数。"
                    f"{detail}",
                )
            return

        if manufacturer_hint and not _same_manufacturer(manufacturer, manufacturer_hint):
            self._finish_autofill(request_token)
            self.status_var.set("返回的制造商与选择项不一致，识别结果已拒绝")
            messagebox.showwarning(
                "制造商不一致",
                "DeepSeek 返回的制造商与您选择的制造商不一致，"
                "为避免填入同名物料的错误参数，本次结果未应用。",
            )
            return

        catalog_name = _catalog_display_name(category, self.catalogs)
        if catalog_name is None or not manufacturer:
            self._finish_autofill(request_token)
            self.status_var.set("DeepSeek 无法确定受支持的物料类型；请手动选择并填写参数")
            detail = f"\n\n查询提示：{'；'.join(notes[:5])}" if notes else ""
            messagebox.showwarning(
                "无法确定物料类型",
                "DeepSeek 未能把该料号唯一归入当前支持的产品类别。"
                "请手动选择物料类型并填写参数。"
                f"{detail}",
            )
            return

        if self.catalog_var.get() != catalog_name:
            self.catalog_var.set(catalog_name)
            self.load_selected_catalog(
                preserve_part_number=part_number,
                autofill_token=request_token,
            )

        # load_selected_catalog 会重建表单；再次核对 token、类别和料号后才继续。
        if (
            request_token != self._autofill_generation
            or self.catalog_var.get() != catalog_name
            or self.product_var.get().strip().casefold() != part_number.casefold()
        ):
            return

        current_values = product_values(self.df, part_number)
        if current_values is not None and _same_manufacturer(
            current_values.get("Manufacture"), manufacturer
        ):
            self._finish_autofill(request_token)
            self._apply_autofill_values(
                {column: current_values.get(column) for column in self.inputs},
                uncertain_fields=[],
                notes=notes,
                source="CSV 候选数据",
            )
            return

        self._start_parameter_lookup(
            request_token=request_token,
            part_number=part_number,
            catalog_name=catalog_name,
            expected_manufacturer=manufacturer_hint or manufacturer,
            api_key=api_key,
        )

    def _start_parameter_lookup(
        self,
        *,
        request_token: int,
        part_number: str,
        catalog_name: str,
        expected_manufacturer: str,
        api_key: str,
    ) -> None:
        """在类型已确定、表单已载入后启动第二阶段参数查询。"""
        if (
            request_token != self._autofill_generation
            or self.catalog_var.get() != catalog_name
            or self.product_var.get().strip().casefold() != part_number.casefold()
        ):
            return

        allowed_fields = list(self.inputs)
        categorical_options = {
            column: options(self.df, column)
            for column in self.categorical_input_columns
            if column in self.inputs
        }
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        self._autofill_busy = True
        self.autofill_button.state(["disabled"])
        self.status_var.set(
            f"已识别为 {catalog_name}，正在通过 DeepSeek 查询 {part_number} 的参数…"
        )

        worker = threading.Thread(
            target=self._run_deepseek_lookup,
            args=(
                result_queue,
                part_number,
                catalog_name,
                allowed_fields,
                categorical_options,
                expected_manufacturer,
                api_key,
            ),
            daemon=True,
        )
        worker.start()
        self.after(
            100,
            self._poll_autofill_result,
            request_token,
            result_queue,
            part_number,
            catalog_name,
            expected_manufacturer,
        )

    @staticmethod
    def _run_deepseek_lookup(
        result_queue: queue.Queue[tuple[str, Any]],
        part_number: str,
        catalog_name: str,
        allowed_fields: list[str],
        categorical_options: dict[str, list[str]],
        expected_manufacturer: str,
        api_key: str,
    ) -> None:
        """后台线程入口：只处理普通数据和网络调用，不接触任何 Tk 对象。"""
        try:
            result = lookup_part_parameters(
                part_number=part_number,
                category=_catalog_category_name(catalog_name),
                allowed_fields=allowed_fields,
                categorical_options=categorical_options,
                expected_manufacturer=expected_manufacturer,
                api_key=api_key,
            )
        except Exception as exc:  # 服务层会把 HTTP/解析错误转换成可显示文本。
            result_queue.put(("error", str(exc)))
        else:
            result_queue.put(("success", result))

    def _poll_autofill_result(
        self,
        request_token: int,
        result_queue: queue.Queue[tuple[str, Any]],
        part_number: str,
        catalog_name: str,
        expected_manufacturer: str,
    ) -> None:
        """在 Tk 主线程消费后台结果，并丢弃已经过期或跨类别的响应。"""
        if request_token != self._autofill_generation:
            return
        try:
            outcome, payload = result_queue.get_nowait()
        except queue.Empty:
            self.after(
                100,
                self._poll_autofill_result,
                request_token,
                result_queue,
                part_number,
                catalog_name,
                expected_manufacturer,
            )
            return

        if (
            self.catalog_var.get() != catalog_name
            or self.product_var.get().strip().casefold() != part_number.casefold()
        ):
            self._finish_autofill(request_token)
            self.status_var.set("料号或产品类别已改变，已丢弃旧的自动查询结果")
            return
        if outcome == "error":
            self._finish_autofill(request_token)
            self.status_var.set("DeepSeek 自动查询失败；现有输入未改变")
            messagebox.showerror("自动填参失败", str(payload))
            return

        returned_part, returned_manufacturer = _result_identity(payload)
        if (
            (returned_part and returned_part.casefold() != part_number.casefold())
            or not _same_manufacturer(returned_manufacturer, expected_manufacturer)
        ):
            self._finish_autofill(request_token)
            self.status_var.set("DeepSeek 两阶段识别结果不一致，参数已拒绝")
            messagebox.showwarning(
                "参数结果已拒绝",
                "参数查询返回的料号或制造商与类型识别阶段不一致，"
                "为避免把其他厂商的参数写入表单，本次结果未应用。",
            )
            return

        parameters, uncertain_fields, notes = _lookup_payload(payload)
        self._finish_autofill(request_token)
        self._apply_autofill_values(
            parameters,
            uncertain_fields=uncertain_fields,
            notes=notes,
            source="DeepSeek",
        )

    def _finish_autofill(self, request_token: int) -> bool:
        """只允许当前请求恢复按钮，避免旧线程干扰更新的请求。"""
        if request_token != self._autofill_generation:
            return False
        self._autofill_busy = False
        self.autofill_button.state(["!disabled"])
        return True

    def _reset_parameter_inputs(self) -> None:
        """只清空参数表单，保留物料号和已有推荐结果。"""
        for column, var in self.inputs.items():
            var.set(self.input_defaults.get(column, ""))
        for column, (first_var, second_var) in self.paired_inputs.items():
            pair_default = self.input_defaults.get(column, "")
            first_var.set(pair_default)
            second_var.set(pair_default)
        self._update_np_input_mode()

    def _apply_autofill_values(
        self,
        parameters: Mapping[str, Any],
        *,
        uncertain_fields: list[str],
        notes: list[str],
        source: str,
    ) -> None:
        """校验自动结果并填表；任何未知字段、非法数值或新分类值都不会进入模型。"""
        uncertain_keys = {_option_key(field) for field in uncertain_fields}
        categorical_values: dict[str, str] = {}
        numeric_values: dict[str, str] = {}
        skipped: list[str] = []
        unavailable: list[str] = []

        for column, raw_value in parameters.items():
            if column not in self.inputs:
                skipped.append(f"{column}（当前类别无此字段）")
                continue

            value = raw_value
            if isinstance(raw_value, Mapping):
                confidence = str(raw_value.get("confidence", "")).strip().casefold()
                value = raw_value.get("value")
                if confidence in {"low", "unknown", "uncertain"}:
                    uncertain_keys.add(_option_key(column))
            if _option_key(column) in uncertain_keys:
                skipped.append(f"{column}（信息不确定）")
                continue

            text = _usable_text(value)
            if not text:
                unavailable.append(column)
                continue
            if column in self.categorical_input_columns:
                accepted = _existing_option(text, options(self.df, column))
                if accepted is None:
                    skipped.append(f"{column}（不在当前下拉选项中：{text}）")
                    continue
                categorical_values[column] = accepted
            elif column in self.numeric_input_columns:
                if not parse_numeric_values(text):
                    skipped.append(f"{column}（无法识别为严格数值：{text}）")
                    continue
                numeric_values[column] = text
            else:
                skipped.append(f"{column}（不是当前可填参数）")

        if not categorical_values and not numeric_values:
            self.status_var.set(f"{source} 没有返回可安全填入的参数；现有输入未改变")
            lines = [
                f"来源：{source}",
                "没有参数通过字段、分类选项和数值格式校验，因此现有输入保持不变。",
            ]
            if skipped:
                lines.append(f"已跳过：{_field_summary(skipped)}")
            if unavailable:
                lines.append(f"无可靠值/留空：{_field_summary(unavailable)}")
            if notes:
                lines.append(f"查询提示：{'；'.join(notes[:5])}")
            if source == "DeepSeek":
                messagebox.showwarning("DeepSeek 未填入参数", "\n\n".join(lines))
            else:
                messagebox.showinfo("CSV 候选数据没有可填参数", "\n\n".join(lines))
            return

        # 自动填参是一次完整替换；这里只清参数，不清物料号和右侧既有结果。
        self._reset_parameter_inputs()
        for column, value in categorical_values.items():
            self.inputs[column].set(value)
        self._update_np_input_mode()

        selector = self.inputs.get(self.pair_selector_column)
        pair_mode = bool(
            selector
            and selector.get().strip().casefold() == self.pair_selector_target
        )
        filled: list[str] = list(categorical_values)
        for column, text in numeric_values.items():
            if pair_mode and column in self.paired_inputs:
                parts = [part.strip() for part in text.split("/")]
                if len(parts) == 2 and all(parts):
                    self.paired_inputs[column][0].set(parts[0])
                    self.paired_inputs[column][1].set(parts[1])
                elif len(parts) == 1 and is_shared_np_rating(column):
                    # VGS 是整颗器件的共享绝对额定值，可以安全写入两侧。
                    self.paired_inputs[column][0].set(text)
                    self.paired_inputs[column][1].set(text)
                else:
                    skipped.append(f"{column}（{self.pair_labels[0]}/{self.pair_labels[1]} 需要两个值）")
                    continue
            else:
                self.inputs[column].set(text)
            filled.append(column)

        skipped.extend(
            f"{field}（服务标记为不确定）"
            for field in uncertain_fields
            if _option_key(field) not in {_option_key(item.split("（", 1)[0]) for item in skipped}
        )
        self.status_var.set(
            f"{source} 已填入 {len(filled)} 项，跳过 {len(skipped)} 项；请核对后再开始匹配"
        )

        lines = [
            f"来源：{source}",
            f"已填入：{_field_summary(filled)}",
        ]
        if skipped:
            lines.append(f"已跳过：{_field_summary(skipped)}")
        if unavailable:
            lines.append(f"无可靠值/留空：{_field_summary(unavailable)}")
        if notes:
            lines.append(f"查询提示：{'；'.join(notes[:5])}")
        lines.append(
            "本次操作没有自动运行匹配；产品类型未改变时保留右侧结果，"
            "自动切换类型时清空旧结果。"
        )
        if source == "DeepSeek":
            lines.append("DeepSeek 结果可能存在识别错误；关键参数、单位和测试条件必须对照原厂规格书人工核对。")
            messagebox.showwarning("DeepSeek 自动填参完成", "\n\n".join(lines))
        else:
            messagebox.showinfo("CSV 候选数据参数已填入", "\n\n".join(lines))

    def clear_inputs(self) -> None:
        """清空当前类别的客户输入，并保留已经显示的匹配结果。"""
        if self._autofill_busy:
            self._autofill_generation += 1
            self._autofill_busy = False
            self.autofill_button.state(["!disabled"])
        self.product_var.set("")
        self.manufacturer_var.set("")
        # 保留本次会话的 Key，但清空操作后恢复遮罩，避免它继续明文显示。
        self.show_api_key_var.set(False)
        self.api_key_entry.configure(show="*")
        self._reset_parameter_inputs()
        self.status_var.set("输入已清空；当前匹配结果已保留")

    def run_match(self) -> None:
        try:
            query = self._collect_query()
            self.results = recommend(self.df, query, top_k=int(self.top_var.get()))
        except Exception as exc:
            messagebox.showerror("无法匹配", str(exc))
            return
        self.has_run_match = True
        self._refresh_results_view()

    def _refresh_results_view(self) -> None:
        """立即应用精简列与车规级行筛选，不重新执行匹配。"""
        filtered = build_result_view(
            self.results,
            automotive_only=self.automotive_only_var.get(),
        )
        display = build_result_view(
            filtered,
            compact=self.compact_results_var.get(),
            critical_columns=self.critical_result_columns,
        )
        self._show(display)
        if not self.has_run_match:
            self.status_var.set("请填写参数后开始匹配")
            return
        if self.results.empty:
            self.status_var.set("没有满足封装和最低参数覆盖率的物料")
            return
        if filtered.empty:
            self.status_var.set(f"完整结果有 {len(self.results)} 个候选，当前筛选下没有车规级物料")
            return
        verified = int((filtered["关键参数检查"] == "通过").sum())
        if self.automotive_only_var.get():
            self.status_var.set(
                f"显示 {len(filtered)}/{len(self.results)} 个车规级候选，其中 {verified} 个通过关键参数规则"
            )
        else:
            self.status_var.set(
                f"找到 {len(filtered)} 个候选，其中 {verified} 个通过关键参数规则"
            )

    def _show(self, data: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(data.columns)
        for column in data.columns:
            self.tree.heading(column, text=column)
            if column in {"关键参数问题", "缺失参数", "风险提示", "核对提示"}:
                width = 300
            else:
                width = 145 if column != "Product" else 170
            self.tree.column(column, width=width, anchor="center")
        for _, row in data.iterrows():
            values = []
            for col, value in row.items():
                if col in {"综合得分", "已知参数匹配度", "参数覆盖率", "方向偏好得分"}:
                    values.append(f"{float(value):.1f}%")
                elif isinstance(value, float):
                    values.append("" if pd.isna(value) else f"{value:g}")
                else:
                    values.append("" if pd.isna(value) else value)
            status = str(row.get("关键参数检查", ""))
            tag = {"通过": "passed", "待人工确认": "pending", "不满足": "failed"}.get(status, "")
            self.tree.insert("", "end", values=values, tags=(tag,) if tag else ())
        self.tree.xview_moveto(0)
        self.tree.yview_moveto(0)

    def export_results(self) -> None:
        if self.results.empty:
            messagebox.showinfo("暂无结果", "请先执行匹配。")
            return
        export_data = build_result_view(
            self.results,
            automotive_only=self.automotive_only_var.get(),
        )
        if export_data.empty:
            messagebox.showinfo("暂无结果", "当前筛选下没有可导出的物料。")
            return
        initialfile = "车规级匹配结果.csv" if self.automotive_only_var.get() else "匹配结果.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv")],
            initialfile=initialfile,
        )
        if path:
            # 精简显示只影响表格列；导出保留完整字段。车规级筛选会同步到导出行。
            export_data.to_csv(Path(path), index=False, encoding="utf-8-sig")
            messagebox.showinfo("导出成功", f"结果已保存到：\n{path}")


if __name__ == "__main__":
    MatchingApp().mainloop()
