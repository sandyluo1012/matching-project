"""芯片替代料智能匹配桌面 GUI。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from matching_model import (
    critical_features,
    discover_catalogs,
    infer_features,
    is_shared_np_rating,
    load_catalog,
    options,
    paired_features,
    plus_minus_features,
    preference_features,
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


def _localized_catalogs(catalogs: dict[str, Path]) -> dict[str, Path]:
    """只改变 GUI 显示名称，保留英文名称对应的原始 CSV 路径。"""
    localized: dict[str, Path] = {}
    for english_name, path in catalogs.items():
        chinese_name = CATALOG_CHINESE_NAMES.get(english_name.casefold())
        display_name = f"{english_name}（{chinese_name}）" if chinese_name else english_name
        localized[display_name] = path
    return localized


def _is_esd_discharge_field(column: str) -> bool:
    """识别 ESD 的 IEC 空气/接触放电规格列。"""
    normalized = "".join(column.casefold().split())
    return "iec61000-4-2" in normalized and "air/contact" in normalized


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
        self.paired_inputs: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        self.single_input_widgets: dict[str, tk.Misc] = {}
        self.pair_input_widgets: dict[str, tk.Misc] = {}
        self.previous_channel_value = ""
        self.results = pd.DataFrame()
        self._style()
        self._layout()
        if self.catalogs:
            self.catalog_var.set(next(iter(self.catalogs)))
            self.load_selected_catalog()
        else:
            messagebox.showerror("数据缺失", "未在 products/MCC/clean 文件夹中找到清理后的 CSV 数据。")

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
        ttk.Label(customer, text="客户当前使用的型号（可选）").pack(anchor="w")
        self.product_var = tk.StringVar()
        ttk.Entry(customer, textvariable=self.product_var).pack(fill="x", pady=(4, 0))
        ttk.Label(
            customer,
            text="客户型号无需存在于 MCC 候选库中；请在下方填写其规格参数。",
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

    def load_selected_catalog(self) -> None:
        self.df = load_catalog(self.catalogs[self.catalog_var.get()])
        self.product_var.set("")
        for widget in self.form.winfo_children():
            widget.destroy()
        self.inputs.clear()
        self.input_defaults.clear()
        self.paired_inputs.clear()
        self.single_input_widgets.clear()
        self.pair_input_widgets.clear()
        self.previous_channel_value = ""
        numeric, categorical = infer_features(self.df)
        critical = critical_features(self.df)
        preferred = preference_features(self.df)
        paired = paired_features(self.df)
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
                suffix += "  [N+P双值]"
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
                n_var = tk.StringVar()
                p_var = tk.StringVar()
                ttk.Label(pair_frame, text="N沟道").grid(row=0, column=0, padx=(0, 4))
                ttk.Entry(pair_frame, textvariable=n_var, width=14).grid(row=0, column=1, sticky="ew")
                ttk.Label(pair_frame, text="P沟道").grid(row=0, column=2, padx=(10, 4))
                ttk.Entry(pair_frame, textvariable=p_var, width=14).grid(row=0, column=3, sticky="ew")
                pair_frame.columnconfigure(1, weight=1)
                pair_frame.columnconfigure(3, weight=1)
                pair_frame.grid(row=index * 2 + 1, column=0, sticky="ew")
                pair_frame.grid_remove()
                self.paired_inputs[column] = (n_var, p_var)
                self.single_input_widgets[column] = widget
                self.pair_input_widgets[column] = pair_frame
            if column == "Channel":
                widget.bind("<<ComboboxSelected>>", lambda _event: self._update_np_input_mode())
        self.count_label.configure(text=f"数据库：{len(self.df):,} 个物料 · {len(numeric)} 个电气参数")
        self._update_np_input_mode()
        # 产品类别切换后表单控件会重建，因此需要重新绑定新控件。
        self._bind_customer_mousewheel(self.customer_panel)
        self._show(pd.DataFrame())

    def _update_np_input_mode(self) -> None:
        """Channel=N+P 时显示独立的 N/P 输入框，否则使用原单值输入框。"""
        channel = self.inputs.get("Channel")
        channel_value = channel.get().strip().casefold() if channel else ""
        np_mode = channel_value == "n+p"
        for column, (n_var, p_var) in self.paired_inputs.items():
            single_var = self.inputs[column]
            single_widget = self.single_input_widgets[column]
            pair_widget = self.pair_input_widgets[column]
            if np_mode:
                current = single_var.get().strip()
                if current:
                    parts = [part.strip() for part in current.split("/")]
                    if len(parts) == 2 and all(parts):
                        n_var.set(parts[0])
                        p_var.set(parts[1])
                    elif self.previous_channel_value in {"n", "n+n"}:
                        n_var.set(current)
                    elif self.previous_channel_value in {"p", "p+p"}:
                        p_var.set(current)
                    elif not self.previous_channel_value:
                        n_var.set(current)
                        if is_shared_np_rating(column):
                            p_var.set(current)
                single_widget.grid_remove()
                pair_widget.grid()
            else:
                if channel_value in {"n", "n+n"} and n_var.get().strip():
                    single_var.set(n_var.get().strip())
                elif channel_value in {"p", "p+p"} and p_var.get().strip():
                    single_var.set(p_var.get().strip())
                pair_widget.grid_remove()
                single_widget.grid()
        self.previous_channel_value = channel_value

    def _collect_query(self) -> dict[str, str]:
        channel = self.inputs.get("Channel")
        np_mode = bool(channel and channel.get().strip().casefold() == "n+p")
        query: dict[str, str] = {}
        for column, var in self.inputs.items():
            if np_mode and column in self.paired_inputs:
                n_value = self.paired_inputs[column][0].get().strip()
                p_value = self.paired_inputs[column][1].get().strip()
                if bool(n_value) != bool(p_value):
                    raise ValueError(f"{column} 的 N沟道和 P沟道参数必须同时填写。")
                query[column] = f"{n_value}/{p_value}" if n_value else ""
            else:
                query[column] = var.get().strip()
        query["Product"] = self.product_var.get().strip()
        return query

    def clear_inputs(self) -> None:
        """清空当前类别的客户输入，并保留已经显示的匹配结果。"""
        self.product_var.set("")
        for column, var in self.inputs.items():
            var.set(self.input_defaults.get(column, ""))
        for n_var, p_var in self.paired_inputs.values():
            n_var.set("")
            p_var.set("")
        self._update_np_input_mode()
        self.status_var.set("输入已清空；当前匹配结果已保留")

    def run_match(self) -> None:
        try:
            query = self._collect_query()
            self.results = recommend(self.df, query, top_k=int(self.top_var.get()))
        except Exception as exc:
            messagebox.showerror("无法匹配", str(exc))
            return
        self._show(self.results)
        if self.results.empty:
            self.status_var.set("没有满足封装和最低参数覆盖率的物料")
        else:
            verified = int((self.results["关键参数检查"] == "通过").sum())
            self.status_var.set(f"找到 {len(self.results)} 个候选，其中 {verified} 个通过关键参数规则")

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

    def export_results(self) -> None:
        if self.results.empty:
            messagebox.showinfo("暂无结果", "请先执行匹配。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV 文件", "*.csv")], initialfile="匹配结果.csv")
        if path:
            self.results.to_csv(Path(path), index=False, encoding="utf-8-sig")
            messagebox.showinfo("导出成功", f"结果已保存到：\n{path}")


if __name__ == "__main__":
    MatchingApp().mainloop()
