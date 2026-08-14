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
    load_catalog,
    options,
    preference_features,
    recommend,
)


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
        self.catalogs = discover_catalogs()
        self.df = pd.DataFrame()
        self.inputs: dict[str, tk.StringVar] = {}
        self.input_defaults: dict[str, str] = {}
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
        self.catalog_box = ttk.Combobox(bar, textvariable=self.catalog_var, values=list(self.catalogs), state="readonly", width=36)
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

        canvas = tk.Canvas(left, highlightthickness=0, width=350)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self.form = ttk.Frame(canvas)
        self.form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.form, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
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

    def load_selected_catalog(self) -> None:
        self.df = load_catalog(self.catalogs[self.catalog_var.get()])
        self.product_var.set("")
        for widget in self.form.winfo_children():
            widget.destroy()
        self.inputs.clear()
        self.input_defaults.clear()
        numeric, categorical = infer_features(self.df)
        critical = critical_features(self.df)
        preferred = preference_features(self.df)
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
            if is_esd_discharge:
                suffix += "  [可选择/可输入]"
            label = column + suffix
            ttk.Label(self.form, text=label).grid(row=index * 2, column=0, sticky="w", pady=(7, 2))
            var = tk.StringVar()
            default_value = "±" if is_esd_discharge else ""
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
        self.count_label.configure(text=f"数据库：{len(self.df):,} 个物料 · {len(numeric)} 个电气参数")
        self._show(pd.DataFrame())

    def clear_inputs(self) -> None:
        """清空当前类别的客户输入，并保留已经显示的匹配结果。"""
        self.product_var.set("")
        for column, var in self.inputs.items():
            var.set(self.input_defaults.get(column, ""))
        self.status_var.set("输入已清空；当前匹配结果已保留")

    def run_match(self) -> None:
        query = {column: var.get().strip() for column, var in self.inputs.items()}
        query["Product"] = self.product_var.get().strip()
        try:
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
