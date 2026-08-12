"""清理 MCC 原始产品表；保留局部 NA，供匹配模型做缺失感知计算。"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent / "products" / "MCC"
RAW_DIR = BASE_DIR / "raw"
CLEAN_DIR = BASE_DIR / "clean"
REPORT_PATH = BASE_DIR / "cleaning_report.md"
COMMON_COLUMNS = {
    "Manufacture", "Product", "Status", "Compliance", "Number of Functions",
    "Configuration", "Package Type", "Polarity", "Channel",
}
MISSING_MARKERS = {"", "-", "—", "nan", "none", "n/a", "na", "null"}


def clean_column(name: str) -> str:
    return " ".join(str(name).replace("\xa0", " ").split())


def is_missing(value: object) -> bool:
    return pd.isna(value) or str(value).strip().casefold() in MISSING_MARKERS


@dataclass
class RemovedPart:
    category: str
    product: str
    problems: list[tuple[str, str]]


@dataclass
class CleanStats:
    category: str
    before: int
    after: int
    removed: list[RemovedPart] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    remaining_missing: dict[str, int] = field(default_factory=dict)


def clean_one(source: Path, target: Path) -> CleanStats:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3:
        raise ValueError(f"{source.name} 缺少 MCC 的三行表头")

    width = len(rows[1])
    title_row = (rows[0] + [""] * width)[:width]
    header_raw = (rows[1] + [""] * width)[:width]
    unit_row = (rows[2] + [""] * width)[:width]
    header = [clean_column(value) for value in header_raw]
    nonempty_headers = [name for name in header if name]
    if len(set(nonempty_headers)) != len(nonempty_headers):
        raise ValueError(f"{source.name} 存在重复的非空列名，无法安全清理")
    product_index = header.index("Product")
    parameter_indexes = [
        index for index, name in enumerate(header)
        if name and name not in COMMON_COLUMNS
    ]
    category = clean_column(title_row[2]) if len(title_row) > 2 else source.stem
    body = [(row + [""] * width)[:width] for row in rows[3:]]

    kept_rows: list[list[str]] = []
    removed: list[RemovedPart] = []
    for row in body:
        problems = [(header[index], row[index]) for index in parameter_indexes if "," in row[index]]
        if problems:
            removed.append(RemovedPart(category, row[product_index], problems))
        else:
            kept_rows.append(row)

    # 全空列没有匹配信息；未命名列含义不明。两者均不进入 clean 数据和 GUI。
    keep_indexes = [
        index for index in range(width)
        if header[index] and not all(is_missing(row[index]) for row in kept_rows)
    ]
    dropped_columns = [
        header[index] or f"未命名列 {index + 1}"
        for index in range(width) if index not in keep_indexes
    ]
    title_row = [title_row[index] for index in keep_indexes]
    header_raw = [header_raw[index] for index in keep_indexes]
    unit_row = [unit_row[index] for index in keep_indexes]
    header = [header[index] for index in keep_indexes]
    kept_rows = [[row[index] for index in keep_indexes] for row in kept_rows]

    df = pd.DataFrame(kept_rows, columns=header)
    remaining_missing = {
        column: int(df[column].map(is_missing).sum())
        for column in df.columns if df[column].map(is_missing).any()
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows([title_row, header_raw, unit_row])
        writer.writerows(df.fillna("").astype(str).values.tolist())
    return CleanStats(category, len(body), len(df), removed, dropped_columns, remaining_missing)


def _mapping_text(values: dict[str, int]) -> str:
    return "；".join(f"{column}：{count}" for column, count in values.items()) or "无"


def main() -> None:
    sources = sorted(RAW_DIR.glob("*.csv"))
    if not sources:
        raise FileNotFoundError(f"没有在 {RAW_DIR} 找到原始 MCC CSV")

    stats = [clean_one(source, CLEAN_DIR / source.name) for source in sources]
    report: list[str] = [
        "# MCC 产品表清理报告", "",
        "缺失值策略：删除数据区整列为空的字段；保留局部 NA，不生成推算值。匹配模型只比较双方均有值的参数，并计算覆盖率。", "",
        "含逗号策略：如果一个物料的任意参数字段包含逗号，则删除整颗物料；通用信息字段不参与此规则。", "",
        "| 产品类别 | 原始数量 | 清理后数量 | 删除物料 | 删除不可用列 | 保留空白 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        report.append(
            f"| {item.category} | {item.before} | {item.after} | {len(item.removed)} | "
            f"{len(item.dropped_columns)} | {sum(item.remaining_missing.values())} |"
        )

    report.extend(["", "## 删除的全空或未命名列", ""])
    for item in stats:
        if item.dropped_columns:
            report.append(f"- **{item.category}**：" + "、".join(f"`{value}`" for value in item.dropped_columns))

    report.extend(["", "## 被删除的物料", ""])
    removed_any = False
    for item in stats:
        for part in item.removed:
            removed_any = True
            report.append(f"- **{part.category} / {part.product}**")
            report.extend(f"  - `{field}`：`{value}`" for field, value in part.problems)
    if not removed_any:
        report.append("没有物料被删除。")

    report.extend(["", "## 保留的局部空白", ""])
    report.append("这些值保持未知，不会被中位数或近邻值替代。")
    report.append("")
    for item in stats:
        if item.remaining_missing:
            report.append(f"- **{item.category}**：{_mapping_text(item.remaining_missing)}")
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8-sig")

    print(f"已生成 {len(stats)} 份 clean CSV")
    print(f"删除全空或未命名列 {sum(len(item.dropped_columns) for item in stats)} 列")
    print(f"保留局部空白 {sum(sum(item.remaining_missing.values()) for item in stats)} 个")
    for item in stats:
        for part in item.removed:
            details = "; ".join(f"{field}={value}" for field, value in part.problems)
            print(f"删除 {part.category}: {part.product} ({details})")
    print(f"报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
