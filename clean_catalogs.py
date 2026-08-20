"""清理 MCC 原始产品表；保留局部 NA，供匹配模型做缺失感知计算。"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from matching_model import is_bounded_numeric_value, parse_numeric_values


BASE_DIR = Path(__file__).resolve().parent / "products" / "MCC"
RAW_DIR = BASE_DIR / "raw"
CLEAN_DIR = BASE_DIR / "clean"
REPORT_PATH = BASE_DIR / "cleaning_report.md"
COMMON_COLUMNS = {
    "Manufacture", "Product", "Status", "Compliance", "Number of Functions",
    "Configuration", "Package Type", "Polarity", "Channel", "ESD Diodes",
}
MISSING_MARKERS = {"", "-", "—", "nan", "none", "n/a", "na", "null"}
OUTER_FORMAT_CHARACTERS = {"\u200b", "\ufeff"}


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
class TrimmedCell:
    category: str
    product: str
    field: str
    original: str
    cleaned: str


@dataclass
class InvalidNumericCell:
    category: str
    product: str
    field: str
    value: str
    reason: str


@dataclass
class CleanStats:
    category: str
    before: int
    after: int
    removed: list[RemovedPart] = field(default_factory=list)
    trimmed_cells: list[TrimmedCell] = field(default_factory=list)
    invalid_numeric_cells: list[InvalidNumericCell] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    remaining_missing: dict[str, int] = field(default_factory=dict)


def trim_cell(value: str) -> str:
    """只移除单元格首尾空白/不可见格式符，绝不拼接内部数字。"""
    start = 0
    end = len(value)
    while start < end and (
        value[start].isspace() or value[start] in OUTER_FORMAT_CHARACTERS
    ):
        start += 1
    while end > start and (
        value[end - 1].isspace() or value[end - 1] in OUTER_FORMAT_CHARACTERS
    ):
        end -= 1
    return value[start:end]


def clean_one(source: Path, target: Path) -> CleanStats:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3:
        raise ValueError(f"{source.name} 缺少 MCC 的三行表头")

    width = len(rows[1])
    title_row = (rows[0] + [""] * width)[:width]
    header_raw = (rows[1] + [""] * width)[:width]
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
    trimmed_cells: list[TrimmedCell] = []
    for original_row in body:
        row = [trim_cell(value) for value in original_row]
        product = row[product_index]
        for index, (original, cleaned) in enumerate(zip(original_row, row)):
            if original != cleaned:
                trimmed_cells.append(
                    TrimmedCell(
                        category=category,
                        product=product,
                        field=header[index] or f"未命名列 {index + 1}",
                        original=original,
                        cleaned=cleaned,
                    )
                )
        problems = [(header[index], row[index]) for index in parameter_indexes if "," in row[index]]
        if problems:
            removed.append(RemovedPart(category, product, problems))
        else:
            kept_rows.append(row)

    invalid_numeric_cells: list[InvalidNumericCell] = []
    for row in kept_rows:
        for index in parameter_indexes:
            value = row[index]
            if is_missing(value) or parse_numeric_values(value):
                continue
            if is_bounded_numeric_value(value):
                reason = "有界值不能作为精确数值，匹配时按未知处理"
            else:
                reason = "格式无法安全解析，匹配时按未知处理"
            invalid_numeric_cells.append(
                InvalidNumericCell(
                    category=category,
                    product=row[product_index],
                    field=header[index],
                    value=value,
                    reason=reason,
                )
            )

    # 全空列没有匹配信息；未命名列含义不明。两者均不进入 clean 数据和 GUI。
    keep_indexes = [
        index for index in range(width)
        if header[index] and not all(is_missing(row[index]) for row in kept_rows)
    ]
    dropped_columns = [
        header[index] or f"未命名列 {index + 1}"
        for index in range(width) if index not in keep_indexes
    ]
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
        # clean 文件采用数据库友好的单表头格式；单位已包含在规范化字段名中。
        writer.writerow(header)
        writer.writerows(df.fillna("").astype(str).values.tolist())
    return CleanStats(
        category=category,
        before=len(body),
        after=len(df),
        removed=removed,
        trimmed_cells=trimmed_cells,
        invalid_numeric_cells=invalid_numeric_cells,
        dropped_columns=dropped_columns,
        remaining_missing=remaining_missing,
    )


def _mapping_text(values: dict[str, int]) -> str:
    return "；".join(f"{column}：{count}" for column, count in values.items()) or "无"


def _cell_text(value: str) -> str:
    """让不可见空白在 Markdown 清洗报告中可见。"""
    return repr(value).replace("`", "\\`")


def main() -> None:
    sources = sorted(RAW_DIR.glob("*.csv"))
    if not sources:
        raise FileNotFoundError(f"没有在 {RAW_DIR} 找到原始 MCC CSV")

    stats = [clean_one(source, CLEAN_DIR / source.name) for source in sources]
    report: list[str] = [
        "# MCC 产品表清理报告", "",
        "输出格式：UTF-8 with BOM、单行标准表头；字段名中的换行、不间断空格和连续空格均已合并。数据单元格只移除首尾 Unicode 空白及常见不可见格式符，内部字符保持不变。", "",
        "缺失值策略：删除数据区整列为空的字段；保留局部 NA，不生成推算值。匹配模型只比较双方均有值的参数，并计算覆盖率。", "",
        "数值格式策略：完整数值、科学计数、± 数值和双分量斜杠值可以解析；有界值或夹杂未知字符的值保留原文并在匹配时按未知处理。", "",
        "含逗号策略：如果一个物料的任意参数字段包含逗号，则删除整颗物料；通用信息字段不参与此规则。", "",
        "| 产品类别 | 原始数量 | 清理后数量 | 删除物料 | 修正首尾空白 | 无法安全解析数值 | 删除不可用列 | 保留空白 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        report.append(
            f"| {item.category} | {item.before} | {item.after} | {len(item.removed)} | "
            f"{len(item.trimmed_cells)} | "
            f"{len(item.invalid_numeric_cells)} | "
            f"{len(item.dropped_columns)} | {sum(item.remaining_missing.values())} |"
        )

    report.extend(["", "## 修正的单元格首尾空白", ""])
    trimmed_any = False
    for item in stats:
        for cell in item.trimmed_cells:
            trimmed_any = True
            report.append(
                f"- **{cell.category} / {cell.product or '（空物料号）'} / {cell.field}**："
                f"`{_cell_text(cell.original)}` → `{_cell_text(cell.cleaned)}`"
            )
    if not trimmed_any:
        report.append("没有发现需要修正的单元格首尾空白。")

    report.extend(["", "## 无法安全转换的数值", ""])
    invalid_any = False
    for item in stats:
        for cell in item.invalid_numeric_cells:
            invalid_any = True
            report.append(
                f"- **{cell.category} / {cell.product or '（空物料号）'} / {cell.field}**："
                f"`{_cell_text(cell.value)}`；{cell.reason}。"
            )
    if not invalid_any:
        report.append("没有发现无法安全转换的数值。")

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
    print(f"修正单元格首尾空白 {sum(len(item.trimmed_cells) for item in stats)} 处")
    print(f"发现无法安全解析数值 {sum(len(item.invalid_numeric_cells) for item in stats)} 处")
    print(f"删除全空或未命名列 {sum(len(item.dropped_columns) for item in stats)} 列")
    print(f"保留局部空白 {sum(sum(item.remaining_missing.values()) for item in stats)} 个")
    for item in stats:
        for part in item.removed:
            details = "; ".join(f"{field}={value}" for field, value in part.problems)
            print(f"删除 {part.category}: {part.product} ({details})")
    print(f"报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
