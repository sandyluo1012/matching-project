from __future__ import annotations

import unittest

import pandas as pd

from app import build_result_view


class ResultViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.DataFrame(
            [
                {
                    "Product": "AUTO-1",
                    "Manufacture": "MCC",
                    "综合得分": 95.0,
                    "已知参数匹配度": 96.0,
                    "参数覆盖率": 100.0,
                    "方向偏好得分": 90.0,
                    "可信度": "高",
                    "关键参数检查": "通过",
                    "车规等级": "车规级",
                    "Package Type": "SOT-23",
                    "Channel": "N",
                    "Drain-Source Voltage VDS (V)": 60.0,
                    "Junction Temperature Tj [max] (°C)": 150.0,
                    "关键参数问题": "无",
                    "缺失参数": "无",
                    "风险提示": "无",
                    "核对提示": "核对 Pinout",
                },
                {
                    "Product": "NORMAL-1",
                    "Manufacture": "MCC",
                    "综合得分": 90.0,
                    "已知参数匹配度": 92.0,
                    "参数覆盖率": 90.0,
                    "方向偏好得分": 85.0,
                    "可信度": "中",
                    "关键参数检查": "待人工确认",
                    "车规等级": "非车规级",
                    "Package Type": "SOT-23",
                    "Channel": "N",
                    "Drain-Source Voltage VDS (V)": 60.0,
                    "Junction Temperature Tj [max] (°C)": 175.0,
                    "关键参数问题": "无",
                    "缺失参数": "EAS",
                    "风险提示": "有缺失",
                    "核对提示": "核对 Pinout",
                },
            ]
        )

    def test_automotive_filter_only_changes_rows(self) -> None:
        view = build_result_view(self.results, automotive_only=True)
        self.assertEqual(view["Product"].tolist(), ["AUTO-1"])
        self.assertEqual(view.columns.tolist(), self.results.columns.tolist())

    def test_compact_view_keeps_summary_constraints_and_critical_values(self) -> None:
        view = build_result_view(
            self.results,
            compact=True,
            critical_columns={"Drain-Source Voltage VDS (V)"},
        )
        self.assertEqual(
            view.columns.tolist(),
            [
                "Product", "Manufacture", "综合得分", "已知参数匹配度", "参数覆盖率",
                "可信度", "关键参数检查", "车规等级", "Package Type", "Channel",
                "Drain-Source Voltage VDS (V)", "关键参数问题", "缺失参数", "风险提示",
            ],
        )
        self.assertNotIn("方向偏好得分", view.columns)
        self.assertNotIn("Junction Temperature Tj [max] (°C)", view.columns)
        self.assertNotIn("核对提示", view.columns)

    def test_filters_can_be_combined_without_mutating_full_results(self) -> None:
        original = self.results.copy(deep=True)
        view = build_result_view(
            self.results,
            compact=True,
            automotive_only=True,
            critical_columns={"Drain-Source Voltage VDS (V)"},
        )
        self.assertEqual(view["Product"].tolist(), ["AUTO-1"])
        pd.testing.assert_frame_equal(self.results, original)


if __name__ == "__main__":
    unittest.main()
