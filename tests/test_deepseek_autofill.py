from __future__ import annotations

import unittest
import queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from deepseek_service import DeepSeekCategoryResult
from app import (
    DEEPSEEK_MANUFACTURERS,
    MatchingApp,
    _catalog_category_name,
    _catalog_display_name,
    _classification_is_ambiguous,
    _classification_payload,
    _existing_option,
    _same_manufacturer,
)


class DummyVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: object) -> None:
        self.value = str(value)


class DummyWidget:
    def __init__(self) -> None:
        self.visible = True
        self.focused = False
        self.states: list[list[str]] = []

    def grid(self) -> None:
        self.visible = True

    def grid_remove(self) -> None:
        self.visible = False

    def focus_set(self) -> None:
        self.focused = True

    def state(self, state: list[str]) -> None:
        self.states.append(state)


def fake_app() -> MatchingApp:
    app = object.__new__(MatchingApp)
    id_column = "Drain Current ID (A)"
    vgs_column = "Gate-Source Voltage VGS (V)"
    app.inputs = {
        "Package Type": DummyVar("OLD-PACKAGE"),
        "Channel": DummyVar("N"),
        id_column: DummyVar("9"),
        vgs_column: DummyVar("±"),
    }
    app.input_defaults = {
        "Package Type": "",
        "Channel": "",
        id_column: "",
        vgs_column: "±",
    }
    app.paired_inputs = {
        id_column: (DummyVar(""), DummyVar("")),
        vgs_column: (DummyVar("±"), DummyVar("±")),
    }
    app.single_input_widgets = {
        id_column: DummyWidget(),
        vgs_column: DummyWidget(),
    }
    app.pair_input_widgets = {
        id_column: DummyWidget(),
        vgs_column: DummyWidget(),
    }
    app.pair_selector_column = "Channel"
    app.pair_selector_target = "n+p"
    app.pair_labels = ("N沟道", "P沟道")
    app.pair_first_single_values = {"n", "n+n"}
    app.pair_second_single_values = {"p", "p+p"}
    app.previous_pair_selector_value = "n"
    app.numeric_input_columns = {id_column, vgs_column}
    app.categorical_input_columns = {"Package Type", "Channel"}
    app.df = pd.DataFrame(
        {"Package Type": ["SOP-8"], "Channel": ["N+P"]}
    )
    app.status_var = DummyVar()
    app.product_var = DummyVar("CUSTOM-1")
    app.results = pd.DataFrame({"Product": ["OLD-RESULT"]})
    return app


class DeepSeekAutofillTests(unittest.TestCase):
    def test_manufacturer_menu_has_the_requested_options_in_order(self) -> None:
        self.assertEqual(
            DEEPSEEK_MANUFACTURERS,
            (
                "DIODES", "Infineon", "Nexperia", "ON", "st", "TI", "VISHAY",
                "强茂", "瞬雷", "扬杰", "长电科技", "长晶", "贝岭", "杰华特", "小华",
                "NXP", "聚鼎", "力特", "LRC", "槟城",
            ),
        )

    def test_category_maps_only_to_an_existing_catalog(self) -> None:
        catalogs = {
            "Power Mosfets（功率MOSFETs）": Path("power.csv"),
            "Zener Diodes（稳压二极管）": Path("zener.csv"),
        }
        self.assertEqual(
            _catalog_category_name("Power Mosfets（功率MOSFETs）"),
            "Power Mosfets",
        )
        self.assertEqual(
            _catalog_display_name("power mosfets", catalogs),
            "Power Mosfets（功率MOSFETs）",
        )
        self.assertIsNone(_catalog_display_name("IGBT", catalogs))

    def test_unsupported_classification_never_exposes_a_category(self) -> None:
        category, part, manufacturer, notes = _classification_payload(
            {
                "part_number": "ABC-1",
                "manufacturer": "Example",
                "category": "Power Mosfets",
                "supported": False,
                "ambiguous": False,
                "notes": ["不受支持"],
            }
        )
        self.assertEqual(category, "")
        self.assertEqual(part, "ABC-1")
        self.assertEqual(manufacturer, "Example")
        self.assertEqual(notes, ["不受支持"])

    def test_ambiguous_flag_is_available_to_the_gui(self) -> None:
        self.assertTrue(_classification_is_ambiguous({"ambiguous": True}))
        self.assertFalse(_classification_is_ambiguous({"ambiguous": False}))

    def test_mcc_manufacturer_aliases_match_but_other_brands_do_not(self) -> None:
        self.assertTrue(
            _same_manufacturer(
                "MCC (Micro Commercial Components)",
                "MCC",
            )
        )
        self.assertFalse(
            _same_manufacturer(
                "MCC (Micro Commercial Components)",
                "onsemi",
            )
        )

    def test_classification_switches_catalog_then_uses_the_new_schema(self) -> None:
        target = "Power Mosfets（功率MOSFETs）"
        calls: dict[str, object] = {}
        fake = SimpleNamespace(
            _autofill_generation=7,
            _autofill_busy=True,
            product_var=DummyVar("ABC-1"),
            catalog_var=DummyVar("Zener Diodes（稳压二极管）"),
            catalogs={
                target: Path("power.csv"),
                "Zener Diodes（稳压二极管）": Path("zener.csv"),
            },
            df=pd.DataFrame({"Product": []}),
            status_var=DummyVar(),
        )

        def load_selected_catalog(**kwargs) -> None:
            calls["load"] = kwargs
            fake.df = pd.DataFrame({"Product": []})

        def start_parameter_lookup(**kwargs) -> None:
            calls["lookup"] = kwargs

        fake.load_selected_catalog = load_selected_catalog
        fake._start_parameter_lookup = start_parameter_lookup
        fake._finish_autofill = lambda token: True
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        result_queue.put(
            (
                "success",
                DeepSeekCategoryResult(
                    part_number="ABC-1",
                    resolved_part_number="ABC-1",
                    manufacturer="Example Semiconductor",
                    category="Power Mosfets",
                    ambiguous=False,
                    supported=True,
                    notes=(),
                    model="test-model",
                ),
            )
        )

        MatchingApp._poll_classification_result(
            fake,
            request_token=7,
            result_queue=result_queue,
            part_number="ABC-1",
            api_key="gui-secret",
            manufacturer_hint="",
        )

        self.assertEqual(fake.catalog_var.get(), target)
        self.assertEqual(
            calls["load"],
            {"preserve_part_number": "ABC-1", "autofill_token": 7},
        )
        self.assertEqual(calls["lookup"]["catalog_name"], target)
        self.assertEqual(
            calls["lookup"]["expected_manufacturer"],
            "Example Semiconductor",
        )
        self.assertEqual(calls["lookup"]["api_key"], "gui-secret")

    @patch("app.classify_part_category")
    def test_classification_worker_passes_key_and_manufacturer_snapshot(
        self,
        classify,
    ) -> None:
        classify.return_value = {"category": "Power Mosfets"}
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        MatchingApp._run_deepseek_classification(
            result_queue,
            "ABC-1",
            ["Power Mosfets"],
            "gui-secret",
            "ON",
        )

        classify.assert_called_once_with(
            part_number="ABC-1",
            categories=["Power Mosfets"],
            api_key="gui-secret",
            manufacturer_hint="ON",
        )
        self.assertEqual(result_queue.get_nowait()[0], "success")

    @patch("app.lookup_part_parameters")
    def test_lookup_worker_reuses_the_same_key_and_selected_manufacturer(
        self,
        lookup,
    ) -> None:
        lookup.return_value = {"parameters": {}}
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        MatchingApp._run_deepseek_lookup(
            result_queue,
            "ABC-1",
            "Power Mosfets（功率MOSFETs）",
            ["Package Type"],
            {"Package Type": ["SOT-23"]},
            "ON",
            "gui-secret",
        )

        lookup.assert_called_once_with(
            part_number="ABC-1",
            category="Power Mosfets",
            allowed_fields=["Package Type"],
            categorical_options={"Package Type": ["SOT-23"]},
            expected_manufacturer="ON",
            api_key="gui-secret",
        )
        self.assertEqual(result_queue.get_nowait()[0], "success")

    @patch("app.messagebox.showwarning")
    def test_ambiguous_result_prompts_for_manufacturer_without_switching_catalog(
        self,
        warning,
    ) -> None:
        manufacturer_box = DummyWidget()
        old_results = pd.DataFrame({"Product": ["OLD"]})
        current_catalog = "Zener Diodes（稳压二极管）"
        fake = SimpleNamespace(
            _autofill_generation=3,
            _autofill_busy=True,
            product_var=DummyVar("DUP-1"),
            catalog_var=DummyVar(current_catalog),
            catalogs={current_catalog: Path("zener.csv")},
            status_var=DummyVar(),
            manufacturer_box=manufacturer_box,
            results=old_results,
        )

        def finish(token: int) -> bool:
            self.assertEqual(token, 3)
            fake._autofill_busy = False
            return True

        fake._finish_autofill = finish
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        result_queue.put(
            (
                "success",
                {
                    "part_number": "DUP-1",
                    "manufacturer": "",
                    "category": None,
                    "ambiguous": True,
                    "supported": True,
                    "notes": ["发现多个制造商"],
                },
            )
        )

        MatchingApp._poll_classification_result(
            fake,
            request_token=3,
            result_queue=result_queue,
            part_number="DUP-1",
            api_key="gui-secret",
            manufacturer_hint="",
        )

        self.assertEqual(fake.catalog_var.get(), current_catalog)
        self.assertIs(fake.results, old_results)
        self.assertTrue(manufacturer_box.focused)
        self.assertIn("请选择制造商", fake.status_var.get())
        warning.assert_called_once()

    @patch("app.messagebox.showwarning")
    def test_selected_manufacturer_mismatch_is_rejected_before_catalog_switch(
        self,
        warning,
    ) -> None:
        current_catalog = "Zener Diodes（稳压二极管）"
        old_results = pd.DataFrame({"Product": ["OLD"]})
        fake = SimpleNamespace(
            _autofill_generation=4,
            _autofill_busy=True,
            product_var=DummyVar("DUP-1"),
            catalog_var=DummyVar(current_catalog),
            catalogs={current_catalog: Path("zener.csv")},
            status_var=DummyVar(),
            results=old_results,
            _finish_autofill=lambda _token: True,
        )
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        result_queue.put(
            (
                "success",
                {
                    "part_number": "DUP-1",
                    "manufacturer": "Nexperia",
                    "category": "Power Mosfets",
                    "ambiguous": False,
                    "supported": True,
                },
            )
        )

        MatchingApp._poll_classification_result(
            fake,
            request_token=4,
            result_queue=result_queue,
            part_number="DUP-1",
            api_key="gui-secret",
            manufacturer_hint="ON",
        )

        self.assertEqual(fake.catalog_var.get(), current_catalog)
        self.assertIs(fake.results, old_results)
        self.assertIn("制造商与选择项不一致", fake.status_var.get())
        warning.assert_called_once()

    @patch("app.messagebox.showinfo")
    def test_autofill_without_key_prompts_and_focuses_key_entry(self, info) -> None:
        api_key_entry = DummyWidget()
        fake = SimpleNamespace(
            _autofill_busy=False,
            deepseek_api_key_var=DummyVar("  "),
            api_key_entry=api_key_entry,
        )

        MatchingApp.autofill_from_part_number(fake)

        self.assertTrue(api_key_entry.focused)
        info.assert_called_once()

    def test_manufacturer_change_invalidates_an_inflight_request(self) -> None:
        button = DummyWidget()
        fake = SimpleNamespace(
            _autofill_generation=9,
            _autofill_busy=True,
            autofill_button=button,
            status_var=DummyVar(),
        )
        fake._cancel_autofill_for_context_change = (
            lambda status: MatchingApp._cancel_autofill_for_context_change(fake, status)
        )

        MatchingApp._manufacturer_edited(fake, None)

        self.assertEqual(fake._autofill_generation, 10)
        self.assertFalse(fake._autofill_busy)
        self.assertEqual(button.states, [["!disabled"]])
        self.assertIn("制造商已修改", fake.status_var.get())

    def test_clear_inputs_keeps_api_key_but_clears_manufacturer(self) -> None:
        key_entry_config: dict[str, str] = {}
        fake = SimpleNamespace(
            _autofill_busy=False,
            product_var=DummyVar("ABC-1"),
            manufacturer_var=DummyVar("ON"),
            deepseek_api_key_var=DummyVar("gui-secret"),
            show_api_key_var=DummyVar("1"),
            api_key_entry=SimpleNamespace(
                configure=lambda **kwargs: key_entry_config.update(kwargs)
            ),
            status_var=DummyVar(),
            _reset_parameter_inputs=lambda: None,
        )

        MatchingApp.clear_inputs(fake)

        self.assertEqual(fake.product_var.get(), "")
        self.assertEqual(fake.manufacturer_var.get(), "")
        self.assertEqual(fake.deepseek_api_key_var.get(), "gui-secret")
        self.assertEqual(fake.show_api_key_var.get(), "False")
        self.assertEqual(key_entry_config, {"show": "*"})

    def test_option_mapping_is_case_and_whitespace_tolerant_but_not_fuzzy(self) -> None:
        self.assertEqual(_existing_option("  sot-23 ", ["SOT-23"]), "SOT-23")
        self.assertIsNone(_existing_option("SOT23", ["SOT-23"]))

    @patch("app.messagebox.showwarning")
    def test_pair_values_are_filled_without_changing_results(self, warning) -> None:
        app = fake_app()
        old_results = app.results
        id_column = "Drain Current ID (A)"
        vgs_column = "Gate-Source Voltage VGS (V)"

        MatchingApp._apply_autofill_values(
            app,
            {
                "Package Type": "sop-8",
                "Channel": "n+p",
                id_column: "24/-18",
                vgs_column: "±20",
            },
            uncertain_fields=[],
            notes=["请核对规格书"],
            source="DeepSeek",
        )

        self.assertEqual(app.inputs["Package Type"].get(), "SOP-8")
        self.assertEqual(app.inputs["Channel"].get(), "N+P")
        self.assertEqual(
            tuple(var.get() for var in app.paired_inputs[id_column]),
            ("24", "-18"),
        )
        self.assertEqual(
            tuple(var.get() for var in app.paired_inputs[vgs_column]),
            ("±20", "±20"),
        )
        self.assertEqual(app.product_var.get(), "CUSTOM-1")
        self.assertIs(app.results, old_results)
        self.assertIn("已填入 4 项", app.status_var.get())
        warning.assert_called_once()

    @patch("app.messagebox.showwarning")
    def test_no_valid_value_preserves_existing_inputs(self, warning) -> None:
        app = fake_app()
        before = {column: var.get() for column, var in app.inputs.items()}

        MatchingApp._apply_autofill_values(
            app,
            {
                "Unknown Field": "1",
                "Drain Current ID (A)": "not-a-number",
            },
            uncertain_fields=[],
            notes=["无法确认"],
            source="DeepSeek",
        )

        self.assertEqual(
            {column: var.get() for column, var in app.inputs.items()},
            before,
        )
        self.assertIn("现有输入未改变", app.status_var.get())
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
