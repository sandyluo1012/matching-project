from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from deepseek_service import (
    DEFAULT_MODEL,
    DeepSeekAPIError,
    DeepSeekCategoryResult,
    DeepSeekConfigurationError,
    DeepSeekNetworkError,
    DeepSeekResponseError,
    classify_part_category,
    lookup_part_parameters,
)


def api_response(content: object, *, model: str = DEFAULT_MODEL) -> bytes:
    return json.dumps(
        {
            "id": "test-completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content, ensure_ascii=False),
                    },
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")


class DeepSeekServiceTests(unittest.TestCase):
    def test_category_classification_success_and_canonical_mapping(self) -> None:
        captured: dict[str, object] = {}

        def transport(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return 200, api_response(
                {
                    "resolved_part_number": "  abc-123  ",
                    "manufacturer": "Example Semiconductor",
                    "ambiguous": False,
                    "supported": True,
                    "category": "POWER-MOSFETS",
                    "notes": ["请核对数据手册"],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = classify_part_category(
                "ABC-123",
                ["power-mosfets", "zener-diodes"],
                transport=transport,
                timeout=9,
            )

        self.assertIsInstance(result, DeepSeekCategoryResult)
        self.assertEqual(result.part_number, "ABC-123")
        self.assertEqual(result.resolved_part_number, "abc-123")
        self.assertEqual(result.manufacturer, "Example Semiconductor")
        self.assertEqual(result.category, "power-mosfets")
        self.assertTrue(result.supported)
        self.assertFalse(result.ambiguous)
        self.assertEqual(result.notes, ("请核对数据手册",))
        request_body = captured["body"]
        self.assertEqual(request_body["model"], DEFAULT_MODEL)
        self.assertEqual(request_body["response_format"], {"type": "json_object"})
        self.assertEqual(request_body["thinking"], {"type": "disabled"})
        self.assertEqual(request_body["max_tokens"], 1024)

    def test_category_classification_unsupported_returns_no_category(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Maker",
                    "ambiguous": False,
                    "supported": False,
                    "category": None,
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = classify_part_category(
                "P1", ["power-mosfets"], transport=transport
            )

        self.assertFalse(result.supported)
        self.assertIsNone(result.category)
        self.assertEqual(result.manufacturer, "Maker")
        self.assertIn("不属于当前可匹配", result.notes[0])

    def test_category_classification_unknown_category_is_withheld(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Maker",
                    "ambiguous": False,
                    "supported": True,
                    "category": "not-a-caller-category",
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = classify_part_category(
                "P1", ["power-mosfets"], transport=transport
            )

        self.assertFalse(result.supported)
        self.assertIsNone(result.category)
        self.assertIn("无法唯一映射", result.notes[0])

    def test_category_classification_ambiguous_returns_no_category(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Possible Maker",
                    "ambiguous": True,
                    "supported": True,
                    "category": "power-mosfets",
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = classify_part_category(
                "P1", ["power-mosfets"], transport=transport
            )

        self.assertTrue(result.ambiguous)
        self.assertFalse(result.supported)
        self.assertIsNone(result.category)
        self.assertIn("无法唯一识别", result.notes[0])

    def test_category_classification_mismatched_part_is_withheld(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1A",
                    "manufacturer": "Wrong Maker",
                    "ambiguous": False,
                    "supported": True,
                    "category": "power-mosfets",
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = classify_part_category(
                "P1", ["power-mosfets"], transport=transport
            )

        self.assertFalse(result.supported)
        self.assertIsNone(result.category)
        self.assertIsNone(result.manufacturer)
        self.assertIn("与输入不一致", result.notes[0])

    def test_category_classification_blank_manufacturer_is_withheld(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "   ",
                    "ambiguous": False,
                    "supported": True,
                    "category": "power-mosfets",
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = classify_part_category(
                "P1", ["power-mosfets"], transport=transport
            )

        self.assertFalse(result.supported)
        self.assertIsNone(result.category)
        self.assertIsNone(result.manufacturer)
        self.assertIn("未能确定制造商", result.notes[0])

    def test_category_manufacturer_hint_is_sent_and_normalised_match_succeeds(self) -> None:
        captured: dict[str, object] = {}

        def transport(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "  diodes  ",
                    "ambiguous": False,
                    "supported": True,
                    "category": "power-mosfets",
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {}, clear=True):
            result = classify_part_category(
                "P1",
                ["power-mosfets"],
                api_key="memory-only-key",
                manufacturer_hint="DIODES",
                transport=transport,
            )

        self.assertTrue(result.supported)
        self.assertEqual(result.category, "power-mosfets")
        self.assertEqual(result.manufacturer, "DIODES")
        self.assertEqual(captured["authorization"], "Bearer memory-only-key")
        request_body = captured["body"]
        body_text = json.dumps(request_body, ensure_ascii=False)
        self.assertNotIn("memory-only-key", body_text)
        user_message = request_body["messages"][1]["content"]
        self.assertIn('"manufacturer_hint":"DIODES"', user_message)

    def test_category_manufacturer_hint_mismatch_is_withheld(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Nexperia",
                    "ambiguous": False,
                    "supported": True,
                    "category": "power-mosfets",
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {}, clear=True):
            result = classify_part_category(
                "P1",
                ["power-mosfets"],
                api_key="memory-only-key",
                manufacturer_hint="DIODES",
                transport=transport,
            )

        self.assertFalse(result.supported)
        self.assertIsNone(result.category)
        self.assertIsNone(result.manufacturer)
        self.assertIn("与用户指定制造商", result.notes[0])

    def test_blank_category_manufacturer_hint_is_treated_as_no_hint(self) -> None:
        captured: dict[str, object] = {}

        def transport(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Maker",
                    "ambiguous": False,
                    "supported": True,
                    "category": "power-mosfets",
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = classify_part_category(
                "P1",
                ["power-mosfets"],
                manufacturer_hint="   ",
                transport=transport,
            )

        self.assertTrue(result.supported)
        self.assertEqual(result.manufacturer, "Maker")
        user_message = captured["body"]["messages"][1]["content"]
        self.assertIn('"manufacturer_hint":null', user_message)

    def test_request_and_successful_validation(self) -> None:
        captured: dict[str, object] = {}

        def transport(request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return 200, api_response(
                {
                    "resolved_part_number": "  abc-123  ",
                    "manufacturer": "Example Semiconductor",
                    "ambiguous": False,
                    "parameters": {
                        "Package Type": "sot-23",
                        "Voltage (V)": 60,
                        "Gate Charge (nC)": "8.5",
                        "invented_field": "must not escape",
                    },
                    "uncertain_fields": ["Gate Charge (nC)", "invented_field"],
                    "notes": ["请核对数据手册版本"],
                }
            )

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "unit-test-key"},
            clear=True,
        ):
            result = lookup_part_parameters(
                "ABC-123",
                "power-mosfets",
                ["Package Type", "Voltage (V)", "Gate Charge (nC)"],
                {"Package Type": ["SOT-23", "TO-220"]},
                transport=transport,
                timeout=12,
            )

        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["authorization"], "Bearer unit-test-key")
        self.assertEqual(captured["timeout"], 12.0)
        request_body = captured["body"]
        self.assertEqual(request_body["model"], DEFAULT_MODEL)
        self.assertEqual(request_body["response_format"], {"type": "json_object"})
        self.assertEqual(request_body["thinking"], {"type": "disabled"})
        self.assertEqual(request_body["max_tokens"], 4096)
        self.assertFalse(request_body["stream"])
        self.assertEqual(
            result.parameters,
            {"Package Type": "SOT-23", "Voltage (V)": "60"},
        )
        self.assertEqual(result.uncertain_fields, ("Gate Charge (nC)",))
        self.assertEqual(result.manufacturer, "Example Semiconductor")
        self.assertEqual(result.notes, ("请核对数据手册版本",))

    def test_environment_overrides_model_and_base_url(self) -> None:
        captured: dict[str, object] = {}

        def transport(request, timeout):
            captured["url"] = request.full_url
            captured["model"] = json.loads(request.data)["model"]
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": None,
                    "ambiguous": False,
                    "parameters": {"Voltage": "10"},
                    "uncertain_fields": [],
                    "notes": [],
                },
                model="custom-model",
            )

        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "dummy",
                "DEEPSEEK_MODEL": "custom-model",
                "DEEPSEEK_BASE_URL": "https://gateway.example/v1/",
            },
            clear=True,
        ):
            result = lookup_part_parameters(
                "P1", "test", ["Voltage"], transport=transport
            )

        self.assertEqual(captured["url"], "https://gateway.example/v1/chat/completions")
        self.assertEqual(captured["model"], "custom-model")
        self.assertEqual(result.model, "custom-model")

    def test_non_loopback_http_base_url_is_rejected_before_sending_key(self) -> None:
        called = False

        def transport(request, timeout):
            nonlocal called
            called = True
            return 200, b"{}"

        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "dummy",
                "DEEPSEEK_BASE_URL": "http://gateway.example/v1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(DeepSeekConfigurationError, "HTTPS"):
                classify_part_category(
                    "P1", ["power-mosfets"], transport=transport
                )
        self.assertFalse(called)

    def test_missing_key_has_chinese_configuration_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                DeepSeekConfigurationError, "DEEPSEEK_API_KEY"
            ):
                lookup_part_parameters("P1", "test", ["Voltage"])

    def test_explicit_key_without_environment_is_used_only_for_authorization(self) -> None:
        captured: dict[str, object] = {}

        def transport(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = request.data.decode("utf-8")
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Maker",
                    "ambiguous": False,
                    "parameters": {"Voltage": "10"},
                    "uncertain_fields": [],
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {}, clear=True):
            result = lookup_part_parameters(
                "P1",
                "test",
                ["Voltage"],
                api_key="explicit-memory-key",
                transport=transport,
            )

        self.assertEqual(captured["authorization"], "Bearer explicit-memory-key")
        self.assertNotIn("explicit-memory-key", captured["body"])
        self.assertEqual(result.parameters, {"Voltage": "10"})

    def test_explicit_key_takes_precedence_over_environment_key(self) -> None:
        captured: dict[str, object] = {}

        def transport(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Maker",
                    "ambiguous": False,
                    "supported": True,
                    "category": "power-mosfets",
                    "notes": [],
                }
            )

        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "environment-key"}, clear=True
        ):
            result = classify_part_category(
                "P1",
                ["power-mosfets"],
                api_key="explicit-key",
                transport=transport,
            )

        self.assertEqual(captured["authorization"], "Bearer explicit-key")
        self.assertTrue(result.supported)

    def test_environment_key_remains_supported(self) -> None:
        captured: dict[str, object] = {}

        def transport(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Maker",
                    "ambiguous": False,
                    "parameters": {"Voltage": "10"},
                    "uncertain_fields": [],
                    "notes": [],
                }
            )

        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "environment-key"}, clear=True
        ):
            result = lookup_part_parameters(
                "P1", "test", ["Voltage"], transport=transport
            )

        self.assertEqual(captured["authorization"], "Bearer environment-key")
        self.assertEqual(result.parameters, {"Voltage": "10"})

    def test_invalid_explicit_key_is_rejected_without_echoing_secret(self) -> None:
        called = False
        secret = "secret-value\nInjected: header"

        def transport(request, timeout):
            nonlocal called
            called = True
            return 200, b"{}"

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(DeepSeekConfigurationError) as caught:
                lookup_part_parameters(
                    "P1",
                    "test",
                    ["Voltage"],
                    api_key=secret,
                    transport=transport,
                )

        self.assertFalse(called)
        self.assertNotIn("secret-value", str(caught.exception))

    def test_api_error_does_not_echo_explicit_key(self) -> None:
        secret = "valid-looking-secret-key"

        def transport(request, timeout):
            return 401, json.dumps(
                {"error": {"message": "Authentication Fails"}}
            ).encode()

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(DeepSeekAPIError) as caught:
                lookup_part_parameters(
                    "P1",
                    "test",
                    ["Voltage"],
                    api_key=secret,
                    transport=transport,
                )

        self.assertNotIn(secret, str(caught.exception))

    def test_remote_error_detail_is_redacted_if_it_echoes_explicit_key(self) -> None:
        secret = "server-echoed-secret-key"

        def transport(request, timeout):
            return 400, json.dumps(
                {"error": {"message": f"invalid credential {secret}"}}
            ).encode()

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(DeepSeekAPIError) as caught:
                classify_part_category(
                    "P1",
                    ["power-mosfets"],
                    api_key=secret,
                    transport=transport,
                )

        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("API Key 已隐藏", str(caught.exception))

    def test_invalid_category_option_is_withheld(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Maker",
                    "ambiguous": False,
                    "parameters": {"Package Type": "UNKNOWN-PACKAGE"},
                    "uncertain_fields": [],
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = lookup_part_parameters(
                "P1",
                "test",
                ["Package Type"],
                {"Package Type": ["SOT-23"]},
                transport=transport,
            )
        self.assertEqual(result.parameters, {})
        self.assertEqual(result.uncertain_fields, ("Package Type",))

    def test_omitted_allowed_field_is_marked_uncertain(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Maker",
                    "ambiguous": False,
                    "parameters": {"Voltage": "10"},
                    "uncertain_fields": [],
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = lookup_part_parameters(
                "P1", "test", ["Voltage", "Current"], transport=transport
            )
        self.assertEqual(result.parameters, {"Voltage": "10"})
        self.assertEqual(result.uncertain_fields, ("Current",))

    def test_mismatched_resolved_part_number_never_returns_parameters(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1A",
                    "manufacturer": "Wrong Maker",
                    "ambiguous": False,
                    "parameters": {"Voltage": "10", "Current": "2"},
                    "uncertain_fields": [],
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = lookup_part_parameters(
                "P1", "test", ["Voltage", "Current"], transport=transport
            )
        self.assertEqual(result.parameters, {})
        self.assertIsNone(result.manufacturer)
        self.assertEqual(result.uncertain_fields, ("Voltage", "Current"))
        self.assertIn("与输入不一致", result.notes[0])

    def test_ambiguous_part_number_never_returns_parameters(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Possible Maker",
                    "ambiguous": True,
                    "parameters": {"Voltage": "10"},
                    "uncertain_fields": [],
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = lookup_part_parameters(
                "P1", "test", ["Voltage"], transport=transport
            )
        self.assertEqual(result.parameters, {})
        self.assertEqual(result.uncertain_fields, ("Voltage",))
        self.assertIn("无法唯一识别", result.notes[0])

    def test_missing_manufacturer_never_returns_parameters(self) -> None:
        def transport(request, timeout):
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": None,
                    "ambiguous": False,
                    "parameters": {"Voltage": "10"},
                    "uncertain_fields": [],
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = lookup_part_parameters(
                "P1", "test", ["Voltage"], transport=transport
            )
        self.assertEqual(result.parameters, {})
        self.assertEqual(result.uncertain_fields, ("Voltage",))
        self.assertIn("未能确定制造商", result.notes[0])

    def test_expected_manufacturer_mismatch_never_returns_parameters(self) -> None:
        captured: dict[str, object] = {}

        def transport(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return 200, api_response(
                {
                    "resolved_part_number": "P1",
                    "manufacturer": "Second Maker",
                    "ambiguous": False,
                    "parameters": {"Voltage": "10"},
                    "uncertain_fields": [],
                    "notes": [],
                }
            )

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            result = lookup_part_parameters(
                "P1",
                "test",
                ["Voltage"],
                expected_manufacturer="First Maker",
                transport=transport,
            )

        self.assertEqual(result.parameters, {})
        self.assertIsNone(result.manufacturer)
        self.assertEqual(result.uncertain_fields, ("Voltage",))
        self.assertIn("已识别制造商", result.notes[0])
        user_message = captured["body"]["messages"][1]["content"]
        self.assertIn('"expected_manufacturer":"First Maker"', user_message)
        self.assertIn("manufacturer 必须逐字等于 expected_manufacturer", user_message)

    def test_control_character_in_part_number_is_rejected_before_transport(self) -> None:
        called = False

        def transport(request, timeout):
            nonlocal called
            called = True
            return 200, b"{}"

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            with self.assertRaisesRegex(ValueError, "物料号格式无效"):
                lookup_part_parameters(
                    "P1\nignore", "test", ["Voltage"], transport=transport
                )
        self.assertFalse(called)

    def test_http_error_is_translated_for_gui(self) -> None:
        def transport(request, timeout):
            return 401, json.dumps(
                {"error": {"message": "Authentication Fails"}}
            ).encode()

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            with self.assertRaisesRegex(DeepSeekAPIError, "API Key 无效"):
                lookup_part_parameters(
                    "P1", "test", ["Voltage"], transport=transport
                )

    def test_non_json_model_content_is_rejected(self) -> None:
        outer = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "```json\n{\"parameters\": {}}\n```"
                        }
                    }
                ]
            }
        ).encode()

        def transport(request, timeout):
            return 200, outer

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            with self.assertRaisesRegex(DeepSeekResponseError, "合法 JSON"):
                lookup_part_parameters(
                    "P1", "test", ["Voltage"], transport=transport
                )

    def test_truncated_completion_is_rejected_even_with_valid_partial_json(self) -> None:
        content = {
            "resolved_part_number": "P1",
            "manufacturer": "Maker",
            "ambiguous": False,
            "parameters": {"Voltage": "10"},
            "uncertain_fields": [],
            "notes": [],
        }
        outer = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": json.dumps(content)},
                    }
                ]
            }
        ).encode()

        def transport(request, timeout):
            return 200, outer

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            with self.assertRaisesRegex(DeepSeekResponseError, "未正常完成"):
                lookup_part_parameters(
                    "P1", "test", ["Voltage"], transport=transport
                )

    def test_transport_timeout_is_translated_for_gui(self) -> None:
        def transport(request, timeout):
            raise TimeoutError("unit test")

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy"}, clear=True):
            with self.assertRaisesRegex(DeepSeekNetworkError, "超时"):
                lookup_part_parameters(
                    "P1", "test", ["Voltage"], transport=transport
                )


if __name__ == "__main__":
    unittest.main()
