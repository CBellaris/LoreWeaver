from pathlib import Path
from types import SimpleNamespace
import unittest

from loreweaver.config import AppConfig
from loreweaver.model_services import resolve_model_service
from loreweaver.model_services.clients.openai_compatible import _uploaded_file_id


class ModelServiceConfigTests(unittest.TestCase):
    def test_requires_canonical_service_config(self) -> None:
        config = AppConfig(
            path=Path("models.yaml"),
            values={
                "providers": {
                    "siliconflow": {
                        "api_key_env": "SILICONFLOW_API_KEY",
                        "base_url": "https://api.siliconflow.cn/v1",
                    }
                },
            },
        )

        with self.assertRaisesRegex(ValueError, "not configured"):
            resolve_model_service(models_config=config, service="embedding")

    def test_resolves_profile_based_service_config(self) -> None:
        config = AppConfig(
            path=Path("models.yaml"),
            values={
                "providers": {"deepseek": {"api_key_env": "DEEPSEEK_API_KEY"}},
                "model_profiles": {
                    "deepseek_eval": {
                        "capability": "chat",
                        "provider": "deepseek",
                        "model": "deepseek-v4-pro",
                        "temperature": 0.2,
                    }
                },
                "services": {
                    "eval_question_generator": {
                        "profile": "deepseek_eval",
                        "max_output_tokens": 384000,
                    }
                },
            },
        )

        service = resolve_model_service(
            models_config=config,
            service="eval_question_generator",
        )

        self.assertEqual(service.provider.name, "deepseek")
        self.assertEqual(service.model, "deepseek-v4-pro")
        self.assertEqual(service.temperature, 0.2)
        self.assertEqual(service.max_output_tokens, 384000)

    def test_uploaded_file_id_prefers_siliconflow_nested_data_id(self) -> None:
        uploaded = SimpleNamespace(id="outer-id", data={"id": "nested-file-id"})

        self.assertEqual(_uploaded_file_id(uploaded), "nested-file-id")

    def test_uploaded_file_id_accepts_openai_style_id(self) -> None:
        uploaded = SimpleNamespace(id="file-openai")

        self.assertEqual(_uploaded_file_id(uploaded), "file-openai")


if __name__ == "__main__":
    unittest.main()
