from pathlib import Path
import unittest

from loreweaver.config import AppConfig
from loreweaver.model_services import resolve_model_service


class ModelServiceConfigTests(unittest.TestCase):
    def test_resolves_legacy_model_config(self) -> None:
        config = AppConfig(
            path=Path("models.yaml"),
            values={
                "providers": {
                    "siliconflow": {
                        "api_key_env": "SILICONFLOW_API_KEY",
                        "base_url": "https://api.siliconflow.cn/v1",
                    }
                },
                "models": {
                    "embedding": {
                        "provider": "siliconflow",
                        "name": "Qwen/Qwen3-Embedding-0.6B",
                        "expected_dimensions": 1024,
                        "batch_size": 16,
                    }
                },
            },
        )

        service = resolve_model_service(models_config=config, service="embedding")

        self.assertEqual(service.provider.name, "siliconflow")
        self.assertEqual(service.model, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(service.expected_dimensions, 1024)
        self.assertEqual(service.batch_size, 16)

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


if __name__ == "__main__":
    unittest.main()
