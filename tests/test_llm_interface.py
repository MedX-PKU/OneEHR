from __future__ import annotations

from oneehr.config.schema import SystemConfig


def test_build_llm_request_uses_backend_defaults_and_params():
    from oneehr.agent.llm import build_llm_request

    req = build_llm_request(
        SystemConfig(
            backend="deepseek",
            model="deepseek-v4-flash",
            api_key_env="DEEPSEEK_API_KEY",
            params={"temperature": 0.0, "timeout_seconds": 10, "seed": 7},
        ),
        "prompt",
        "system",
    )

    assert req.base_url == "https://api.deepseek.com/v1"
    assert req.provider_model == "deepseek-v4-flash"
    assert req.temperature == 0.0
    assert req.timeout_seconds == 10
    assert req.seed == 7
