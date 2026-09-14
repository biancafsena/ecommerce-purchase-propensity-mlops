"""Tests for application settings."""

import pytest
from pydantic import ValidationError

from purchase_propensity.config import Settings, get_settings


def test_settings_accept_valid_values() -> None:
    """Settings should accept valid explicit values."""
    settings = Settings(
        environment="test",
        random_seed=123,
        model_decision_threshold=0.65,
        api_port=9000,
        _env_file=None,
    )

    assert settings.environment == "test"
    assert settings.random_seed == 123
    assert settings.model_decision_threshold == 0.65
    assert settings.api_port == 9000


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_settings_reject_invalid_threshold(threshold: float) -> None:
    """Threshold should remain between zero and one."""
    with pytest.raises(ValidationError):
        Settings(model_decision_threshold=threshold, _env_file=None)


def test_get_settings_returns_cached_instance() -> None:
    """Settings factory should reuse the validated instance."""
    get_settings.cache_clear()

    first_settings = get_settings()
    second_settings = get_settings()

    assert first_settings is second_settings
