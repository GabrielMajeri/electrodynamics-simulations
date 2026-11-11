from collections.abc import Generator

import jax
import pytest


@pytest.fixture
def jax_config() -> Generator[None]:
    "Fixture setting up JAX configuration for tests."
    jax.config.update("jax_enable_x64", True)
    yield
