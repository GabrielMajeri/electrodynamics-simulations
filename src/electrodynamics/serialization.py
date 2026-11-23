import json
from datetime import datetime
from json import JSONEncoder
from pathlib import Path
from typing import Any

import jax
import jax_dataclasses as jdc
import numpy as np

from electrodynamics.beams import GaussianBeamParameters
from electrodynamics.pulse import GaussianPulseParameters


class ConfigurationEncoder(JSONEncoder):
    "JSON encoder for experiment configuration dictionaries."

    def default(self, obj):
        if jdc.is_dataclass(obj):
            if isinstance(obj, GaussianBeamParameters):
                d = jdc.asdict(obj)
                # Since the waist radius is usually expressed as a multiple of the wavelength,
                # include a human-readable value here as well.
                d["waist_radius_in_multiples_of_lambda"] = (
                    f"{d['waist_radius'] / d['wavelength']} * wavelength"
                )
                return d

            if isinstance(obj, GaussianPulseParameters):
                d = jdc.asdict(obj)
                d["phi_0_in_multiples_of_tau_0"] = f"{d['phi_0'] / d['tau_0']} * tau_0"
                return d

            return jdc.asdict(obj)  # pyright: ignore[reportArgumentType]

        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}

        if isinstance(obj, (jax.Array, np.ndarray)):
            return obj.tolist()

        if isinstance(obj, np.generic):
            return obj.item()

        return super().default(obj)


def prepare_output_directory(
    experiment_name: str,
    run_datetime: datetime,
    overwrite: bool = False,
    overwrite_run_tag: str = "latest",
    run_tag: str = "",
    timestamp_suffix: str = "",
    runs_root: Path = Path("runs"),
) -> Path:
    "Creates and returns the output directory for a simulation run."
    if run_tag:
        output_directory = runs_root / experiment_name / run_tag
    elif overwrite:
        output_directory = runs_root / experiment_name / overwrite_run_tag
    else:
        timestamp = run_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        tag = f"{timestamp_suffix}_{timestamp}" if timestamp_suffix else timestamp
        output_directory = runs_root / experiment_name / tag

    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def save_json(path: Path, data: dict[str, Any]) -> None:
    "Serializes `data` to JSON using `ConfigurationEncoder`."
    with open(path, "w") as file:
        json.dump(data, file, indent=2, cls=ConfigurationEncoder)
