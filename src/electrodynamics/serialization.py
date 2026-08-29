from json import JSONEncoder

import jax
import jax_dataclasses as jdc

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

        if isinstance(obj, jax.Array):
            return obj.tolist()

        return super().default(obj)
