from json import JSONEncoder

import jax
import jax_dataclasses as jdc


class ConfigurationEncoder(JSONEncoder):
    "JSON encoder for experiment configuration dictionaries."

    def default(self, obj):
        if jdc.is_dataclass(obj):
            return jdc.asdict(obj)  # pyright: ignore[reportArgumentType]

        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}

        if isinstance(obj, jax.Array):
            return obj.tolist()

        return super().default(obj)
