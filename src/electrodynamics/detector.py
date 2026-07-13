import jax
import jax.numpy as jnp
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class DetectorParameters:
    width: float
    height: float
    z_distance: float

    grid_size_x: int
    grid_size_y: int


def initialize_detector_positions(parameters: DetectorParameters) -> jax.Array:
    ys = jnp.linspace(
        -parameters.height / 2, parameters.height / 2, parameters.grid_size_y
    )
    xs = jnp.linspace(
        -parameters.width / 2, parameters.width / 2, parameters.grid_size_x
    )

    positions: list[tuple[float, float, float]] = []

    for y in ys:
        for x in xs:
            positions.append((x, y, parameters.z_distance))

    return jnp.asarray(positions)
