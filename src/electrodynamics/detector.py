import jax
import jax.numpy as jnp
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class DetectorParameters:
    "Parameters for the simulated (rectangular) detector."

    width: float
    "Detector width (in atomic units)"

    height: float
    "Detector height (in atomic units)"

    z_distance: float
    "Detector distance from the origin in the `z` direction (in atomic units)"

    grid_size_x: int
    "Number of points for discretizing the detector in the `x` direction"

    grid_size_y: int
    "Number of points for discretizing the detector in the `y` direction"


def initialize_detector_positions(parameters: DetectorParameters) -> jax.Array:
    "Initializes an array of detector positions."

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
