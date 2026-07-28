#include "detector.h++"

#include "constants.h++"

std::vector<Vector3D> initialize_detector_positions()
{
    constexpr auto num_detector_points = detector_grid_size_x * detector_grid_size_y;
    std::vector<Vector3D> detector_positions(num_detector_points);

    for (std::size_t row = 0; row < detector_grid_size_y; ++row)
    {
        for (std::size_t column = 0; column < detector_grid_size_x; ++column)
        {
            const auto y = -detector_height + column * (2 * detector_height) / detector_grid_size_y;
            const auto x = -detector_width + row * (2 * detector_width) / detector_grid_size_x;

            detector_positions[row * detector_grid_size_x + column] = {x, y, detector_z};
        }

        // const auto x = detector_width / 2.0 * std::cos(2 * pi * index / detector_grid_size_x);
        // const auto y = detector_width / 2.0 * std::sin(2 * pi * index / detector_grid_size_x);
        // detector_positions[index] = {x, y, detector_z};
    }

    return detector_positions;
}
