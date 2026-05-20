r_0s = particle_displacements

# Compute detector displacement (in each particle frame of reference)
x_0s = detector_positions[:, np.newaxis, :] - centers[np.newaxis, :, :]
x_0s_norms = np.linalg.vector_norm(x_0s, axis=-1)

n_0s = x_0s / x_0s_norms[:, :, np.newaxis]

if False:
    frequency = omega_laser * 1.00
    n_0_dot_r_0 = np.vecdot(n_0s[0], r_0s[0])
    exponent = frequency * timestamps - frequency / c * n_0_dot_r_0

    if False:
        plt.title("$\\frac{d}{dt} \\left(n_0 \\cdot r_0\\right) (t)$")

        # TODO: Need to include timestamps for dt!!
        plt.plot(timestamps, np.gradient(n_0_dot_r_0), label="Derivative")
        plt.axhline(
            c,
            xmin=0,
            xmax=integration_duration,
            color="orange",
            label="Speed of light",
        )

        plt.grid()
        plt.legend()

        plt.xlabel("$t$")
        plt.ylabel("Derivative")

        plt.savefig("plots/derivative_of_n_0_dot_r_0.pdf")
        plt.close()

    # Check again that the exponent doesn't go to 0
    # TODO: Need to include timestamps for dt!!
    assert np.all(np.abs(np.gradient(n_0_dot_r_0, axis=-1)) < c)

    if False:
        # Plot imaginary part of the exponent of the oscillatory kernel
        plt.title("Exponent ($g(t)$)")
        plt.plot(timestamps, exponent)
        plt.grid()
        plt.savefig("plots/g.pdf")
        plt.close()

    if False:
        # Plot the derivative of the exponent
        plt.title("$g'(t)$")
        plt.plot(timestamps, np.gradient(exponent))
        plt.grid()
        plt.savefig("plots/dg_dt.pdf")
        plt.close()

    oscillatory_kernel = np.exp(1j * exponent)

    if False:
        plt.title("Oscillatory kernel")
        plt.plot(timestamps, np.angle(oscillatory_kernel), marker=".", linewidth=0)
        plt.xlabel("$t$")
        plt.ylabel("")
        plt.grid()
        plt.savefig("plots/exp_i_g.pdf")
        plt.close()

    # The integrand is now the oscillatory kernel times the derivative of the position term
    # TODO: Need to include timestamps for dt!!
    integrand = oscillatory_kernel * np.gradient(n_0_dot_r_0)

    if False:
        plt.title("Integrand")
        plt.plot(timestamps, np.abs(integrand), marker=".", linewidth=0)
        plt.xlabel("$t$")
        plt.ylabel("")
        plt.grid()
        plt.savefig("plots/integrand.pdf")
        plt.close()

    # TODO: use an integration method specialized for highly-oscillatory integrals

    dt = timestamps[1] - timestamps[0]
    # integral = (1 / x_0s_norms[0]) * dt * np.sum(np.exp(1j * exponent))
    integral = dt * np.sum(integrand)
    print("Integral value:", integral)
    print("Integral absolute value:", np.abs(integral))

dt = timestamps[1] - timestamps[0]

frequency = omega_laser

print("Computing (n_0, r_0(t)) dot products")

start_time = perf_counter()

n_0s_dot_r_0s = np.vecdot(n_0s[:, :, np.newaxis, :], r_0s)

g = frequency * timestamps - frequency / c * n_0s_dot_r_0s

exponent = 1j * g

end_time = perf_counter()
duration = end_time - start_time
print(f"Computing exponents took {duration:.4g} seconds")

start_time = perf_counter()

n_0s_dot_v_0s = np.gradient(n_0s_dot_r_0s, axis=-1)

end_time = perf_counter()
duration = end_time - start_time
print(f"Computing velocities took {duration:.4g} seconds")

start_time = perf_counter()

oscillatory_kernel = np.exp(exponent)
integrand = oscillatory_kernel * n_0s_dot_v_0s

end_time = perf_counter()
duration = end_time - start_time
print(f"Computing integrands took {duration:.4g} seconds")

start_time = perf_counter()

# Approximate integral using Riemann sum
result = (1 / x_0s_norms) * dt * np.sum(integrand, axis=-1)

# Sum across particles
result = np.sum(result, axis=-1)

end_time = perf_counter()
duration = end_time - start_time
print(f"Computing integrals using Riemann sums took {duration:.4g} seconds")

final_time = perf_counter()
total_duration = final_time - initial_time
print(f"Execution took a total of {total_duration:.3g} seconds")

print("Plotting results")

plt.title("Integration results")

plt.plot(detector_positions[:, 0], np.abs(result), label="$|\\phi(x_0)|$")
# plt.plot(phi, np.angle(result), label="$\\arg \\phi(x_0)$")

plt.xlabel("$x_0$")
plt.ylabel("Scalar potential absolute value ($|\\phi|$)")

# plt.xlabel("$\\phi$")
# plt.ylabel("Scalar potential argument ($\\arg \\phi$)")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("plots/scalar_potential_vs_x_0.pdf")
plt.close()
