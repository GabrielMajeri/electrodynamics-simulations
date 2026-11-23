# Physics & Unit Conventions

This document is the single source of truth for the physical units, coordinate systems, field normalizations, and sign conventions used across `electrodynamics-simulations`.

---

## 1. Unit System (Gaussian-Like Atomic Units)

All simulations operate in atomic units ($\hbar = m_e = |e| = 4\pi\epsilon_0 = 1$) paired with Gaussian electromagnetic formulas (no $4\pi\epsilon_0$ factors in Coulomb/Liénard-Wiechert terms).

| Quantity | Constant / Value in Code | Physical Meaning | Source |
| --- | --- | --- | --- |
| Speed of light $c$ | `SPEED_OF_LIGHT = 137.036` | $c \approx 1/\alpha$ in a.u. | [`constants.py`](../src/electrodynamics/constants.py#L3) |
| Electron mass $m_e$ | `ELECTRON_MASS = 1` | $1\,m_e$ (Hartree a.u.) | [`constants.py`](../src/electrodynamics/constants.py#L6) |
| Electron charge $q$ | `ELECTRON_CHARGE = -1` | $-1\,|e|$ | [`constants.py`](../src/electrodynamics/constants.py#L9) |
| Hartree energy | `HARTREE_TO_EV = 27.211386` | $1\,\mathrm{Ha}$ in $\mathrm{eV}$ | [`constants.py`](../src/electrodynamics/constants.py#L12) |
| Electron rest energy | `ELECTRON_REST_ENERGY_MEV = 0.5109989` | $m_e c^2$ in $\mathrm{MeV}$ | [`constants.py`](../src/electrodynamics/constants.py#L15) |
| Default laser frequency $\omega_0$ | `DEFAULT_LASER_FREQUENCY = 0.057` | Ti:Sapphire ($\lambda_0 = 2\pi c/\omega_0 \approx 15105.66\,\mathrm{a.u.} \approx 800\,\mathrm{nm}$) | [`constants.py`](../src/electrodynamics/constants.py#L18) |

### Reference SI Conversion Factors

Exact 2018 CODATA values for translating simulation outputs to laboratory SI units (planned helper functions in [`constants.py`](../src/electrodynamics/constants.py) are tracked in [`PLAN.md`](../PLAN.md)):

- Length ($1\,\mathrm{a.u.} = a_0$): `BOHR_RADIUS_M = 5.29177210544e-11` $\mathrm{m}$
- Time ($1\,\mathrm{a.u.} = \hbar / E_h$): `ATOMIC_TIME_S = 2.4188843265864e-17` $\mathrm{s}$
- Energy ($1\,\mathrm{Ha} = E_h$): `HARTREE_TO_JOULES = 4.3597447222060e-18` $\mathrm{J}$
- Charge ($1\,\mathrm{a.u.} = e$): `ELEMENTARY_CHARGE_C = 1.602176634e-19` $\mathrm{C}$

---

## 2. 4-Vectors & Trajectory Integration

- **Position 4-vector**: $x^\mu = (ct, x, y, z)$, shape `(N, 4)`.
- **Momentum 4-vector**: $u^\mu = (p^0, p_x, p_y, p_z) = (\gamma m_e c, \gamma m_e \mathbf{v})$ (with $m_e = 1$), obeying the mass-shell invariant $u_0^2 - |\mathbf{u}|^2 = (m_e c)^2$.
- **Proper-time vs. lab-time integration**:
  - **Proper-time RK4** ([`integration_step_rk4`](../src/electrodynamics/integrate.py#L130), paired with `is_lab_time=False` in [`fields.py`](../src/electrodynamics/fields.py#L136)): integrates with respect to proper time $\tau$, where $dt = \gamma\, d\tau$. Accordingly, `fields.py` absorbs $\gamma$ into the velocity factor $\gamma\boldsymbol{\beta} = \mathbf{u}/c$ when evaluating the Liénard-Wiechert integrand.
  - **Lab-time RK4** ([`integration_step_rk4_lab_time`](../src/electrodynamics/integrate.py#L173), paired with `is_lab_time=True` in [`fields.py`](../src/electrodynamics/fields.py#L136)): integrates directly in laboratory time $t$, using $\boldsymbol{\beta} = \mathbf{v}/c = \mathbf{u}/(\gamma c)$ and $d\mathbf{u}/dt = (q/m_e)(\mathbf{E} + \mathbf{v}\times\mathbf{B}_{\mathrm{beam}})$.
- **Unaccelerated drift subtraction**: Passing `initial_momentum` to [`compute_scattered_electric_and_magnetic_fields`](../src/electrodynamics/fields.py#L129) subtracts $\boldsymbol{\beta}_0 = \mathbf{u}_{\mathrm{init}}/(\gamma_{\mathrm{init}} c)$ (or $\gamma\boldsymbol{\beta}_0$ in proper time) from $\boldsymbol{\beta}(t)$ so uniform coasting outside the laser pulse produces identically zero spurious boundary-window radiation.

---

## 3. Incident Laser Beams & Pulses

### Magnetic Field Scaling (`beams.py`)

In [`src/electrodynamics/beams.py`](../src/electrodynamics/beams.py), the magnetic field is returned in **Lorentz-force units** $\mathbf{B}_{\mathrm{beam}} = \mathbf{B}_{\mathrm{Gauss}} / c$:

$$B_x = -\frac{E_y}{c}, \qquad B_y = \frac{E_x}{c}, \qquad B_z = \frac{i Q}{\omega}\left(y E_x - x E_y\right)\;\text{(approximate, or full }\nabla_\perp\text{ when exact)}.$$

This matches the Lorentz force implementation $\frac{d\mathbf{u}}{dt} = \frac{q}{m_e}(\mathbf{E} + \mathbf{v}\times\mathbf{B}_{\mathrm{beam}})$ in [`fields.py`](../src/electrodynamics/fields.py#L89-L120) without an extra $1/c$ factor. **Do not** confuse $\mathbf{B}_{\mathrm{beam}}$ with the Gaussian-unit scattered field $\tilde{\mathbf{B}}_{\mathrm{Gauss}}$ returned by `fields.py` (Section 4).

### Normalized Amplitude & Polarization

- **Normalized amplitude $a_0$**: Peak electric field is $E_0 = a_0 \frac{m_e c \omega_0}{|q|}$.
- **Polarization Jones vector** ([`polarization.py`](../src/electrodynamics/polarization.py)): Normalized so $|p_x|^2 + |p_y|^2 = 1$. For any unit-norm polarization (linear or circular), the cycle-averaged transverse squared field is $\langle |\mathbf{E}_\perp(\mathbf{x}, t)|^2 \rangle_{\mathrm{cycle}} = \frac{1}{2}|u(x,y)|^2 g(\phi)^2$.
- **Azimuthal phase convention**: [`compute_laguerre_gauss_beam_fields`](../src/electrodynamics/beams.py#L179) multiplies the carrier $\exp(-i k z + i\omega t)$ by the singularity-free Cartesian vortex factor
  $$V(x, y) = \left[\frac{\sqrt{2}}{w_z}(x - i\,\mathrm{sgn}(\ell)\,y)\right]^{|\ell|} = \left(\frac{\sqrt{2}\,r}{w_z}\right)^{|\ell|} e^{-i\ell\phi},$$
  giving time-domain phase $\cos(\omega t - k z - \ell\phi)$ and positive laboratory-frame OAM $L_z = +\ell$ per photon when $\ell > 0$.
- **Exact vs. approximate paraxial longitudinal fields**: When `exact_paraxial=True` (default), $E_z = -\frac{i}{k}\nabla_\perp \cdot \mathbf{E}_\perp$ and $B_z = \frac{i}{\omega}(\partial_x E_y - \partial_y E_x)$ differentiate the full Laguerre polynomial, vortex factor, and wavefront curvature $R(z)$.

### Beam Class Normalization Difference

`GaussianBeamParameters` and `LaguerreGaussBeamParameters` differ by a factor of $\sqrt{2}$ in peak amplitude normalization:

| Beam Class | Peak Factor | Transverse Power Integral $\iint \|u(x,y)\|^2\,dA$ | Incident Pulse Energy $U_{\mathrm{inc}}$ |
| --- | --- | --- | --- |
| `LaguerreGaussBeamParameters` | $\sqrt{\dfrac{2\,p!}{(p+|\ell|)!}} \, E_0$ | $\pi w_0^2 E_0^2$ | $\dfrac{c}{8} w_0^2 E_0^2 \mathcal{T}_g$ |
| `GaussianBeamParameters` | $E_0$ | $\dfrac{\pi}{2} w_0^2 E_0^2$ | $\dfrac{c}{16} w_0^2 E_0^2 \mathcal{T}_g$ |
| `LaserParameters` (plane wave) | $E_0$ | Fluence $F_{\mathrm{inc}} = \dfrac{c}{8\pi} E_0^2 \mathcal{T}_g$ | — |

The LG normalization matches Hu et al., *Sci. Rep.* 8:7282 (2018), Eq. (3) (verified in `tests/test_beams.py::test_lg_exact_on_axis_field_hu_paper`).

### Temporal Pulse Envelopes (`pulse.py`)

For phase coordinate $\phi = (ct - s(z_{\mathrm{proj}} - z_{\mathrm{ref}}))/c$, the cycle-averaged energy integral $\mathcal{T}_g = \int_{-\infty}^{\infty} g(\phi)^2\,d\phi$ is:

| Pulse Class | Envelope $g(\phi)$ | Exact Integral $\mathcal{T}_g$ |
| --- | --- | --- |
| `GaussianPulseParameters` | $\exp\!\left(-\frac{(\phi-\phi_0)^2}{\tau_0^2}\right)$ | $\tau_0 \sqrt{\pi / 2}$ |
| `CosineSquaredPulseParameters` | $\cos^2\!\left(\frac{\pi(\phi-\phi_0)}{2\tau_0}\right)\,\mathbf{1}_{|\phi-\phi_0|\le \tau_0}$ | $\dfrac{3}{4}\tau_0$ |
| `PulseWithFlatPeakParameters` | Gaussian rise/fall ($\tau_0$) + flat peak ($N_{\mathrm{peak}}\tau_0$) | $\tau_0 \sqrt{\pi / 2} + N_{\mathrm{peak}}\tau_0$ |

Since the pulse propagates along $\hat{\mathbf{k}}_{\mathrm{in}}$, $U_{\mathrm{inc}}$ is invariant under oblique incidence rotation in [`compute_oblique_beam_fields`](../src/electrodynamics/beams.py#L377).

---

## 4. Scattered Fourier Fields & Parseval Energy Flux (`fields.py`)

[`compute_scattered_electric_and_magnetic_fields`](../src/electrodynamics/fields.py#L129) accumulates the one-sided positive-frequency Fourier transform with $\frac{1}{2\pi}$ normalization (and $|q| = 1$ implied for electrons):

$$\tilde{\mathbf{E}}(\mathbf{x}, \omega) = \frac{1}{2\pi} \int \mathbf{E}_{\mathrm{Gauss}}(\mathbf{x}, t) e^{+i\omega t}\,dt, \qquad \tilde{\mathbf{B}}_{\mathrm{Gauss}}(\mathbf{x}, \omega) = \frac{1}{2\pi} \int \mathbf{B}_{\mathrm{Gauss}}(\mathbf{x}, t) e^{+i\omega t}\,dt.$$

- **Gaussian-unit scattered magnetic field**: In the far field ($O(1/|\mathbf{R}|)$), $\tilde{\mathbf{B}}_{\mathrm{Gauss}} = \mathbf{n} \times \tilde{\mathbf{E}}$, so $|\tilde{\mathbf{B}}_{\mathrm{Gauss}}| = |\tilde{\mathbf{E}}|$ (unlike `beams.py`, where $|\mathbf{B}_{\mathrm{beam}}| = |\mathbf{E}|/c$).
- **Exact cancellation-free retarded phase**: Evaluates $t + |\mathbf{x} - \mathbf{r}(t)|/c$ via $\delta r = (-2(\mathbf{x}\cdot\mathbf{r}) + |\mathbf{r}|^2)/(|\mathbf{x}-\mathbf{r}| + |\mathbf{x}|)$ to avoid float64 subtraction loss at large detector distances.
- **Parseval spectral energy flux density**: Because $\mathbf{E}_{\mathrm{Gauss}}(t)$ is real ($\tilde{\mathbf{E}}(-\omega) = \tilde{\mathbf{E}}(\omega)^*$), integrating the Gaussian Poynting vector $\mathbf{S}(t) = \frac{c}{4\pi}\mathbf{E}_{\mathrm{Gauss}}(t)\times\mathbf{B}_{\mathrm{Gauss}}(t)$ over time yields:
  $$\int_{-\infty}^{\infty} \mathbf{S}(t)\,dt = c \int_0^\infty \mathrm{Re}\!\left[\tilde{\mathbf{E}}(\mathbf{x}, \omega) \times \tilde{\mathbf{B}}_{\mathrm{Gauss}}(\mathbf{x}, \omega)^*\right]\,d\omega.$$
  Thus the **spectral energy flux density** crossing a detector element with outward unit normal $\hat{\mathbf{n}}_{\mathrm{det}}$ is:
  $$\frac{d^2 U}{d\omega\, dA}(\mathbf{x}, \omega) = c\, \mathrm{Re}\!\left[\tilde{\mathbf{E}}(\mathbf{x}, \omega) \times \tilde{\mathbf{B}}_{\mathrm{Gauss}}(\mathbf{x}, \omega)^*\right] \cdot \hat{\mathbf{n}}_{\mathrm{det}} \;\approx\; c |\tilde{\mathbf{E}}(\mathbf{x}, \omega)|^2 (\hat{\mathbf{r}}\cdot\hat{\mathbf{n}}_{\mathrm{det}})\quad\text{(far field)}.$$

---

## 5. Radiated OAM Spectral Density & Sign Convention (`angular_momentum.py`)

[`compute_angular_momentum_derivative`](../src/electrodynamics/angular_momentum.py#L44) computes the laboratory-frame $z$-component of orbital angular momentum spectral density in Gaussian atomic units:

$$\frac{d^2 L_z}{d\omega\, dA} = \frac{c}{\omega} \sum_{i \in \{x,y,z\}} \mathrm{Re}\!\left[\frac{1}{i}\, \tilde{E}_i^* \left(x\partial_y - y\partial_x\right) \tilde{E}_i\right].$$

- **Spherical phase compensation**: Prior to taking Cartesian finite differences, the radially symmetric spherical carrier phase $\exp\left(i\frac{\omega}{c}(|\mathbf{r}_{\mathrm{det}}| - r_{\mathrm{center}})\right)$ is factored out. Because $(x\partial_y - y\partial_x)|\mathbf{r}_{\mathrm{det}}| = 0$ identically, this eliminates Cartesian finite-difference grid dispersion ($\sin(4\phi)$ aliasing).
- **Fourier vs. time-domain sign convention**: Because $\tilde{\mathbf{E}}(\omega)$ uses $e^{+i\omega t}$, an incident wave with time-domain phase $\cos(\omega t - \ell\phi)$ carries positive-frequency Fourier phase $e^{+i\ell\phi}$, yielding:
  $$\mathrm{Re}\!\left(\frac{d^2 L_z}{d\omega\, dA}\right) = +\frac{\ell}{\omega}\, c |\tilde{\mathbf{E}}|^2 = +\frac{\ell}{\omega}\, \frac{d^2 U}{d\omega\, dA}.$$
- **Lab-frame $L_z$ vs. beam-axis helicity**:
  - Always integrate over detector area with [`integrate_detector_oam_spectral_density`](../src/electrodynamics/angular_momentum.py#L127) (never raw `np.sum`).
  - [`compute_oam_per_photon(..., propagation_sign=1 | -1)`](../src/electrodynamics/angular_momentum.py#L135) returns $\langle \ell \rangle = \mathrm{propagation\_sign} \cdot \omega \frac{dL_z/d\omega}{dU/d\omega}$. Set `propagation_sign=1` for lab-frame $L_z$, or `-1` for beam-axis helicity of a reflected beam propagating along $-\hat{\mathbf{z}}$.

---

## 6. Coordinate Frames & Reflection Kinematics (`detector.py`, `fields.py`)

- **Default scattering geometry**: Laser beam propagates along $+\hat{\mathbf{z}}$ (or oblique $\hat{\mathbf{k}}_{\mathrm{in}} = (\sin\theta_{\mathrm{in}}, 0, \cos\theta_{\mathrm{in}})$ in the $x$-$z$ plane via [`compute_oblique_beam_fields`](../src/electrodynamics/beams.py#L377)); counter-propagating relativistic electrons initialize along $-\hat{\mathbf{z}}$ ([`generate_initial_particle_momenta_moving_towards_laser`](../src/electrodynamics/initial_conditions.py#L76)).
- **Reflected detector frame** ([`initialize_detector_positions_reflected`](../src/electrodynamics/detector.py#L78), [`project_fields_to_detector_frame`](../src/electrodynamics/detector.py#L118)):
  - Detector center lies at $R_{\mathrm{det}}\,\hat{\mathbf{k}}_{\mathrm{re}}$ where $\hat{\mathbf{k}}_{\mathrm{re}} = (\sin\theta_{\mathrm{re}}, 0, -\cos\theta_{\mathrm{re}})$.
  - Local orthonormal basis: $\hat{\mathbf{e}}_{x'} = (\cos\theta_{\mathrm{re}}, 0, \sin\theta_{\mathrm{re}})$, $\hat{\mathbf{e}}_{y'} = (0, 1, 0)$, $\hat{\mathbf{e}}_{z'} = (-\sin\theta_{\mathrm{re}}, 0, \cos\theta_{\mathrm{re}}) = -\hat{\mathbf{k}}_{\mathrm{re}}$.
  - At $\theta_{\mathrm{re}} = 0$, this reduces identically to a detector plane at $z = -R_{\mathrm{det}}$ facing the origin with outward normal $-\hat{\mathbf{z}}$.
- **Relativistic reflection kinematics** ([`compute_relativistic_reflection_parameters`](../src/electrodynamics/fields.py#L10)): Follows Chen et al., *Sci. Rep.* 12:12524 (2022) for exact $\theta'_{\mathrm{in}}$, $\theta_{\mathrm{re}}$, Doppler factor $D = \omega_1 / \omega_0$, and pulse-profile tilt angle $\varphi$.

---

## 7. Analytical Reference Benchmarks

1. **Thin-sheet relativistic flying-mirror reflectivity**: For a cold relativistic sheet of $N_e$ electrons over disk radius $R_{\mathrm{sheet}}$ (surface density $\sigma_e = N_e / (\pi R_{\mathrm{sheet}}^2)$) and weak laser amplitude $a_0 \ll 1$:
   $$\alpha = \frac{2\pi q^2 \sigma_e}{m_e c \gamma (1 + \beta\cos\theta_{\mathrm{in}}) \omega_0},$$
   $$R_{\mathrm{photon}} \approx \alpha^2 \;\; \left(\text{dense limit: }\frac{\alpha^2}{1+\alpha^2}\right),$$
   $$U_{\mathrm{rad}} = D\, R_{\mathrm{photon}}\, U_{\mathrm{inc}}.$$
   Energy partition per reflected photon: fraction $1/D$ originates from the incident laser pulse and $(D-1)/D$ from electron kinetic recoil. *(Validated against `runs/flying_mirror/2026-11-22_18-13-32`: $U_{\mathrm{rad, sim}} \approx 6.208\times 10^{-3}\,\mathrm{Ha}$ vs. $U_{\mathrm{rad, theory}} = 6.206\times 10^{-3}\,\mathrm{Ha}$, $0.02\%$ agreement with zero free parameters.)*
2. **Relativistic Larmor radiated power**:
   $$P_{\mathrm{Larmor}} = \frac{dU_{\mathrm{Larmor}}}{dt} = \frac{2 q^2}{3 c^3}\left(\left|\frac{d\mathbf{u}}{d\tau}\right|^2 - \left(\frac{du^0}{d\tau}\right)^2\right) = \frac{2 q^2}{3 c^3} \gamma^2 \left(\left|\frac{d\mathbf{u}}{dt}\right|^2 - \left(\frac{du^0}{dt}\right)^2\right).$$
   For $N_e = 1$, integrating detector energy over all angles and frequencies provides an exact conservation check against $U_{\mathrm{Larmor}}$.
