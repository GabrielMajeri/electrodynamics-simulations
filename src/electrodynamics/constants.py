### Physical constants (in atomic units)

SPEED_OF_LIGHT: float = 137.036
"Velocity of light in vacuum, in atomic units."

ELECTRON_MASS: float = 1
"Mass of the electron, in atomic units."

ELECTRON_CHARGE: float = -1
"Charge of the electron, in atomic units."

HARTREE_TO_EV: float = 27.211386245981
"Conversion factor from Hartree atomic energy units to electronvolts (eV)."

ELECTRON_REST_ENERGY_MEV: float = 0.5109989
"Electron rest mass energy $E = m_e * c^2$, in MeV."

DEFAULT_LASER_FREQUENCY: float = 0.057
"Default Ti:Sapphire laser angular frequency (~800 nm wavelength), in atomic units."


### SI reference conversion factors from 2022 CODATA release
# Source: https://codata.org/initiatives/data-science-and-stewardship/fundamental-physical-constants/

BOHR_RADIUS_M: float = 5.29177210544e-11
"Bohr radius (1 atomic unit of length), in meters."

ATOMIC_TIME_S: float = 2.4188843265864e-17
"Atomic unit of time (hbar / E_h), in seconds."

HARTREE_TO_JOULES: float = 4.3597447222060e-18
"Hartree energy (1 atomic unit of energy), in joules."

ELEMENTARY_CHARGE_C: float = 1.602176634e-19
"Elementary charge (1 atomic unit of charge), in coulombs."


def au_length_to_meters(length_au: float) -> float:
    "Converts a length from atomic units (Bohr radii) to meters."
    return float(length_au * BOHR_RADIUS_M)


def au_time_to_seconds(time_au: float) -> float:
    "Converts a time interval from atomic units to seconds."
    return float(time_au * ATOMIC_TIME_S)


def au_energy_to_joules(energy_au: float) -> float:
    "Converts an energy from Hartree atomic units to joules."
    return float(energy_au * HARTREE_TO_JOULES)


def au_energy_to_ev(energy_au: float) -> float:
    "Converts an energy from Hartree atomic units to electronvolts (eV)."
    return float(energy_au * HARTREE_TO_EV)
