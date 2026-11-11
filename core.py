"""
core.py — hydrogen wavefunction calculation and grid construction
=====================================

This module handles physics and data preparation only; it performs no plotting, so it can be reused by the CLI, GUI, and tests.

Improvements over the original Time_Hydrogen/core.py
------------------------------------
1. Classical turning points use the correct formula ``r = n^2 a0 [1 +/- sqrt(1 - l(l+1)/n^2)]``.
2. If ``assoc_laguerre`` overflows, the radial function falls back to a log-space recurrence, which is more stable at high n.
3. Supports complex and real spherical harmonics (cos / sin), consistent with the New ICS orbital model.
4. Adds ``build_plane_data`` / ``build_volume_data`` to unify 2D cross-section and
   3D volume data construction for the six plot modules.
"""

from __future__ import annotations

import numpy as np
import scipy.constants as const
import scipy.special as spe
from scipy.special import sph_harm_y


class HydrogenWavefunction:
    """Hydrogen wavefunction calculator (associated Laguerre polynomials + log-space stabilization)."""

    def __init__(self):
        self.a0 = const.value("Bohr radius")  # Bohr radius (m)

    # ------------------------------------------------------------------
    # Classical quantities
    # ------------------------------------------------------------------
    def classical_turning_points(self, n, l):
        """
        Classical turning points r_min / r_max.

        Correct formula (effective centrifugal potential l(l+1))::

            r_min/max = n² a₀ [1 ± sqrt(1 - l(l+1)/n²)]
        """
        eps_sq = l * (l + 1) / (n * n)
        if eps_sq >= 1.0:
            eps_sq = 1.0 - 1e-15
        root = np.sqrt(1.0 - eps_sq)
        r_min = n * n * self.a0 * (1.0 - root)
        r_max = n * n * self.a0 * (1.0 + root)
        return r_min, r_max

    def scale_for(self, n, l, factor=1.8):
        """Half-width of a 2D cross-section (m). Defaults to 1.8 x r_max, as in the original project."""
        _, r_max = self.classical_turning_points(n, l)
        return factor * r_max

    def log_factorial(self, n):
        """ln(n!), computed with the gamma function."""
        return spe.loggamma(n + 1)

    # ------------------------------------------------------------------
    # Radial wavefunction
    # ------------------------------------------------------------------
    @staticmethod
    def _log_assoc_laguerre(rho, k, alpha):
        """
        Evaluate the magnitude and sign of the associated Laguerre polynomial L_k^(alpha)(rho) in log space.

        Returns (sign, log_abs). Used only when ``spe.assoc_laguerre`` overflows.
        """
        rho = np.asarray(rho, dtype=float)
        if k == 0:
            return np.ones_like(rho), np.zeros_like(rho)

        s_prev = np.ones_like(rho)
        l_prev = np.zeros_like(rho)

        v1 = 1.0 + alpha - rho
        s_curr = np.sign(v1)
        l_curr = np.log(np.abs(v1) + 1e-300)
        if k == 1:
            return s_curr, l_curr

        for m in range(1, k):
            coef1 = 2.0 * m + alpha + 1.0 - rho
            coef2 = float(m) + alpha

            log_a = np.log(np.abs(coef1) + 1e-300) + l_curr
            sign_a = np.sign(coef1) * s_curr

            log_b = np.log(coef2 + 1e-300) + l_prev
            sign_b = s_prev

            big_a = log_a >= log_b
            log_big = np.where(big_a, log_a, log_b)
            log_small = np.where(big_a, log_b, log_a)

            eff_sign_big = np.where(big_a, sign_a, -sign_b)
            eff_sign_small = np.where(big_a, -sign_b, sign_a)

            ratio = np.exp(log_small - log_big)
            factor = eff_sign_big + eff_sign_small * ratio

            s_next = np.sign(factor)
            l_next = log_big + np.log(np.abs(factor) + 1e-300) - np.log(m + 1.0)

            s_prev, l_prev = s_curr, l_curr
            s_curr, l_curr = s_next, l_next

        return s_curr, l_curr

    def R_nl(self, r, n, l):
        """Radial wavefunction R_nl(r) (m^-3/2). Supports scalars and arrays of any shape."""
        r_arr = np.asarray(r, dtype=float)
        shape = r_arr.shape
        r_flat = r_arr.ravel()
        r_safe = np.where(r_flat < 1e-100 * self.a0, 1e-100 * self.a0, r_flat)

        log_coeff = 0.5 * (
            3.0 * np.log(2.0 / (self.a0 * n))
            + self.log_factorial(n - l - 1)
            - np.log(2.0 * n)
            - self.log_factorial(n + l)
        )

        rho = 2.0 * r_safe / (self.a0 * n)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rho = np.log(np.where(rho > 0.0, rho, 1.0))
        log_exp = -rho / 2.0 + l * log_rho

        k = n - l - 1
        alpha = 2 * l + 1

        with np.errstate(over="ignore", invalid="ignore"):
            lag = spe.assoc_laguerre(rho, k, alpha)
        bad = ~np.isfinite(lag)

        if not np.any(bad):
            sign_l = np.sign(lag)
            log_l = np.log(np.abs(lag) + 1e-300)
        else:
            sign_l = np.sign(np.where(bad, 1.0, lag))
            log_l = np.log(np.abs(np.where(bad, 1.0, lag)) + 1e-300)
            s_bad, l_bad = self._log_assoc_laguerre(rho[bad], k, alpha)
            sign_l = sign_l.copy()
            log_l = log_l.copy()
            sign_l[bad] = s_bad
            log_l[bad] = l_bad

        log_result = np.clip(log_coeff + log_exp + log_l, -700.0, 700.0)
        result = sign_l * np.exp(log_result)

        origin_mask = r_flat < 1e-100 * self.a0
        if l != 0:
            result = np.where(origin_mask, 0.0, result)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result.reshape(shape)

    # ------------------------------------------------------------------
    # Angular wavefunction
    # ------------------------------------------------------------------
    def sph_harm_compat(self, m, l, phi, theta):
        """Version-compatible spherical-harmonic call: sph_harm_y(l, m, theta, phi)."""
        return sph_harm_y(l, m, theta, phi)

    def angular(
        self,
        l,
        m,
        phi,
        theta,
        orbital_type="complex",
        real_component="cos",
    ):
        """
        Angular wavefunction.

        orbital_type='complex' : complex spherical harmonic Y_l^m.
        orbital_type='real'    : real spherical harmonic; for m>0, real_component selects
                                 'cos' (e.g. p_x) or 'sin' (e.g. p_y).
        """
        if orbital_type == "complex":
            return self.sph_harm_compat(m, l, phi, theta)

        m_abs = abs(m)
        if m_abs == 0:
            return np.real(self.sph_harm_compat(0, l, phi, theta))

        yp = self.sph_harm_compat(m_abs, l, phi, theta)
        yn = self.sph_harm_compat(-m_abs, l, phi, theta)
        sign = -1.0 if (m_abs % 2) else 1.0

        if real_component == "cos":
            result = (yn + sign * yp) / np.sqrt(2.0)
        else:
            result = 1j * (yn - sign * yp) / np.sqrt(2.0)

        # Real harmonics should be real; remove floating-point residuals and sin(m*pi) noise
        result = np.real(result)
        result = np.where(np.abs(result) < 1e-10, 0.0, result)
        return result

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------
    def create_extended_grid(self, n, l, grid_size=400, factor=1.8):
        """Create an XZ-plane (Y=0) grid, kept for backward compatibility."""
        scale = self.scale_for(n, l, factor=factor)
        x = np.linspace(-scale, scale, grid_size)
        z = np.linspace(-scale, scale, grid_size)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        return X, Y, Z, scale


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------
def cart2sph(x, y, z):
    """Cartesian coordinates -> spherical coordinates (r, theta, phi)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arccos(np.clip(z / np.where(r > 0, r, 1), -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi


# ---------------------------------------------------------------------------
# Unified data construction
# ---------------------------------------------------------------------------
_PLANE_META = {
    "xy": {
        "name": "XY",
        "note": "z=0",
        "xlabel": "X (m)",
        "ylabel": "Y (m)",
        "xlabel_dyn": "x (m)",
        "ylabel_dyn": "y (m)",
    },
    "xz": {
        "name": "XZ",
        "note": "y=0",
        "xlabel": "X (m)",
        "ylabel": "Z (m)",
        "xlabel_dyn": "x (m)",
        "ylabel_dyn": "z (m)",
    },
    "yz": {
        "name": "YZ",
        "note": "x=0",
        "xlabel": "Y (m)",
        "ylabel": "Z (m)",
        "xlabel_dyn": "y (m)",
        "ylabel_dyn": "z (m)",
    },
}


def plane_meta(plane):
    """Return plane metadata (case-insensitive; invalid values fall back to xz)."""
    return _PLANE_META.get(str(plane).lower(), _PLANE_META["xz"])


# A plane is treated as nodal when its largest |psi| is below this fraction
# of a coarse three-dimensional reference amplitude.  Numerical residual on
# an exact nodal plane is ~1e-16 of the 3D maximum, so 1e-10 is safely above
# round-off while still far below any physically meaningful plane amplitude.
NODAL_RTOL = 1e-10
_REFERENCE_CACHE = {}


def _reference_amplitude(
    calc, n, l, m, scale, orbital_type, real_component, resolution=41
):
    """
    Coarse 3D reference max|psi| used to detect nodal planes.

    The reference is deliberately coarse: it is only used to decide whether
    a plane is numerically zero, not to normalize the plane.  The result is
    cached because build_plane_data may be called repeatedly for the same
    state (static/dynamic, several plots).
    """
    key = (
        int(n), int(l), int(m), round(float(scale), 15),
        str(orbital_type), str(real_component), int(resolution),
    )
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]

    x = np.linspace(-scale, scale, resolution)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    mask = r <= scale
    if not np.any(mask):
        value = 0.0
    else:
        r_in = r[mask]
        theta = np.arccos(np.clip(Z[mask] / (r_in + 1e-100), -1.0, 1.0))
        phi = np.arctan2(Y[mask], X[mask])
        radial = calc.R_nl(r_in, n, l)
        angular = calc.angular(
            l, m, phi, theta,
            orbital_type=orbital_type, real_component=real_component,
        )
        psi = np.nan_to_num(
            radial * angular, nan=0.0, posinf=0.0, neginf=0.0,
        )
        value = float(np.max(np.abs(psi))) if psi.size else 0.0

    _REFERENCE_CACHE[key] = value
    if len(_REFERENCE_CACHE) > 128:
        _REFERENCE_CACHE.clear()
    return value


def build_plane_data(
    calc,
    n,
    l,
    m,
    plane="xz",
    scale=None,
    grid_size=400,
    orbital_type="complex",
    real_component="cos",
):
    """
    Build the complex wavefunction and probability density on a selected plane.

    Returned dictionary fields
    ------------
    C1, C2 : 2D coordinate grids (corresponding to xlabel / ylabel)
    x, y, z : 2D Cartesian coordinates
    r       : 2D radius (m)
    psi     : complex wavefunction, normalized by max|psi|; zero on a nodal plane
    density : |psi|^2, normalized to max=1; zero on a nodal plane
    nodal_plane : bool, whether this plane is a nodal plane
    nodal_note  : nodal-plane message (empty string for a non-nodal plane)
    plane_max_raw, global_ref, relative_amplitude : diagnostics
    plane   : 'xy' / 'xz' / 'yz'
    plane_name, plane_note, xlabel, ylabel, xlabel_dyn, ylabel_dyn

    Notes
    -----
    Nodal-plane detection uses max|psi| on a coarse 3D grid as a reference instead of
    normalizing on the plane itself; otherwise floating-point residuals (~1e-16) would be amplified into spurious patterns.
    """
    meta = plane_meta(plane)
    plane = str(plane).lower()
    if plane not in ("xy", "xz", "yz"):
        plane = "xz"

    if scale is None:
        scale = calc.scale_for(n, l)

    coords = np.linspace(-scale, scale, grid_size)
    C1, C2 = np.meshgrid(coords, coords)

    if plane == "xy":
        x, y, z = C1, C2, np.zeros_like(C1)
    elif plane == "yz":
        y, z, x = C1, C2, np.zeros_like(C1)
    else:  # xz
        x, z, y = C1, C2, np.zeros_like(C1)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(y, x)

    radial = calc.R_nl(r.ravel(), n, l)
    angular = calc.angular(
        l, m, phi.ravel(), theta.ravel(),
        orbital_type=orbital_type, real_component=real_component,
    )
    psi = (radial * angular).reshape(C1.shape)
    psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)

    plane_max = float(np.max(np.abs(psi))) if psi.size else 0.0
    global_ref = _reference_amplitude(
        calc, n, l, m, scale, orbital_type, real_component,
    )
    if global_ref > 0:
        relative = plane_max / global_ref
    else:
        relative = 0.0

    nodal_plane = (global_ref <= 0) or (plane_max <= global_ref * NODAL_RTOL)
    if nodal_plane:
        psi = np.zeros_like(psi, dtype=complex)
        density = np.zeros_like(psi, dtype=float)
        nodal_note = "Nodal plane: |ψ| = 0 on this section"
    else:
        psi = psi / plane_max
        density = np.abs(psi) ** 2
        nodal_note = ""

    return {
        "C1": C1,
        "C2": C2,
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "psi": psi,
        "density": density,
        "scale": scale,
        "grid_size": grid_size,
        "plane": plane,
        "plane_name": meta["name"],
        "plane_note": meta["note"],
        "nodal_plane": nodal_plane,
        "nodal_note": nodal_note,
        "plane_max_raw": plane_max,
        "global_ref": global_ref,
        "relative_amplitude": relative,
        "xlabel": meta["xlabel"],
        "ylabel": meta["ylabel"],
        "xlabel_dyn": meta["xlabel_dyn"],
        "ylabel_dyn": meta["ylabel_dyn"],
    }


def shell_from_density(r, density, a0=None):
    """
    Shell density 4*pi*r^2*|psi|^2, normalized to a maximum of 1.

    ``a0`` is used only to avoid division by zero or negative zero at r=0; it may be omitted.
    """
    r_arr = np.asarray(r, dtype=float)
    dens_arr = np.asarray(density, dtype=float)
    floor = 1e-100 * (a0 if a0 else 1.0)
    r_safe = np.where(r_arr < floor, floor, r_arr)
    shell = 4.0 * np.pi * r_safe**2 * dens_arr
    max_shell = float(np.nanmax(shell)) if shell.size else 0.0
    if max_shell > 0:
        shell = shell / max_shell
    return shell


def build_volume_data(
    calc,
    n,
    l,
    m,
    scale,
    resolution=50,
    orbital_type="complex",
    real_component="cos",
):
    """
    Build the complex wavefunction on a 3D cubic grid.

    Only points with r <= scale are computed; psi=0 outside the sphere.
    Returns X, Y, Z, r, mask, psi, density, shell.
    Both density and shell are normalized to max=1 inside the sphere.
    """
    x = np.linspace(-scale, scale, resolution)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    mask = r <= scale

    psi = np.zeros(X.shape, dtype=complex)
    r_in = r[mask]
    theta = np.arccos(np.clip(Z[mask] / (r_in + 1e-100), -1.0, 1.0))
    phi = np.arctan2(Y[mask], X[mask])

    radial = calc.R_nl(r_in, n, l)
    angular = calc.angular(
        l, m, phi, theta,
        orbital_type=orbital_type, real_component=real_component,
    )
    psi_in = radial * angular
    max_abs = float(np.max(np.abs(psi_in))) if psi_in.size else 0.0
    if max_abs > 0:
        psi_in = psi_in / max_abs
    psi[mask] = psi_in

    density = np.abs(psi) ** 2
    shell = shell_from_density(r, density, a0=calc.a0)
    return X, Y, Z, r, mask, psi, density, shell


# ---------------------------------------------------------------------------
# Point-cloud cuts (for the standalone point-cloud tool / future extensions)
# ---------------------------------------------------------------------------
def get_cut_mask(points, mode, which_parts=None):
    """
    3D point-cloud cut mask.

    mode: 'none' | 'half_x' | 'half_y' | 'half_z' | 'quadrant' |
          'octant' | 'hexadecant'
    """
    points = np.asarray(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if mode == "none":
        return np.ones(len(points), dtype=bool)
    if mode == "half_x":
        return x >= 0
    if mode == "half_y":
        return y >= 0
    if mode == "half_z":
        return z >= 0
    if mode == "quadrant":
        quadrant = np.zeros(len(points), dtype=int)
        quadrant[(x >= 0) & (y >= 0)] = 0
        quadrant[(x < 0) & (y >= 0)] = 1
        quadrant[(x < 0) & (y < 0)] = 2
        quadrant[(x >= 0) & (y < 0)] = 3
        if which_parts is None:
            which_parts = [0, 1, 2, 3]
        return np.isin(quadrant, which_parts)
    if mode == "octant":
        octant = (
            ((x >= 0).astype(int) << 2)
            + ((y >= 0).astype(int) << 1)
            + (z >= 0).astype(int)
        )
        if which_parts is None:
            which_parts = list(range(8))
        return np.isin(octant, which_parts)
    if mode == "hexadecant":
        phi = np.arctan2(y, x)
        norms = np.linalg.norm(points, axis=1)
        theta = np.arccos(np.clip(z / (norms + 1e-100), -1.0, 1.0))
        phi_bin = ((phi + np.pi) // (np.pi / 2)).astype(int) % 4
        theta_bin = np.clip((theta // (np.pi / 4)).astype(int), 0, 3)
        hex_index = phi_bin * 4 + theta_bin
        if which_parts is None:
            which_parts = list(range(16))
        return np.isin(hex_index, which_parts)
    return np.ones(len(points), dtype=bool)
