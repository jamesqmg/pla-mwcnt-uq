"""FFT-based homogenization (Moulinec-Suquet basic scheme) for Tier-3 FE-RVE.

This is the sanctioned reopen path for the wavy-absolute-E1 question
(fe-rve-asset-inventory.md sec 14 REVISIT-IF): the voxel-FE staircase at the
330x CNT/PLA stiffness contrast defeats mesh convergence, but a spectral
(FFT) homogenizer solves the same periodic microstructure with no mesh at all,
and Willot's rotated-scheme discrete Green operator specifically suppresses the
high-frequency ringing that continuous-operator FFT and voxel-FE both suffer at
sharp high-contrast interfaces.

Method (Moulinec & Suquet 1994/1998): the periodic Lippmann-Schwinger fixed
point  eps = E_bar - Gamma^0 * tau,  tau = sigma - C0:eps,  is iterated in
Fourier space. C0 is a homogeneous isotropic reference; Gamma^0 is its Green
operator. Only per-voxel isotropic Lame fields (lam, mu) are stored -- every
phase here is isotropic -- so stress is sigma = lam*tr(eps)*I + 2mu*eps.

Green operator (classical, Moulinec-Suquet): with a_i = tau_hat_ik xi_k and
b = xi_i a_i,
    (Gamma^0 : tau_hat)_ij = (xi_i a_j + xi_j a_i)/(2 mu0 |xi|^2)
                             - C1 xi_i xi_j b / |xi|^4,   C1 = (lam0+mu0)/(mu0(lam0+2mu0)).
This form is degree-0 homogeneous in xi, so the frequency may be rescaled by any
common (complex) scalar. Willot's rotated scheme (Willot 2015, C. R. Mecanique,
eq. 36) replaces xi by k^R = (i/4) tan(q_i/2) prod_j(1+e^{i q_j}); its common
complex prefactor cancels here, leaving the real effective frequency
    xi_i^W = sin(pi m_i/N_i) * prod_{j!=i} cos(pi m_j/N_j).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree

from src.uq.rve_geometry import closest_points_segment_segment
from src.uq.rve_wavy_geometry import WavyFiberGeometry, fiber_polyline


def lame_from_E_nu(E: float, nu: float) -> tuple[float, float]:
    """Lame parameters (lambda, mu) from Young's modulus and Poisson's ratio."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def _frequency_grids(shape: tuple[int, int, int], scheme: str):
    """Real effective frequency component grids (kx, ky, kz), each of `shape`.

    "classical" uses integer DFT wavenumbers (degree-0 homogeneity makes the
    physical 2*pi/L scaling irrelevant). "willot" uses the rotated-scheme
    effective real frequency sin(pi m_i/N) prod_{j!=i} cos(pi m_j/N).
    """
    ms = [np.fft.fftfreq(n) * n for n in shape]  # integer indices per axis
    if scheme == "classical":
        kx, ky, kz = np.meshgrid(*ms, indexing="ij")
        return kx, ky, kz
    if scheme == "willot":
        angles = [np.pi * m / n for m, n in zip(ms, shape)]
        s = [np.sin(a) for a in angles]
        c = [np.cos(a) for a in angles]
        sx, sy, sz = np.meshgrid(*s, indexing="ij")
        cx, cy, cz = np.meshgrid(*c, indexing="ij")
        return sx * cy * cz, cx * sy * cz, cx * cy * sz
    raise ValueError(f"unknown scheme {scheme!r}; use 'classical' or 'willot'")


def _apply_green(tau: np.ndarray, k: np.ndarray, lam0: float, mu0: float) -> np.ndarray:
    """Real-space fluctuation strain -Gamma^0 * tau (mean-zero). tau, result: (3,3,*).

    Invalid modes (|xi|^2 ~ 0: the mean and the rotated-scheme spurious corner
    nulls) are zeroed, so the returned field has no mean and no undefined-mode
    content; the macro strain is imposed separately by the caller.
    """
    that = np.fft.fftn(tau, axes=(-3, -2, -1))  # (3,3,*) complex
    k2 = np.einsum("i...,i...->...", k, k)
    invalid = k2 < 1e-12
    k2_safe = np.where(invalid, 1.0, k2)

    a = np.einsum("ik...,k...->i...", that, k)  # (3,*)
    b = np.einsum("i...,i...->...", k, a)       # (*,)
    C1 = (lam0 + mu0) / (mu0 * (lam0 + 2.0 * mu0))

    ki_aj = np.einsum("i...,j...->ij...", k, a)      # (3,3,*)
    sym = ki_aj + np.swapaxes(ki_aj, 0, 1)
    kikj = np.einsum("i...,j...->ij...", k, k)       # (3,3,*)
    gamma_tau = sym / (2.0 * mu0 * k2_safe) - C1 * kikj * b / (k2_safe**2)

    eps_hat = -gamma_tau
    eps_hat[..., invalid] = 0.0
    return np.real(np.fft.ifftn(eps_hat, axes=(-3, -2, -1)))


def homogenize_fft(
    lam: np.ndarray,
    mu: np.ndarray,
    macro_strain: np.ndarray,
    domain_size: float,
    lam0: float | None = None,
    mu0: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    scheme: str = "classical",
    return_field: bool = False,
) -> tuple[np.ndarray, dict]:
    """Homogenized macro-stress (3x3) for an imposed macro-strain on the periodic
    isotropic Lame fields lam, mu (each (Nx,Ny,Nz)). Returns (macro_stress, info);
    with return_field=True, info["stress_field"] holds the (3,3,Nx,Ny,Nz) local stress."""
    shape = lam.shape
    if lam0 is None:
        lam0 = 0.5 * (float(lam.min()) + float(lam.max()))
    if mu0 is None:
        mu0 = 0.5 * (float(mu.min()) + float(mu.max()))

    kx, ky, kz = _frequency_grids(shape, scheme)
    k = np.stack([kx, ky, kz])  # (3,*)

    E = np.asarray(macro_strain, dtype=float)
    eps = np.empty((3, 3) + shape)
    for i in range(3):
        for j in range(3):
            eps[i, j] = E[i, j]

    error = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
        sigma = 2.0 * mu * eps
        for d in range(3):
            sigma[d, d] += lam * tr

        # Equilibrium convergence: ||xi . sigma_hat|| / ||sigma_hat(0)||.
        shat = np.fft.fftn(sigma, axes=(-3, -2, -1))
        div = np.einsum("j...,ij...->i...", k, shat)
        s0 = shat[:, :, 0, 0, 0]
        den = np.sqrt(np.sum(np.abs(s0) ** 2))
        error = float(np.sqrt(np.sum(np.abs(div) ** 2)) / den) if den > 0 else 0.0
        if not np.isfinite(error):
            raise FloatingPointError(
                f"FFT iteration diverged (non-finite residual) at iter {it}. The reference "
                f"(lam0={lam0:.4g}, mu0={mu0:.4g}) is inadmissibly soft for the basic scheme "
                f"at this contrast; use a stiffer reference (mu0 >= mu_max/2)."
            )
        if error < tol:
            break

        # Polarization tau = sigma - C0:eps, then eps = E_bar - Gamma^0 * tau.
        c0eps = 2.0 * mu0 * eps
        for d in range(3):
            c0eps[d, d] += lam0 * tr
        tau = sigma - c0eps
        fluct = _apply_green(tau, k, lam0, mu0)
        for i in range(3):
            for j in range(3):
                eps[i, j] = E[i, j] + fluct[i, j]

    macro_stress = np.zeros((3, 3))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sigma = 2.0 * mu * eps
    for d in range(3):
        sigma[d, d] += lam * tr
    for i in range(3):
        for j in range(3):
            macro_stress[i, j] = float(sigma[i, j].mean())
    info = {"iterations": it, "error": error}
    if return_field:
        info["stress_field"] = sigma  # (3,3,Nx,Ny,Nz) local stress
    return macro_stress, info


def von_mises_field(sigma: np.ndarray) -> np.ndarray:
    """Per-voxel von Mises stress from a (3,3,Nx,Ny,Nz) local stress field."""
    s = sigma
    dev = ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
    shear = s[0, 1] ** 2 + s[1, 2] ** 2 + s[0, 2] ** 2
    return np.sqrt(0.5 * dev + 3.0 * shear)


_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _voigt_unit_strains() -> list[np.ndarray]:
    """Six symmetric unit macro-strains, Voigt order [xx,yy,zz,yz,xz,xy]; shear
    cases set the off-diagonal pair to 0.5 so engineering gamma = 1."""
    cases = []
    for a in range(3):
        e = np.zeros((3, 3))
        e[a, a] = 1.0
        cases.append(e)
    for i, j in ((1, 2), (0, 2), (0, 1)):
        e = np.zeros((3, 3))
        e[i, j] = e[j, i] = 0.5
        cases.append(e)
    return cases


def _stress_to_voigt(sigma: np.ndarray) -> np.ndarray:
    return np.array([sigma[i, j] for i, j in _VOIGT_PAIRS])


def full_stiffness_tensor_fft(
    lam: np.ndarray,
    mu: np.ndarray,
    domain_size: float,
    tol: float = 1e-6,
    max_iter: int = 1000,
    scheme: str = "classical",
    lam0: float | None = None,
    mu0: float | None = None,
) -> np.ndarray:
    """Homogenized 6x6 Voigt stiffness C (same convention as rve_stiffness)."""
    C = np.zeros((6, 6))
    for k, macro in enumerate(_voigt_unit_strains()):
        sigma, _ = homogenize_fft(
            lam, mu, macro, domain_size, lam0=lam0, mu0=mu0,
            tol=tol, max_iter=max_iter, scheme=scheme,
        )
        C[:, k] = _stress_to_voigt(sigma)
    return C


def voxelize_straight_fibers(
    n: int,
    domain_size: float,
    radius: float,
    axis: int,
    matrix: tuple[float, float],
    fiber: tuple[float, float],
    n_fibers_per_side: int = 1,
    supersample: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-voxel (lam, mu) fields and achieved Vf for aligned straight fibers on
    a regular (Nx=Ny=Nz=n) grid. Fibers run along `axis`; membership is the 2D
    transverse periodic min-image distance to the nearest fiber center. Each
    voxel is `supersample`^2 sub-sampled in the transverse plane to soften the
    in-plane staircase (partial-volume fiber fraction), then Lame params are the
    linear phi-mix used by composite_lame_fields."""
    transverse = [a for a in range(3) if a != axis]
    step = domain_size / n_fibers_per_side
    centers = np.array([
        [(i + 0.5) * step, (j + 0.5) * step]
        for i in range(n_fibers_per_side) for j in range(n_fibers_per_side)
    ])  # (n_fibers, 2), in [0, domain)

    # Sub-sample offsets within a voxel edge h, in the two transverse axes.
    h = domain_size / n
    offs = (np.arange(supersample) + 0.5) / supersample - 0.5  # centered in [-0.5,0.5)
    sub = np.array(list(itertools.product(offs, offs))) * h  # (supersample^2, 2)

    # Voxel-center transverse coordinates.
    coord = (np.arange(n) + 0.5) * h
    grid = np.stack(np.meshgrid(coord, coord, coord, indexing="ij"), axis=0)  # (3,n,n,n)
    tvals = [grid[transverse[0]], grid[transverse[1]]]

    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=2))) * domain_size
    image_centers = (centers[:, None, :] + shifts[None, :, :]).reshape(-1, 2)  # (9F,2)

    phi = np.zeros((n, n, n))
    for du, dv in sub:
        pu = (tvals[0] + du)[..., None]  # (n,n,n,1)
        pv = (tvals[1] + dv)[..., None]
        d2 = (pu - image_centers[:, 0]) ** 2 + (pv - image_centers[:, 1]) ** 2
        inside = (d2.min(axis=-1) <= radius**2)
        phi += inside
    phi /= sub.shape[0]

    lam_m, mu_m = lame_from_E_nu(*matrix)
    lam_f, mu_f = lame_from_E_nu(*fiber)
    lam = (1.0 - phi) * lam_m + phi * lam_f
    mu = (1.0 - phi) * mu_m + phi * mu_f
    return lam, mu, float(phi.mean())


def voxelize_wavy_fibers(
    geometry: WavyFiberGeometry,
    n: int,
    matrix: tuple[float, float],
    fiber: tuple[float, float],
    supersample: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-voxel (lam, mu) fields and achieved Vf for a wavy-fiber RVE on a
    structured n^3 grid (composite partial-volume, matching the voxel-FE
    classify_wavy_fiber_fractions convention so the FFT-vs-FE comparison is
    apples-to-apples). Voxels well inside/outside a fiber are 1/0 from the
    vectorized nearest-segment banding; only straddling voxels are
    supersample^3 sub-sampled and exactly segment-distance tested."""
    domain = geometry.domain_size
    radius = geometry.radius
    axis = geometry.axis
    transverse = [a for a in range(3) if a != axis]
    h = domain / n

    seg_a, seg_b = [], []
    for fib in geometry.fibers:
        poly = fiber_polyline(fib, domain, geometry.n_samples)
        for su, sv in itertools.product((-1, 0, 1), repeat=2):
            shifted = poly.copy()
            shifted[:, transverse[0]] += su * domain
            shifted[:, transverse[1]] += sv * domain
            seg_a.append(shifted[:-1])
            seg_b.append(shifted[1:])
    A = np.concatenate(seg_a, axis=0)
    B = np.concatenate(seg_b, axis=0)
    midpoints = 0.5 * (A + B)
    half_len_max = 0.5 * float(np.linalg.norm(B - A, axis=1).max())
    tree = cKDTree(midpoints)

    def _min_dist(point: np.ndarray) -> float:
        best = np.inf
        for ci in tree.query_ball_point(point, radius + half_len_max):
            d = closest_points_segment_segment(point, point, A[ci], B[ci])
            if d < best:
                best = d
        return best

    coord = (np.arange(n) + 0.5) * h
    centers = np.stack(np.meshgrid(coord, coord, coord, indexing="ij"), axis=-1)  # (n,n,n,3)
    flat = centers.reshape(-1, 3)
    d_mid, _ = tree.query(flat, k=1)
    band = h * np.sqrt(3.0) * 0.5  # half-diagonal of a voxel from its center
    definitely_in = d_mid + half_len_max < radius - band
    definitely_out = d_mid - half_len_max > radius + band

    phi_flat = np.zeros(flat.shape[0])
    phi_flat[definitely_in] = 1.0
    offs = (np.arange(supersample) + 0.5) / supersample - 0.5
    sub = np.array(list(itertools.product(offs, offs, offs))) * h  # (s^3, 3)
    straddling = np.nonzero(~definitely_in & ~definitely_out)[0]
    for e in straddling:
        p0 = flat[e]
        inside = sum(1 for s in sub if _min_dist(p0 + s) <= radius)
        phi_flat[e] = inside / sub.shape[0]

    phi = phi_flat.reshape(n, n, n)
    lam_m, mu_m = lame_from_E_nu(*matrix)
    lam_f, mu_f = lame_from_E_nu(*fiber)
    lam = (1.0 - phi) * lam_m + phi * lam_f
    mu = (1.0 - phi) * mu_m + phi * mu_f
    return lam, mu, float(phi.mean())
