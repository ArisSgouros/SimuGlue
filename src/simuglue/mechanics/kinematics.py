import numpy as np
from simuglue.mechanics.linalg import _sqrtm_spd, _expm_sym, _logm_spd

def strain_from_F(F: np.ndarray, measure: str) -> np.ndarray:
    """
    Compute strain measure from deformation gradient F.

    measure:
      - 'engineering'    : eps = 0.5 * ((F - I) + (F - I).T)
      - 'green-lagrange' : E   = 0.5 * (F.T @ F - I)
      - 'hencky'         : E   = log U,  U = sqrtm(F.T @ F)
    """
    F = np.asarray(F, float)
    I = np.eye(3)

    if measure == "engineering":
        return 0.5 * ((F - I) + (F - I).T)

    elif measure == "green-lagrange":
        C = F.T @ F
        return 0.5 * (C - I)

    elif measure == "hencky":
        C = F.T @ F
        U = _sqrtm_spd(C)
        return _logm_spd(U)

    else:
        raise ValueError(f"Unknown measure: {measure}")


def _right_stretch_from_strain(E: np.ndarray, measure: str) -> np.ndarray:
    """
    Internal: recover right stretch U from symmetric strain E.

    Assumptions:
      - 'engineering'    : small-strain approx, U ≈ I + E
      - 'green-lagrange' : C = I + 2E must be SPD, U = sqrtm(C)
      - 'hencky'         : E = log U, so U = expm(E) must be SPD
    """
    E = 0.5 * (E + E.T)
    I = np.eye(3)

    if measure == "engineering":
        return I + E

    elif measure == "green-lagrange":
        C = I + 2.0 * E
        # SPD check
        w = np.linalg.eigvalsh(C)
        if np.any(w <= 0.0):
            raise ValueError(
                "Green-Lagrange strain incompatible with SPD C = I + 2E "
                f"(non-positive eigenvalues: {w})."
            )
        return _sqrtm_spd(C)

    elif measure == "hencky":
        U = _expm_sym(E)
        # SPD check on U
        w = np.linalg.eigvalsh(U)
        if np.any(w <= 0.0):
            raise ValueError(
                "Hencky strain incompatible with SPD U = exp(E) "
                f"(non-positive eigenvalues: {w})."
            )
        return U

    else:
        raise ValueError(f"Unknown measure: {measure}")


def defgrad_from_strain(
    E: np.ndarray,
    measure: str,
    R: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construct deformation gradient F from a given strain tensor E.

    Semantics:
      - Reconstruct right stretch U from E via `_right_stretch_from_strain`.
      - If R is None: return F = U  (pure stretch, no rotation).
      - If R is provided: return F = R @ U.

    Notes:
      - This is not a unique inverse in general; only U is determined by E.
      - For 'engineering', this is a small-strain approximation.
      - For 'green-lagrange' and 'hencky', SPD consistency is enforced.
    """
    U = _right_stretch_from_strain(E, measure)

    if R is None:
        return U

    R = np.asarray(R, float)
    return R @ U

