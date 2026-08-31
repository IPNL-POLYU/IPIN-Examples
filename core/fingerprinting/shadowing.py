"""Spatially correlated shadow fading for synthetic radio maps.

Shadow fading is a property of a *location*, not of a measurement. The same
wall, the same filing cabinet and the same lift shaft attenuate the same AP
from the same spot on every visit, so two surveys of one building disagree at a
point only by the fast, per-sample part of the variability -- never by the whole
of it.

Modelling the whole term as per-draw noise is the difference between a radio map
and a table of random numbers, and it is not a cosmetic one:

- The radio map stops being a smooth function of position, and smoothness is the
  only property fingerprinting exploits. On the Chapter 5 grid database, nearest
  neighbour scored 6.93 m against noiseless queries where the 5 m grid's own
  quantisation floor is 2.04 m -- ``sqrt(2 s^2 / 12)``, the rms distance from a
  uniform position to the nearest node -- and 2.18 m is what it achieves on a
  map with no shadowing at all. Almost all of that gap was the map arguing with
  itself.
- A query drawn at a point becomes inconsistent with the map at that same point,
  which is the correlation the method is built on.

This module supplies the missing piece: a zero-mean Gaussian random field
``S_ap(p)``, one independent realisation per (floor, AP), evaluable at any
position rather than only at the surveyed reference points. A generator adds it
to the path loss when it builds the map; a query generator adds *the same field
at the query's own position*, so the query and the map agree about where the
walls are.

Construction
------------
Random Fourier modes (Rahimi & Recht's "random features", which is Bochner's
theorem read as a sampling recipe)::

    S(p) = sigma * sqrt(2 / K) * sum_{k=1..K} cos(w_k . p + phi_k)

with ``phi_k ~ U(0, 2 pi)`` and ``w_k`` drawn from the spectral density of the
chosen covariance. For the squared-exponential kernel
``C(r) = sigma^2 exp(-r^2 / (2 L^2))`` that density is Gaussian with covariance
``I / L^2``, so ``w_k ~ N(0, I / L^2)`` -- one line, no grid, no interpolation.

Three properties matter here and all three are why this form was chosen over a
smoothed white-noise grid:

- ``E[S(p)] = 0`` and ``Var[S(p)] = sigma^2`` exactly, for every ``p``, so the
  ``shadow_fading_std_dBm`` a dataset declares keeps meaning what it says.
- ``Cov[S(p), S(q)] = sigma^2 exp(-||p - q||^2 / (2 L^2))``, so the field is
  smooth on the scale of ``L`` and decorrelated well beyond it.
- It is defined at *every* position, not on a lattice, so a query at an
  arbitrary point between reference points needs no interpolation scheme whose
  choice would then be a second modelling decision nobody wrote down.

The kernel is squared-exponential rather than the exponential (Gudmundson) form
usually fitted to drive tests. Both are standard; the exponential one is not
mean-square differentiable, and a nowhere-differentiable radio map is the wrong
idealisation for a chapter whose whole argument is that the map varies smoothly
with position. The decorrelation length is the parameter that carries the
physics here, not the shape of the tail.

Reproducibility
---------------
The field is a pure function of ``(seed, decorrelation_m, sigma_dB, n_modes)``
and the (floor, AP) index. Each realisation draws from
``np.random.default_rng([seed, floor_id, ap_index])``, so AP 3 on floor 1 is the
same field whatever else the building contains -- the dense, sparse and baseline
surveys of one building see one radio environment, and adding an AP does not
redraw the others. :meth:`ShadowingField.to_meta` writes everything needed to
rebuild it, and :meth:`ShadowingField.from_meta` rebuilds it, so the dataset
generator and any consumer of the dataset share one code path rather than two
copies of the same constants.

Author: Li-Ta Hsu
References: Chapter 5, Section 5.1 (radio map construction).
"""

from dataclasses import dataclass

import numpy as np

#: Decorrelation length of the shadowing field, in metres.
#:
#: Indoor shadowing decorrelation distances are reported in the 5-10 m range,
#: set by the spacing of the walls and partitions doing the shadowing. 8 m sits
#: inside that range and, importantly, *above* the 5 m survey grid the Chapter 5
#: databases use: a survey can only represent a field it samples faster than the
#: field varies, so a radio map on a 5 m grid presupposes a correlation length
#: longer than 5 m. A building whose shadowing decorrelated in 2 m would not be
#: a building fingerprinting works in, which is a statement about surveys rather
#: than a knob to tune.
DEFAULT_DECORRELATION_M = 8.0

#: Number of Fourier modes per realisation.
#:
#: The marginal variance is sigma^2 in expectation for any K; K controls how
#: closely one realisation's *empirical* covariance follows the target. 256 is
#: cheap (one cosine per mode per point per AP) and puts the pooled empirical
#: standard deviation within a percent of sigma on the shipped databases.
DEFAULT_N_MODES = 256


@dataclass(frozen=True)
class ShadowingField:
    """A spatially correlated shadow-fading field, one realisation per AP.

    Call the instance to evaluate it. Instances are immutable and carry no
    state beyond their drawn modes, so evaluating at a position never changes
    what a later evaluation returns.

    Attributes:
        sigma_dB: Marginal standard deviation of the field (dB).
        decorrelation_m: Correlation length L (metres) of the squared-exponential
            covariance ``sigma^2 exp(-r^2 / (2 L^2))``.
        n_modes: Number of Fourier modes K per realisation.
        seed: Seed the realisations were drawn from.
        frequencies: Angular frequencies w_k, shape (n_floors, n_aps, K, 2).
        phases: Phases phi_k, shape (n_floors, n_aps, K).

    Examples:
        >>> field = ShadowingField.build(n_aps=2, n_floors=1, sigma_dB=4.0, seed=42)
        >>> s = field(np.array([[10.0, 20.0], [10.5, 20.0]]), floor_id=0)
        >>> s.shape
        (2, 2)
        >>> # Half a metre apart, well inside the correlation length: nearly equal.
        >>> bool(abs(s[0, 0] - s[1, 0]) < 1.0)
        True
    """

    sigma_dB: float
    decorrelation_m: float
    n_modes: int
    seed: int
    frequencies: np.ndarray
    phases: np.ndarray

    @classmethod
    def build(
        cls,
        *,
        n_aps: int,
        n_floors: int,
        sigma_dB: float,
        seed: int,
        decorrelation_m: float = DEFAULT_DECORRELATION_M,
        n_modes: int = DEFAULT_N_MODES,
    ) -> "ShadowingField":
        """Draw the field's modes.

        Args:
            n_aps: Number of access points; one independent realisation each.
            n_floors: Number of floors; each floor gets its own realisation of
                every AP's field, because each floor has its own walls.
            sigma_dB: Marginal standard deviation (dB) of the field.
            seed: Base seed. Realisation (floor, ap) is drawn from
                ``np.random.default_rng([seed, floor, ap])``, so it does not
                depend on how many other APs or floors exist.
            decorrelation_m: Correlation length L in metres.
            n_modes: Number of Fourier modes per realisation.

        Returns:
            A :class:`ShadowingField`.

        Raises:
            ValueError: If any size or scale is not positive.
        """
        if n_aps < 1 or n_floors < 1 or n_modes < 1:
            raise ValueError(
                f"n_aps, n_floors and n_modes must all be >= 1, got "
                f"{n_aps}, {n_floors}, {n_modes}"
            )
        if sigma_dB < 0.0:
            raise ValueError(f"sigma_dB must be non-negative, got {sigma_dB}")
        if decorrelation_m <= 0.0:
            raise ValueError(f"decorrelation_m must be positive, got {decorrelation_m}")

        frequencies = np.empty((n_floors, n_aps, n_modes, 2))
        phases = np.empty((n_floors, n_aps, n_modes))

        for floor in range(n_floors):
            for ap in range(n_aps):
                rng = np.random.default_rng([seed, floor, ap])
                # Spectral density of the squared-exponential kernel: N(0, I/L^2).
                frequencies[floor, ap] = rng.normal(
                    0.0, 1.0 / decorrelation_m, size=(n_modes, 2)
                )
                phases[floor, ap] = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)

        return cls(
            sigma_dB=float(sigma_dB),
            decorrelation_m=float(decorrelation_m),
            n_modes=int(n_modes),
            seed=int(seed),
            frequencies=frequencies,
            phases=phases,
        )

    @classmethod
    def from_meta(cls, meta: dict) -> "ShadowingField":
        """Rebuild the field a dataset was generated with, from its metadata.

        This is the half that makes the model shared rather than duplicated: a
        query generator calls it and gets the *same* field the radio map was
        built from, so a query and the map agree about the building. Two hand
        copies of these constants would only have to drift once.

        Args:
            meta: A dataset's ``meta`` dict, carrying ``shadow_field`` as
                written by :meth:`to_meta`, plus ``n_floors``.

        Returns:
            A :class:`ShadowingField` identical to the generator's.

        Raises:
            KeyError: If the metadata carries no ``shadow_field`` block, which
                means the dataset predates this model and its shadowing is not
                reconstructible at all.
        """
        spec = meta["shadow_field"]
        return cls.build(
            n_aps=int(spec["n_aps"]),
            n_floors=int(spec["n_floors"]),
            sigma_dB=float(spec["sigma_dB"]),
            seed=int(spec["seed"]),
            decorrelation_m=float(spec["decorrelation_length_m"]),
            n_modes=int(spec["n_modes"]),
        )

    def to_meta(self) -> dict:
        """Everything needed to rebuild this field, as JSON-serialisable data."""
        return {
            "type": "random_fourier_modes",
            "kernel": "squared_exponential",
            "sigma_dB": self.sigma_dB,
            "decorrelation_length_m": self.decorrelation_m,
            "n_modes": self.n_modes,
            "seed": self.seed,
            "n_aps": int(self.frequencies.shape[1]),
            "n_floors": int(self.frequencies.shape[0]),
        }

    @property
    def n_aps(self) -> int:
        """Number of APs the field carries a realisation for."""
        return int(self.frequencies.shape[1])

    @property
    def n_floors(self) -> int:
        """Number of floors the field carries a realisation for."""
        return int(self.frequencies.shape[0])

    def __call__(self, xy: np.ndarray, floor_id: int | np.ndarray) -> np.ndarray:
        """Evaluate the field at arbitrary positions.

        Args:
            xy: Positions, shape (2,) for one point or (n, 2) for many. Metres,
                in the same frame as the database's ``locations``.
            floor_id: Floor index, a scalar or an (n,) integer array.

        Returns:
            Shadowing in dB. Shape (n_aps,) for a single point, else
            (n, n_aps).

        Raises:
            ValueError: If positions are not 2-D coordinates, if the floor
                count does not match the positions, or if a floor index has no
                realisation.
        """
        points = np.asarray(xy, dtype=float)
        single = points.ndim == 1
        points = np.atleast_2d(points)
        if points.shape[1] != 2:
            raise ValueError(
                f"positions must be 2-D coordinates (n, 2), got shape "
                f"{points.shape}"
            )

        floors = np.atleast_1d(np.asarray(floor_id, dtype=int))
        if floors.size == 1:
            floors = np.full(points.shape[0], int(floors[0]))
        if floors.shape[0] != points.shape[0]:
            raise ValueError(
                f"got {floors.shape[0]} floor ids for {points.shape[0]} positions"
            )
        if floors.min() < 0 or floors.max() >= self.n_floors:
            raise ValueError(
                f"floor ids {floors.min()}..{floors.max()} outside the "
                f"{self.n_floors} floors this field was built for"
            )

        scale = self.sigma_dB * np.sqrt(2.0 / self.n_modes)
        out = np.empty((points.shape[0], self.n_aps))

        # One einsum per floor: each floor has its own realisation, and grouping
        # keeps this a handful of vectorised calls rather than one per point.
        for floor in np.unique(floors):
            rows = floors == floor
            # (n_f, 2) . (n_aps, K, 2) -> (n_f, n_aps, K)
            projected = np.einsum("nd,akd->nak", points[rows], self.frequencies[floor])
            out[rows] = scale * np.cos(projected + self.phases[floor]).sum(axis=2)

        return out[0] if single else out
