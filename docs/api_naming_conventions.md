# Semantic Naming and Public API Conventions

This guide complements PEP 8 and the Google Python Style Guide. Those guides
standardize syntax; this document standardizes the domain meaning carried by
names in the IPIN examples.

The goal is that a learner can identify a value's role, units, coordinate frame,
and shape at a public API boundary without opening the implementation.

## Scope

These rules apply to:

- public functions, methods, classes, and dataclass fields in `core/`;
- reader-facing variables and helper functions in `ch2_*` through `ch8_*`;
- notebook code cells and documented code examples;
- new or substantially changed APIs.

Compact mathematical notation remains appropriate inside a small algorithmic
scope when the code maps it to the notation used by the book.

## Public names and mathematical notation

Prefer descriptive public parameters and results:

```python
def update_position(
    measured_position_xy_m: np.ndarray,
    measurement_covariance_xy_m2: np.ndarray,
) -> StateEstimate:
    ...
```

Avoid making callers infer the meaning of public `x`, `z`, `P`, `Q`, `R`, `H`,
or `S` arguments. These symbols may be used locally next to an explicit mapping:

```python
# Book notation: x is the state estimate and P is its covariance.
x = state_estimate
P = state_covariance
```

An equation-specific callback may retain the book symbol (`process_model(x, u,
dt_s)`) when its enclosing factory and docstring define every symbol, shape,
unit, and frame.

## Roles

Use one modifier for each data role:

| Role | Modifier | Example |
| --- | --- | --- |
| Ground truth | `true_` | `true_position_xy_m` |
| Sensor measurement | `measured_` | `measured_ranges_m` |
| Model prediction | `predicted_` | `predicted_state` |
| Estimator output | `estimated_` | `estimated_pose_se2` |
| Initial value | `initial_` | `initial_position_xy_m` |
| Previous/next sample | `previous_`, `next_` | `previous_velocity_mps` |

Do not alternate among `truth`, `gt`, `true_pos`, and `poses_true` in
reader-facing code. Existing serialized dataset keys may remain stable for
compatibility, but loaders should expose typed/descriptive accessors.

## Units

Include a unit suffix when a numeric public value is otherwise ambiguous:

| Quantity | Suffix | Example |
| --- | --- | --- |
| Time | `_s`, `_ms`, `_hz` | `timestamp_s`, `sample_rate_hz` |
| Distance/position | `_m`, `_xy_m`, `_xyz_m` | `range_m`, `position_xy_m` |
| Velocity/acceleration | `_mps`, `_mps2` | `velocity_xy_mps`, `accel_body_mps2` |
| Angle/angular rate | `_rad`, `_deg`, `_rad_s` | `yaw_rad`, `gyro_rad_s` |
| Radio power | `_dbm`, `_db` | `rss_dbm`, `path_loss_db` |
| Variance/covariance | squared unit | `range_variance_m2`, `covariance_xy_m2` |

Use `_std_` for a standard deviation and `_variance_` or `_covariance_` for a
squared quantity. Do not use a bare `noise_std` in APIs that mix metres,
radians, degrees, and dBm.

Unit conversion functions should name both source and destination, for example
`deg_per_hour_to_rad_per_sec`. If two quantities have the same dimensional
coefficient but different sampling interpretations, expose separate functions
instead of relying on a generic name.

## Coordinate frames and transforms

Use established frame suffixes: `_ecef`, `_enu`, `_ned`, `_map`, `_body`,
`_camera`, and `_sensor`. Add dimensional/units suffixes when useful:

```python
position_enu_m
scan_body_xy_m
accel_map_mps2
```

Transform names state their direction:

```python
rotation_body_to_map
translation_camera1_to_camera2_m
pose_sensor_to_map
```

Generic `BODY` is insufficient when more than one axis convention is supported.
Use an explicit convention from `core.coords.frames`: `BODY_CH2` (Chapter 2,
`core/coords/rotations.py`), `BODY_FLU` (Chapter 6, `core/sensors/`,
`FrameConvention`), or `BODY_FRD` (the standard aerospace/vehicle convention,
used by no chapter in this repository but kept for readers arriving from
other literature) -- or a `FrameConvention` object whose axes are documented.

Public ndarray documentation must state:

1. shape;
2. units;
3. frame;
4. element order when it is not obvious.

## Domain vocabulary

Use these nouns consistently:

| Noun | Meaning |
| --- | --- |
| `anchor` | Fixed RF/UWB ranging infrastructure |
| `access_point` | Wi-Fi access point |
| `reference_point` | Surveyed fingerprint-database location |
| `landmark` | Environmental feature estimated or observed by SLAM |

Text may introduce common synonyms such as beacon, but code should not switch
nouns for the same object. Use `reference_anchor_index` for the anchor selected
as the TDOA reference.

## Method verbs and side effects

| Prefix/name | Contract |
| --- | --- |
| `load_*` | Read data or configuration |
| `generate_*`, `simulate_*` | Produce synthetic data |
| `compute_*` | Deterministic calculation without state mutation |
| `estimate_*`, `localize_*` | Produce a domain estimate |
| `solve_*` | Run a numerical solver and return convergence information |
| `predict`, `update` | Mutate a stateful filter |
| `process_*`, `step` | Advance a stateful pipeline; docstring must say it mutates state |
| `run_*` | Orchestrate a complete example or pipeline |
| `plot_*` | Build and return figure objects; do not save or show implicitly |
| `save_*` | Write an artifact |
| `create_*` | Construct/configure an object or callback set |

A boolean should not change the type or meaning of another positional argument.
Prefer mutually exclusive keyword parameters or separate descriptive methods.

## Return values

Use a dataclass or `NamedTuple` when a result has multiple domain fields. A
dictionary is appropriate for open-ended metadata, not for the primary contract.

```python
@dataclass(frozen=True)
class PositioningResult:
    estimated_position_xy_m: np.ndarray
    converged: bool
    iterations: int
    residual_rmse_m: float
    covariance_xy_m2: np.ndarray | None = None
```

Result fields include units and frames where needed. During compatibility
migrations, a new result may implement tuple unpacking or the old function may
wrap the new API.

## Abbreviations

Use widely established domain abbreviations in type or algorithm names (`EKF`,
`UWB`, `RSS`, `ICP`, `NDT`, `SLAM`). Spell out abbreviations that create a
domain collision. For example, expose `maximum_a_posteriori_localize` alongside
the historical `map_localize`, because `map` also refers to a building or SLAM
map.

Prefer complete words in public names:

- `estimated`, not `est`;
- `position`, not `pos`;
- `reference`, not `ref`;
- `index`, not `idx`.

Short forms remain acceptable for loop indices and tightly scoped equations.

## Compatibility policy

Semantic cleanup is additive before it is breaking:

1. Add the descriptive function, keyword, property, or result type.
2. Keep the old public name as a documented alias or wrapper.
3. Add an equivalence test proving old and new paths return the same result.
4. Migrate examples, notebooks, and documentation to the descriptive name.
5. Emit `DeprecationWarning` only after the replacement is available.
6. Remove the legacy name only in a documented breaking release.

Serialized dataset keys are more expensive to rename than Python accessors and
should normally remain readable through the loader.

## Review checklist

For every new or changed public API, verify:

- [ ] The function/method verb matches its side effects.
- [ ] Physical values have unambiguous units.
- [ ] Frame-sensitive arrays identify their frame or accept a convention object.
- [ ] Truth, measurement, prediction, and estimate names are distinct.
- [ ] Ndarray shapes, units, frames, and element order are documented.
- [ ] Multi-field results are typed rather than positional tuples/raw dicts.
- [ ] Mathematical abbreviations are explained at the public boundary.
- [ ] Compatibility aliases and equivalence tests accompany a rename.
