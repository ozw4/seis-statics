# Refraction Design Mapping

This document records how the standalone `seis_statics.refraction` package maps
to the canonical `seisviewer2d` refraction design areas. It is a release
handoff reference, not an application integration plan.

## Scope Boundary

`seis_statics.refraction` owns pure numerical and array/dataclass transforms:

- first-layer V1 resolution and direct-arrival V1 estimation
- layer option normalization and layer observation masks
- half-intercept and GLI design matrices
- robust least-squares solves for global and cell V2 bedrock velocity
- T1LST 1/2/3-layer weathering and thickness formulas
- weathering replacement, datum, field-static, source-depth, uphole, and manual
  static composition
- multi-layer time-term solving and conversion

Applications own SEG-Y I/O, FastAPI/Pydantic schemas, job runtime, artifact
registries/writers, UI state, and migration orchestration.

## Public API Contract

Import from `seis_statics.refraction` for the stable public API. The package
keeps SciPy-heavy solver/design modules behind lazy `__getattr__` exports so
lightweight imports such as `seis_statics.refraction.options`,
`seis_statics.refraction.types`, and `seis_statics.refraction.status` remain
usable without eager solver imports.

The release-readiness tests pin `seis_statics.refraction.__all__` and verify
that every exported symbol resolves.

## Canonical Mapping

| Canonical design area | Package modules | Public entrypoints / dataclasses |
| --- | --- | --- |
| Shared refraction input model | `types.py`, `options.py` | `RefractionStaticInputModel`, `RefractionEndpointTable`, `RefractionStaticModelOptions`, `RefractionStaticSolverOptions`, `RefractionStaticDatumOptions`, `RefractionStaticConversionOptions` |
| Layer model and observation assignment | `layer_config.py`, `layer_observations.py` | `RefractionLayerConfig`, `normalize_refraction_layer_config`, `build_refraction_layer_observation_masks`, `refraction_layer_observation_qc` |
| First-layer V1 | `first_layer.py`, `v1.py` | `ResolvedRefractionFirstLayer`, `resolve_weathering_velocity_m_s`, `estimate_global_v1_from_direct_arrivals` |
| GLI / half-intercept design | `design_matrix.py`, `half_intercept.py`, `bedrock.py` | `build_refraction_static_design_matrix`, `build_refraction_static_design_matrix_from_arrays`, `build_refraction_half_intercept_design`, `estimate_global_bedrock_slowness_from_input_model` |
| Robust refraction solve | `solver.py` | `build_refraction_static_solver_system`, `solve_refraction_static_least_squares`, `solve_refraction_static_design_least_squares`, `RefractionStaticSolveResult` |
| Cell V2 local bedrock velocity | `cell_grid.py`, `cell_coordinates.py`, `cell_regularization.py`, `cell_velocity_status.py`, `solver.py` | `RefractionStaticRefractorCellOptions`, `build_refraction_cell_grid`, `assign_observation_midpoint_cells`, `project_refraction_cell_coordinates`, `cell_v2_m_s`, `row_midpoint_v2_m_s` |
| T1LST 1/2/3-layer formulas | `t1lsst.py` | `compute_t1lsst_1layer_weathering_correction`, `compute_t1lsst_2layer_weathering_correction`, `compute_t1lsst_3layer_weathering_correction`, `T1LSST_SIGN_CONVENTION` |
| Weathering replacement and datum | `weathering.py`, `weathering_replacement.py`, `datum.py` | `build_refraction_weathering_model_from_half_intercept_result`, `build_refraction_weathering_replacement_statics`, `build_refraction_datum_statics`, `build_refraction_endpoint_datum_statics` |
| Field corrections | `source_depth.py`, `uphole.py`, `manual_static.py`, `field_composition.py` | `resolve_refraction_source_depth`, `resolve_refraction_uphole`, `resolve_refraction_manual_static`, `compose_refraction_endpoint_field_corrections`, `compose_refraction_final_trace_shift` |
| Multi-layer time-term conversion | `multilayer_solver.py`, `multilayer_conversion.py` | `solve_refraction_multilayer_time_terms`, `build_refraction_multilayer_conversion`, `compute_refraction_multilayer_datum_statics_from_input_model` |
| Status and QC vocabulary | `status.py`, `types.py` | `REFRACTION_STATIC_STATUSES`, `LOCAL_V2_STATUS_VALUES`, `classify_refraction_endpoint_static_status` |

## Model Scope

The supported release scope is 1-layer, 2-layer, and 3-layer numerical
refraction statics. The local V2 cell mode is the migrated `solve_cell`
workflow: cells are explicit input/model state, observations are assigned by
midpoint, velocity is solved per active cell, and low-fold or outside-grid
observations are rejected with stable status strings.

The package does not infer application configuration from files, artifacts, or
viewer state. Callers must provide normalized arrays and dataclasses.

## Sign Convention

Refraction static outputs use:

```text
corrected(t) = raw(t - shift_s)
```

Time delays are positive when an event is late. Correcting a positive delay
therefore produces a negative applied shift. Tests cover this convention across
T1LST, source-depth, uphole, manual static, weathering replacement, datum, and
final field composition paths.

## Migration Handoff

For `seisviewer2d`, the next migration step is:

1. Depend on released `seis-statics` version `0.4.0`.
2. Replace application imports of local numerical modules with
   `seis_statics.time_term` and `seis_statics.refraction`.
3. Keep SEG-Y, FastAPI/Pydantic schemas, jobs, artifacts, and UI orchestration
   in `seisviewer2d`.
4. Remove the local `seisviewer2d/seis_statics/` copy after imports and tests
   pass.

This package must not import back from `seisviewer2d`.
