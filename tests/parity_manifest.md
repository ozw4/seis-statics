# seisviewer2d Parity Manifest

Canonical source: `seisviewer2d` time-term and refraction numerical design
docs/tests as migrated through issues 001-023.

This manifest records the package-side regression coverage that replaces the
application-local numerical tests. It intentionally excludes SEG-Y I/O,
FastAPI/Pydantic schemas, job runtime, artifact writing, and UI assertions.

| Package regression area | Package tests | Parity notes |
| --- | --- | --- |
| Package boundary and import contract | `tests/test_seis_statics_no_app_dependency.py`, `tests/test_seis_statics_package_import.py`, `tests/test_seis_statics_release_readiness.py` | Blocks `app`, FastAPI, Pydantic, `segyio`, `seisviewer2d`, path mutation, and pins public `__all__`. |
| time-term input/moveout | `tests/test_seis_statics_time_term_moveout.py` | Preserves sorted trace arrays, distance-source selection, reciprocal pair mapping, and moveout QC. |
| time-term design matrix | `tests/test_seis_statics_time_term_design_matrix.py` | Preserves source/receiver node row layout, valid-pick and moveout filtering, trace row maps, and node observation counts. |
| time-term sparse solver | `tests/test_seis_statics_time_term_sparse_solver.py` | Preserves sparse least-squares, damping objective convention, gauge handling, endpoint support, prediction masks, and NaN unsupported traces. |
| time-term robust solver and applied shifts | `tests/test_seis_statics_time_term_robust_solver.py`, `tests/test_seis_statics_time_term_apply_shift.py` | Preserves robust rejection safety and `applied_weathering_shift_s_sorted = -trace_time_term_delay_s_sorted`. |
| refraction shared types/options/status | `tests/test_seis_statics_refraction_types.py`, `tests/test_seis_statics_refraction_options.py`, `tests/test_seis_statics_refraction_status.py`, `tests/test_seis_statics_refraction_layer_config.py` | Preserves application-free dataclasses, literal option validation, layer scope, and status vocabulary. |
| refraction V1 / first layer | `tests/test_seis_statics_refraction_v1.py`, `tests/test_seis_statics_refraction_first_layer.py` | Preserves direct-arrival V1 estimation, robust group filtering, velocity bounds, and resolved first-layer validation. |
| refraction layer observations | `tests/test_seis_statics_refraction_layer_observations.py` | Preserves half-open offset gates, overlap policies, independent/exclusive masks, and rejection reasons. |
| refraction GLI design matrix | `tests/test_seis_statics_refraction_design_matrix.py`, `tests/test_seis_statics_refraction_cell_design_matrix.py` | Preserves global/fixed/cell matrix layout, node and cell low-fold filtering, sorted trace row maps, and QC summaries. |
| refraction robust solver | `tests/test_seis_statics_refraction_solver.py`, `tests/test_seis_statics_refraction_cell_solver.py`, `tests/test_seis_statics_refraction_cell_robust_synthetic.py` | Preserves global/fixed/cell solve parity, robust rejection safety, rank/identifiability checks, gauge canonicalization, and cell V2 outputs. |
| refraction cell geometry and regularization | `tests/test_seis_statics_refraction_cell_grid.py`, `tests/test_seis_statics_refraction_cell_coordinates.py`, `tests/test_seis_statics_refraction_cell_regularization.py` | Preserves row-major cell IDs, midpoint assignment, 3D/projected coordinates, smoothing rows, and active-cell mapping. |
| refraction T1LST / half-intercept / bedrock | `tests/test_seis_statics_refraction_t1lsst.py`, `tests/test_seis_statics_refraction_t1lsst_multilayer.py`, `tests/test_seis_statics_refraction_half_intercept.py`, `tests/test_seis_statics_refraction_bedrock.py` | Preserves 1/2/3-layer formulas, endpoint component rows, canonical sign convention, and bedrock slowness estimation. |
| refraction weathering and replacement | `tests/test_seis_statics_refraction_weathering.py`, `tests/test_seis_statics_refraction_weathering_replacement.py` | Preserves thickness formulas, local V2 overlays, status propagation, replacement shifts, and JSON-safe QC. |
| refraction datum and field composition | `tests/test_seis_statics_refraction_datum.py`, `tests/test_seis_statics_refraction_field_composition.py`, `tests/test_seis_statics_refraction_manual_static.py`, `tests/test_seis_statics_refraction_source_depth.py`, `tests/test_seis_statics_refraction_uphole.py` | Preserves datum/elevation shifts, source-depth/up-hole/manual static composition, final trace-shift sign, and invalid-status handling. |
| refraction multi-layer conversion | `tests/test_seis_statics_refraction_multilayer_solver.py`, `tests/test_seis_statics_refraction_multilayer_conversion.py` | Preserves `v2_t1`, `v3_t2`, and `vsub_t3` time-term solves, layer velocity order checks, local V2 projection, and datum conversion outputs. |
| residual and trace-shift compatibility | `tests/test_seis_statics_residual_design_matrix.py`, `tests/test_seis_statics_residual_sparse_solver.py`, `tests/test_seis_statics_residual_robust_solver.py`, `tests/test_seis_statics_trace_shift.py` | Preserves pre-existing source/receiver residual and trace application contracts used alongside time-term/refraction statics. |
