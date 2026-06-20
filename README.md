# seis-statics

`seis-statics` is a lightweight Python package for seismic static-correction utilities.

It provides numerical core routines for source/receiver residual statics, trace time-shift application, and simple datum static calculations. The package is intended to be shared by `seisviewer2d`, `seisai`, and related seismic-processing workflows.

## Scope

This package contains pure numerical/statics utilities only.

It does not provide:

- SEG-Y I/O
- viewer/UI functionality
- FastAPI services
- job/artifact management
- seismic MAE or denoising models

Those responsibilities should remain in higher-level applications such as `seisviewer2d` or `seisai`.

## Installation

For local development:

```bash
python -m pip install -e .
```

If using the package from another local repository:

```bash
python -m pip install -e ../seis-statics
```

For migration work, keep sibling checkouts separate:

```text
workspaces/
  seis-statics/
  seisviewer2d/
```

`seisviewer2d` may depend on this package during integration, but runtime and
test dependencies must remain one-way: `seisviewer2d -> seis_statics`.
Sibling checkout or bind-mount access to `seisviewer2d` is for read-only
reference and regression comparison only. Package code, normal tests, and
fixtures in this repository must be standalone and must not import from,
symlink to, or add `seisviewer2d`/`app` paths at runtime.

## Main APIs

### Source/receiver residual statics from lag observations

Use this when lag observations are already estimated, for example from MAE/BlindTrace pilot traces and NCC.

```python
import numpy as np

from seis_statics.residual import solve_source_receiver_statics

lag_s = np.array([0.010, 0.012, -0.004, -0.002])
source_id = np.array([1, 1, 2, 2])
receiver_id = np.array([10, 11, 10, 11])
weight = np.ones_like(lag_s)

result = solve_source_receiver_statics(
    lag_s=lag_s,
    source_id=source_id,
    receiver_id=receiver_id,
    weight=weight,
    robust=True,
)

print(result.source_delay_s)
print(result.receiver_delay_s)
print(result.trace_delay_s)
print(result.applied_shift_s)
```

This solves the additive model:

```text
lag_i ≈ S_source(i) + R_receiver(i)
```

where `S` is the source-consistent delay and `R` is the receiver-consistent delay.

### First-break residual statics

Use this for first-break or pick-based residual statics workflows, such as those used by `seisviewer2d`.

```python
from seis_statics.residual import solve_first_break_residual_statics
```

This API preserves the first-break residual statics workflow while keeping the numerical implementation outside the application layer.

### Applying trace shifts

```python
from seis_statics.trace_shift import apply_trace_shifts_to_array

corrected = apply_trace_shifts_to_array(
    traces=traces,
    sample_interval_s=dt,
    trace_shift_s_sorted=result.applied_shift_s,
)
```

## Sign convention

The package uses the following convention:

```text
delay_s > 0
```

means the observed event is later than the reference event.

To correct a positive delay, the applied trace shift is negative:

```text
applied_shift_s = -delay_s
```

For `solve_source_receiver_statics`, the result includes:

```python
result.trace_delay_s
result.applied_shift_s
```

where:

```text
result.applied_shift_s = -result.trace_delay_s
```

## Invalid traces and zero weights

`valid_mask=False` means the trace is not used as a valid lag observation. For such traces, the source/receiver delay prediction is not evaluated, and the result values are `NaN`:

```text
trace_delay_s = NaN
applied_shift_s = NaN
residual_s = NaN
```

If a trace should remain evaluable but should not influence the least-squares solution, use:

```text
valid_mask=True
weight=0.0
```

This is useful for workflows where an endpoint is known but the lag observation has low confidence.

## Gauge freedom

The source/receiver decomposition has a gauge freedom:

```text
S_source + c
R_receiver - c
```

gives the same trace delay.

Therefore, source and receiver terms should not be interpreted independently without considering the gauge constraint. The most stable quantity is usually the trace-level sum:

```text
S_source(i) + R_receiver(i)
```

## Typical use with seisai MAE

A minimal MAE-assisted residual static workflow is:

```text
raw gather
→ MAE / BlindTrace pilot prediction
→ NCC lag estimation between pilot and observed trace
→ solve_source_receiver_statics
→ apply_trace_shifts_to_array
→ pre-aligned gather
```

In this use case, `seisai` should generate the lag observations, source IDs, receiver IDs, and confidence weights. `seis-statics` only performs the source/receiver decomposition and shift application.

## Development

Install in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest -q tests
```

If using a `src` layout before installation:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q tests
```

Check that the package does not depend on application code or application-only
dependencies:

```bash
python -m pytest -q tests/test_seis_statics_no_app_dependency.py
python -m pytest -q tests/test_seis_statics_package_import.py
```

## Dependencies

Core dependencies are intentionally minimal:

```text
numpy
scipy
```

Application-specific dependencies such as `segyio`, `FastAPI`, `pydantic`, or viewer-related packages should not be required by this package.
