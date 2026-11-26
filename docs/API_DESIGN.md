# API evolution proposal

This is a design proposal for incremental releases. It does not add public classes to XPOSim 0.34.0. The current `import XPOSim`, its 40 exports and existing keyword conventions remain the compatibility baseline.

## Core objects

| Proposed object | Responsibility and contract |
| --- | --- |
| `Field` | Device array, immutable grid specification, wavelength and normalization metadata. Explicit `to_backend` and `to_numpy` operations. Validate backend, device, shape and precision at construction. `Beam` can be an alias only if it introduces no different coherence semantics. |
| `Backend` | Factories, FFTs, reductions, device identity, synchronization, memory statistics and supported dtypes. `auto` chooses a backend at construction, with a reported reason. Runtime device failure raises a contextual exception. Explicit migration to CPU avoids silently discarding device state or hiding a failed calculation. |
| `FresnelPropagator` | Free propagation with a declared boundary mode, precision and batch-memory budget. Reuse the current tested Fourier implementation initially. |
| `OpticalElement` | Reuse the existing hierarchy. Add adapters that consume and return `Field` alongside the existing array methods. |
| `System` | Ordered elements and explicit drifts between documented reference planes. Validate wavelength/backend compatibility before evaluation. Return fields or requested diagnostics without implicit host conversion. |
| `ElementRegistry` | Register validated element factories with unique names, configuration schema versions and explicit conflict errors. Deserialization reads registered types, without arbitrary module imports or executable YAML tags. |

Use SI floats as the stable numerical core. The new API should use explicit suffixes such as `distance_m` and `energy_kev`. Existing keyword names remain accepted. Optional unit libraries can be adapters with explicit conversion boundaries.

## Intended usage

The following is illustrative syntax for review and implementation in a later stage.

```python
backend = Backend.select("auto", precision="double")
field = Field.from_array(amplitude, grid=grid, wavelength_m=1e-10, backend=backend)
system = System([lens, Drift(distance_m=0.05), aperture])
result = system.run(field)
image = result.intensity().to_numpy()
```

A parameter scan should yield bounded chunks with stable parameter ordering, explicit output-retention options and a declared failure policy. Full-field output must report its predicted memory requirement. High-level CRL workflows should reuse the same element and propagation implementations.

## Precision, memory and concurrency

Proposed precision modes pair float32 coordinates/complex64 fields and float64 coordinates/complex128 fields. Phase accumulation may need a documented higher-precision path. Each mode needs its own validation tolerances and supported backend capability checks. Current double-precision results establish the reference baseline.

Add `close()` and context-manager support to retained result sessions, with explicit semantics for subsequent `plane_at` calls. Clear result-owned buffers and temporary source files on close. Never clear a backend's process-wide memory allocator as a side effect of closing one result. Cache keys must include geometry, wavelength, dtype, backend/device and boundary convention. Tensor mutation needs an ownership contract because read-only attributes alone do not prevent mutation of device arrays.

## Persistence

A versioned JSON document should store SI parameters, element registry IDs, schema versions, grid/boundary/precision choices, material provenance, library version and optional relative NPZ references with checksums. Round trips must preserve the public configuration. Add an optional YAML representation through safe loading of the same schema. Large device arrays and runtime sessions are external resources. Validate paths and reject unknown versions with actionable errors.

## Compatibility and migration

Introduce the new package alongside a compatibility facade for `import XPOSim`. Before moving material resources, add tests for installed wheels and source distributions, including execution from an unrelated working directory. Preserve current exports, signatures, default double precision and old single-file usage until an announced migration period is completed.

Use semantic versioning from a declared compatibility milestone. During 0.x releases still announce breaking changes explicitly. A proposed deprecation minimum is two minor releases and six months, whichever is longer. Removal requires a migration example and a major-version decision after 1.0. New warnings need `stacklevel` pointing to the caller and compatibility tests. No current API is newly deprecated by this audit package.

## Acceptance and rollback

Every adapter must reproduce legacy complex fields and public metrics within declared tolerances. Serialization needs round-trip, invalid-schema and resource-lifetime tests. Publish typed protocols and check the new core with mypy or Pyright before extending that requirement to legacy internals. Split packaging, backend adapters, fields, systems and serialization into independent reviewable changes. Revert one additive layer if its compatibility checks fail.
