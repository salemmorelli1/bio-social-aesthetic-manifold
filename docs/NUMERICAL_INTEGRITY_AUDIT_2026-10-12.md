# Numerical and input-integrity audit

Audit date: October 12, 2026 (America/New_York). Baseline main:
`7a4f7286ab63bce9660ce903f526d9bd6fa8c311`, tree
`34d9cb3f1ed195c11e690b66618193ac7b32e648`. Corrected engine: 2.0.1.

The baseline passed 94 Python tests, 15 AME TypeScript tests, and six legacy
JavaScript tests. The audit preserves the proper-rotation GPA, 132-dimensional
tangent chart, PCA, analytic diagonal-target shrinkage, eight frontal metrics,
four additional projected mesh measurements, and simulated reference choices.
It does not add appearance ratings, demographic inference, clinical norms, or
prescriptive changes.

## Reproduced defects and corrections

### 1. Core centering still depended on a distant coordinate origin

The frontal module and AME already translated before scaling. The core GPA
normalizer instead averaged raw coordinates. Rounding in that mean could leave
a translation component in the supposedly centered unit preshape.

For the canonical configuration translated by `(1e13,-1e13)`, the old core
reported partial Procrustes distance **0.0394408500**, while subtracting the
known offset from the *same quantized input* gave **0.0175231779**. The issue
was an additional arithmetic error, not recovery of lost input digits.

The core now subtracts an observed point before computing the relative mean.
Its overflow/underflow fallback normalizes translated differences when those
differences are representable, and prescales raw coordinates only when
opposite-sign subtraction overflows. Regressions compare preshapes, centroid
size, and reported distances against explicit recentering at four origins.
Input quantization and subnormal precision limits remain disclosed.

### 2. Coordinate layout and scalar types were silently changed

Both Python bridges reshaped any array containing 136 elements. Thus a `2x68`
matrix or an extra batch axis could become an incorrectly ordered face. Casting
accepted numeric strings and booleans, discarded complex imaginary components,
and dropped masked-coordinate information. An unrepresentable integer could
also escape the frontal JSON error boundary.

Both bridges now accept only a `(68,2)` matrix or a flat `(136,)` vector of
finite real coordinates. They validate scalar types before float conversion,
reject missing masks, and return structured errors for unrepresentable values.
The legacy JSON file parser enforces the same layouts and numeric JSON tokens;
it cannot convert `null`, booleans, strings, or extra nesting into measurements.
Delimited numeric text and the existing flat Pyodide bridge remain supported.
These checks establish a layout contract, not anatomical landmark validity.

### 3. GPA convergence controls accepted invalid or overflowing values

NaN tolerance could run all iterations, infinity could claim immediate
convergence, and a boolean could act as an iteration limit. Adding one to a
maximum NumPy int64 limit could wrap the loop bound and execute zero iterations.

The limit must now be a positive integer, converted to a Python integer before
loop arithmetic. Tolerance must be a positive finite representable real scalar.
Masked GPA samples are rejected before array conversion can discard their mask.

### 4. Cached calibration arrays could be accidentally modified

The frozen dataclasses still held writable NumPy arrays. A caller could change
a shared consensus or covariance and silently affect later analyses.
Cached fitted arrays, including the GPA sample, Cholesky factor, PCA directions,
and reference distances, are now read-only. This prevents ordinary accidental
mutation; it is not a security boundary against deliberate process modification.

### 5. An incomplete mesh could be labeled a complete observation

JavaScript `some` skips sparse array slots. Removing vertex 11 from a 478-slot
mesh, outside the 68 selected indices, still passed AME validation and produced
68 mapped points. The legacy adapter checked only selected vertices and could
accept incomplete arrays, invalid image dimensions, or overflowing projections.

All mesh slots are now checked. AME retains its exact 478-point contract; the
legacy adapter retains both complete 468- and 478-point schemas. Pixel dimensions
must be positive integers, and projected coordinates must remain finite.

### 6. Missing residuals could become zero displacements

The legacy warp helper converted `null`, `false`, and numeric strings with
`Number()`. A missing residual could therefore become a zero-vector warp.
Numeric geometry helpers now require actual finite JavaScript numbers, and
point validation visits sparse slots. Valid zero residuals retain their identity
warp behavior.

### 7. The legacy photo route could silently choose one of several faces

The task requested only one detection, and the adapter selected the first face
if a caller returned several. It now requests up to two, so ambiguity is
observable, and rejects any result other than exactly one visible face. The
AME route already had this rejection. Model integrity checks and browser-local
image processing remain in place.

### 8. Locked web dependencies failed the current security gate

The original lock produced ten npm findings: seven high and three moderate.
The patched lock produces zero findings at the audit date. Next receives a
compatible 15.5.27 patch. Tailwind 3's brace-parser dependency had no published
patch, so the build moves to Tailwind 4.3.3 and its PostCSS plugin, with explicit
loading of the existing configuration and source directory. React stays on 19.
Sharp is pinned to 0.35.5 and source-map-js to 1.2.2 through overrides.

This is a dependency remediation and build migration, not evidence that any
listed vulnerability was exploited in this application. Reachability varies
between a trusted static build and a self-hosted request-processing application.
The lock, type checks, full static build, and installed-Edge rendering check
must all pass; the npm security gate remains enabled.

## Reproduction and publication evidence

- The 23 initially failing Python regression cases were reproduced before
  correction; five other new boundary cases already passed. Additional cases
  cover integer wraparound, huge tolerance, masked GPA samples, and source
  verification. Both positive and negative cases are retained.
- The simulated-reference fixture retains its existing numeric values within
  the established `1e-12` cross-language tolerance and updates the engine label
  to 2.0.1. No recalibration or new population sample is claimed.
- The 27-page report is rebuilt against the **unchanged numerical fixture** and
  the new engine label. Pixel comparison of all pages found only the version
  character on page 20 changed; the other 26 pages were identical. Two clean
  builds must reproduce the committed PDF bytes on both CI operating systems.
- `SOURCE_SHA256SUMS.json` verifies tracked delivery bytes, including the PDF.
  `.gitattributes` fixes text checkouts to LF on Windows and Linux.
- PR CI runs the Python, JavaScript, AME type/build, dependency audit, reference
  fixture, and PDF checks on Linux and Windows. Windows separately runs the
  built preview in installed Edge and preserves a rendering artifact.

The Edge smoke check validates initial rendering, hidden video styling, layout,
and absence of unsolicited model/network requests. It does not measure detector
accuracy, pose error, camera calibration, or empirical reliability on real
photos. Simulated reference positions remain descriptive in-sample context,
not population percentiles, p-values, or appearance judgments.

## Primary implementation and security references

- [NumPy mean and floating-point precision](https://numpy.org/doc/2.3/reference/generated/numpy.mean.html)
- [Tailwind's PostCSS/configuration upgrade guide](https://tailwindcss.com/docs/upgrade-guide)
- [Brace-parser advisory](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm)
- [Sharp/librsvg advisory and patched release](https://github.com/advisories/GHSA-wq5f-xc86-pv6w)
- [PostCSS selector-parser advisory](https://github.com/advisories/GHSA-rj75-hqrm-r3gf)
- [source-map-js advisory](https://github.com/advisories/GHSA-68fv-2mgg-jv7q)

The numerical discrepancies and regressions above were derived and reproduced
from this repository. The corrections do not validate synthetic references as
empirical biological or clinical standards.
