# Mathematical Theory of the Shape Manifold Demonstration

## 1. Scope and interpretation

This application is a deterministic, synthetic demonstration of landmark-based
geometric morphometrics. It estimates no empirical biological population,
infers no demographic or identity category, assigns no aesthetic value, and
generates no recommended coordinate modifications. Reference A, Reference B,
and the pooled reference are simulated probability models used to demonstrate
the mechanics of shape-space statistics.

The website also contains a local image-confirmation view. That view is
architecturally separate from this mathematical pipeline: image pixels and
image metadata are not passed to Python, used to obtain landmarks, or joined
to any statistical result.

The central object is a two-dimensional landmark configuration

\[
X=\begin{bmatrix}
x_1 & y_1\\
\vdots & \vdots\\
x_k & y_k
\end{bmatrix}\in\mathbb R^{k\times 2},\qquad k=68.
\]

The landmarks follow the conventional Dlib 68-point indexing topology:

| Indices | Region |
|---:|---|
| 0–16 | Jawline |
| 17–21 | Right eyebrow |
| 22–26 | Left eyebrow |
| 27–30 | Nose bridge |
| 31–35 | Nose base |
| 36–41 | Right eye |
| 42–47 | Left eye |
| 48–59 | Outer lip |
| 60–67 | Inner lip |

These labels define coordinate correspondence only. They do not imply that a
synthetic configuration is a biometric norm.

## 2. Translation and centroid size

Let \(\mathbf 1_k\) be a length-\(k\) vector of ones and define the centering
matrix

\[
C_k=I_k-\frac{1}{k}\mathbf 1_k\mathbf 1_k^\mathsf T.
\]

The centered configuration is

\[
X_c=C_kX=X-\mathbf 1_k\bar x^\mathsf T,
\qquad
\bar x=\frac{1}{k}X^\mathsf T\mathbf 1_k.
\]

Centroid size is the Frobenius norm of the centered configuration:

\[
c(X)=\|X_c\|_F
=\left[\sum_{j=1}^{k}\|x_j-\bar x\|_2^2\right]^{1/2}.
\]

Dividing by centroid size produces a unit preshape,

\[
Z=\frac{X_c}{c(X)},
\qquad \|Z\|_F=1.
\]

Centering removes two translational degrees of freedom. Unit scaling removes
one additional degree of freedom.

## 3. Generalized Procrustes Analysis

For preshapes \(Z_1,\ldots,Z_n\), Generalized Procrustes Analysis seeks proper
rotations \(R_i\in SO(2)\) and a unit-norm consensus \(M\) minimizing

\[
G(M,R_1,\ldots,R_n)
=\sum_{i=1}^{n}\|Z_iR_i-M\|_F^2,
\qquad
M^\mathsf TC_k=M^\mathsf T,
\qquad
\|M\|_F=1.
\]

The implementation follows the least-squares Procrustes procedure described in
the Stony Brook/SUNY morphometrics literature:

1. center every configuration;
2. divide each configuration by its centroid size;
3. select an initial preshape as the working consensus;
4. rotate every preshape toward that consensus;
5. average the aligned configurations and renormalize the mean;
6. repeat until the change in consensus is below a numerical tolerance.

### 3.1 SVD rotation

For an individual preshape \(Z\) and current consensus \(M\), form

\[
H=Z^\mathsf TM=U\Sigma V^\mathsf T.
\]

The unconstrained orthogonal solution is \(UV^\mathsf T\). Because the engine
prohibits reflection, it uses

\[
R=U\operatorname{diag}\!\left(1,
\det(UV^\mathsf T)\right)V^\mathsf T,
\qquad \det(R)=1.
\]

This is the proper rotation minimizing \(\|ZR-M\|_F^2\).

### 3.2 Convergence diagnostics

At iteration \(t\), convergence is declared when

\[
\|M^{(t+1)}-M^{(t)}\|_F<10^{-10}.
\]

The application reports the number of iterations, convergence status,
per-reference chord distances, and the generalized residual sum of squares

\[
G_\star=\sum_{i=1}^{n}\|Z_iR_i-\widehat M\|_F^2.
\]

## 4. Procrustes distances

After optimal rotational alignment, the partial Procrustes or chord distance is

\[
d_P(Z,M)=\|ZR-M\|_F.
\]

For unit preshapes with \(a=\langle \operatorname{vec}(ZR),
\operatorname{vec}(M)\rangle\), the engine also reports the full Procrustes
distance

\[
d_F(Z,M)=\sqrt{\max(0,1-a^2)}.
\]

Both are nonnegative descriptions of geometric separation. Neither is a
percentile, probability, classification, or value judgment.

## 5. Kendall tangent space

The centered unit preshapes occupy a sphere, and planar rotations define an
equivalence relation on that sphere. Consequently, the relevant shape space is
curved. Local multivariate methods operate in a tangent plane at the GPA
consensus \(M\).

For \(k\) planar landmarks, the shape-space dimension is

\[
2k-4=2(68)-4=132,
\]

because two translations, one uniform scale, and one rotation are removed.

### 5.1 Nuisance directions

In the ambient vector space \(\mathbb R^{136}\), the engine constructs four
nuisance vectors:

\[
t_x=(1,0,1,0,\ldots,1,0)^\mathsf T,
\]

\[
t_y=(0,1,0,1,\ldots,0,1)^\mathsf T,
\]

\[
s=\operatorname{vec}(M),
\]

and the infinitesimal planar-rotation direction

\[
q=\operatorname{vec}(JM),
\qquad
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
\]

Let \(B\in\mathbb R^{136\times132}\) be an orthonormal basis for the null
space of \([t_x,t_y,s,q]^\mathsf T\). Thus

\[
B^\mathsf TB=I_{132}.
\]

### 5.2 Central tangent projection

For aligned preshape \(Y=ZR\), define \(a=\langle
\operatorname{vec}(Y),\operatorname{vec}(M)\rangle\). The central projection
onto the tangent plane is

\[
v(Y)=\frac{\operatorname{vec}(Y)}{a}-\operatorname{vec}(M).
\]

The nonredundant tangent coordinates are

\[
z=B^\mathsf Tv(Y)\in\mathbb R^{132}.
\]

This is a local linear approximation. The engine rejects configurations with
near-zero \(a\), for which a tangent approximation at \(M\) would be
inappropriate.

## 6. Tangent-space principal components

Let \(z_1,\ldots,z_n\) be tangent coordinates for the aligned synthetic
reference sample and

\[
\bar z=\frac1n\sum_{i=1}^{n}z_i.
\]

The sample covariance is

\[
S=\frac{1}{n-1}\sum_{i=1}^{n}(z_i-\bar z)(z_i-\bar z)^\mathsf T.
\]

The symmetric eigendecomposition is

\[
S=V\Lambda V^\mathsf T,
\qquad
\lambda_1\ge\lambda_2\ge\cdots\ge0.
\]

For an analyzed configuration with tangent coordinate \(z\), its displayed PC
scores are

\[
s_j=v_j^\mathsf T(z-\bar z),
\]

and the explained-variance ratio of PC \(j\) is

\[
\rho_j=\frac{\lambda_j}{\sum_{\ell=1}^{132}\lambda_\ell}.
\]

The application reports the first ten scores and visualizes the first six.

## 7. Structured simulated covariance

The synthetic ensembles are generated from twelve smooth deformation modes
covering correlated changes in the jaw, brows, eyes, nose, lips, and bilateral
structure. Independent low-amplitude landmark noise is projected into the
tangent subspace. Random translation, rotation, and scale are then added as
nuisance transformations so that the GPA implementation must remove them.

This construction generates genuine cross-landmark covariance. It does not
claim to reproduce covariance from any biological population.

Because finite-sample covariance estimation can be unstable, the engine uses a
fixed diagonal-target shrinkage estimator:

\[
\widehat\Sigma_\lambda
=(1-\lambda)S+\lambda\operatorname{diag}(S)+\varepsilon I,
\]

where \(\lambda=0.20\) and

\[
\varepsilon
=10^{-6}\frac{\operatorname{tr}\left((1-\lambda)S+
\lambda\operatorname{diag}(S)\right)}{132},
\]

subject to a machine-precision lower bound. Shrinking toward
\(\operatorname{diag}(S)\)—rather than a scalar multiple of the identity—keeps
coordinate-specific variance while retaining \(80\%\) of the estimated
off-diagonal structure.

## 8. Regularized Mahalanobis distance

For tangent displacement \(\delta=z-\bar z\), the squared distance is

\[
D_M^2=\delta^\mathsf T\widehat\Sigma_\lambda^{-1}\delta.
\]

The inverse quadratic form is evaluated by Cholesky factorization and triangular
solution rather than by explicitly constructing a dense inverse. The reported
distance is

\[
D_M=\sqrt{\max(0,D_M^2)}.
\]

Because \(\widehat\Sigma_\lambda\) is simulated and its shrinkage parameter is
fixed, \(D_M\) has no empirical percentile or inferential calibration in this
application.

## 9. Residual shape-difference vectors

In aligned unit shape space, the descriptive residual at landmark \(j\) is

\[
r_j=\widehat M_j-Y_j.
\]

Its magnitude is \(\|r_j\|_2\), and the root mean square magnitude is

\[
r_{\mathrm{RMS}}
=\left(\frac1k\sum_{j=1}^{k}\|r_j\|_2^2\right)^{1/2}.
\]

The interface may magnify arrows by a display factor of \(1\), \(3\), or
\(6\). Magnification changes only the rendering. It does not change any
statistic. The vectors describe the coordinate difference between two aligned
configurations; they are not recommendations to alter an observed object.

## 10. Gradient identity and why optimization is omitted

For a fixed positive-definite covariance matrix, the gradient of squared
Mahalanobis distance is

\[
\nabla_zD_M^2=2\widehat\Sigma_\lambda^{-1}(z-\bar z).
\]

Minimizing this expression without a scientifically identified response model,
constraints, and explicit estimand sends every configuration toward the
reference mean. That operation is mathematically trivial and substantively
prescriptive. The application therefore computes the gradient identity only as
documentation and performs no gradient descent, gradient ascent, BFGS, or
coordinate optimization.

## 11. Splines and kernels

These methods are documented for future population-level research and are not
used in the present browser demonstration. The definitions follow the
statistical-learning treatment in Hastie, Tibshirani, and Friedman.

### 11.1 Smoothing splines

For observations \((x_i,y_i)\), a cubic smoothing spline estimates a function
\(f\) by minimizing

\[
\sum_{i=1}^{n}\left[y_i-f(x_i)\right]^2
+\lambda\int\left[f''(t)\right]^2dt.
\]

The first term measures lack of fit. The integrated squared second derivative
penalizes curvature, and \(\lambda\ge0\) controls the bias–variance tradeoff.
The solution is a natural cubic spline with knots at the distinct observed
values of \(x\).

For longitudinal landmark trajectories, the same construction could smooth a
tangent coordinate through time. Valid inference would still need to account
for repeated observations, landmark uncertainty, and selection of \(\lambda\).

### 11.2 Positive-semidefinite kernels

A kernel \(K:\mathcal X\times\mathcal X\to\mathbb R\) is positive
semidefinite when, for every finite collection \(x_1,\ldots,x_n\) and every
\(a\in\mathbb R^n\),

\[
\sum_{i=1}^{n}\sum_{j=1}^{n}a_i a_jK(x_i,x_j)\ge0.
\]

Such a kernel acts as an inner product in a reproducing-kernel Hilbert space:

\[
K(x,x')=\langle\phi(x),\phi(x')\rangle_{\mathcal H}.
\]

A radial basis kernel on tangent coordinates is

\[
K(z,z')=\exp\left(-\gamma\|z-z'\|_2^2\right),\qquad\gamma>0.
\]

Kernel regression or classification would require independently defined,
validated outcomes and out-of-sample assessment. A kernel does not create an
outcome definition by itself.

## 12. Density estimation on and near manifolds

For observations on an \(m\)-dimensional manifold embedded in
\(\mathbb R^D\), an intrinsic kernel-density estimator has the schematic form

\[
\widehat p_h(x)
=\frac{1}{nh^m}\sum_{i=1}^{n}
K\!\left(\frac{d_{\mathcal M}(x,X_i)}{h}\right),
\]

where \(d_{\mathcal M}\) is a manifold distance. The normalization depends on
intrinsic dimension \(m\), not ambient dimension \(D\).

Near a boundary, an ordinary symmetric kernel places part of its mass outside
the support and becomes biased. Boundary-direction estimation and
cut-and-normalize corrections address this problem when the boundary is
unknown. Bayesian mixtures near an unknown manifold instead represent the data
as lying in a noisy tube around an unknown lower-dimensional support and can
adapt to different smoothness along and perpendicular to that support.

The current application does not estimate a density. These methods define a
future extension that would require a real, de-identified sample and an
explicit sampling design.

## 13. Study-context metadata

The interface records three optional study-design annotations:

- Environmental Stress Index;
- Pathogen Prevalence Index;
- Operational Sex Ratio.

They are exported under `study_context_metadata` with
`analytic_role: "annotation_only"`. They do not enter GPA, covariance, PCA,
Mahalanobis distance, or residual calculations. No causal or associational
claim is made about them.

To analyze context scientifically, a future study would need a defined target
population, measurement protocol, sampling frame, repeated-rater structure,
predeclared estimand, hierarchical model, uncertainty propagation, and
out-of-sample validation.

## 14. Numerical safeguards

The engine applies the following safeguards:

- exactly 136 finite coordinate values are required;
- zero and near-zero centroid size are rejected;
- reflections are excluded from Procrustes alignment;
- the tangent basis is checked for dimension \(132\);
- near-orthogonal shapes are rejected from the local tangent approximation;
- covariance is symmetrized, shrunk, and ridge-regularized;
- Cholesky solution replaces explicit inversion;
- JSON serialization rejects NaN and infinity;
- deterministic seeds make every simulated reference reproducible.

## 15. Invariance and limitations

The reported shape statistics are invariant, up to floating-point error, to
global translation, uniform scale, and proper planar rotation of the input.
They are not invariant to landmark relabeling, missing landmarks, reflection,
nonuniform image distortion, perspective, expression, or measurement error.

The demonstration assumes exact one-to-one landmark correspondence. It does not
perform image landmark detection. Applying these methods to empirical data
requires documented landmark acquisition, reliability assessment, handling of
missingness, and ethical review appropriate to the study.

The complete annotated bibliography and implementation map appear in the
[repository README](../README.md).
