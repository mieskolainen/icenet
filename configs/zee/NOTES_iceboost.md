# iceboost: Hessian modes with Torch autograd

m.mieskolainen@imperial.ac.uk, 2025

`icenet/deep/iceboostgrad.py` <br>
`icenet/deep/iceboost.py`

## Basics

Let the model raw output prediction scalar be $f_i$, scalar labels $y_i$, weights $w_i$, and loss $L(f)$.

The gradient vector entries and Hessian (diagonal) of the loss with respect to $f$ are:

$$
g_i = \frac{dL(f)}{df_i}, \quad h_i = \frac{d^2 L(f)}{df_i^2},
$$

where each $i$ corresponds to a single training data entry (event) in gradient boosting. Now xgboost computes the optimal weight for a leaf $j$ over its associated instance set $I_j$ (region of feature space) with $L_2$ damping parameter $\lambda > 0$ by:

$$
w_j^{\star} = - \frac{ \sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}.
$$

For more information, see: https://arxiv.org/abs/1603.02754

**Example**: a weighted BCE loss function

$$
L(f) = -\sum_i w_i[y_i \log(p_i) + (1-y_i) \log(1-p_i)],
$$

where

$$
p_i = \text{sigmoid}(f_i) = \frac{1}{1 + e^{-f_i}},
$$

with $f_i$ as the raw logit. This loss has analytical, simple gradient and Hessian (diagonal) vectors:

$$
g = w \odot (p - y), \quad h = w \odot p \odot (1 - p),
$$

where $\odot$ denotes element wise product. We could simply encode these by hand. However, for arbitrary loss functions, `iceboost` uses the full `torch` autograd machinery. Note that negative weights explicitly mess up $h$, which should be positive in Newton (second order gradient descent) updates.

While the gradient is relatively cheap to obtain with autograd, the exact Hessian diagonal (curvature) is not. The following different options are provided.

## Mode: Constant Hessian

**Specifier**: `:hessian:constant:c`, with a constant $c > 0$.

$$
\hat{h} = c.
$$

**Pros**: Newton step becomes a scaled gradient step (very conservative). No extra computational cost. Robust baseline.

**Cons**: Convergence is slow (many trees needed), and can make the model underfit.

## Mode: Iterative finite-difference with EMA

**Specifier**: `:hessian:iterative:beta`, with EMA parameter $0 < \beta < 1$.

Finite-difference curvature between boosting steps $t-1$ and $t$:

$$
\hat{h}^{(t), FD} = \frac{g^{(t)} - g^{(t-1)}} {f^{(t)} - f^{(t-1)} + \epsilon},
$$

where $\epsilon$ is a regularizer and the division is elementwise.

Then, exponential moving average (EMA) is applied for smoothing:

$$
\hat{h}^{(t)} = \beta \hat{h}^{(t-1)} + (1 - \beta) \hat{h}^{(t), FD}.
$$

**Pros**: Smoother than per-iteration raw finite difference; very cheap computationally.

**Cons**: Biased (time lag) and can be noisy, needs previous gradient and prediction vectors stored.

## Mode: Stochastic Hutchinson estimator with autograd

**Specifier**: `:hessian:hutchinson:m`, with $m \geq 1$ probe slice vectors.

Draw $m$ random Rademacher vectors $v^{(k)} \in \{-1, +1\}^N \; \text{ i.i.d. with } \mathbb{E}[v_i^{(k)}]=0,\ \mathbb{E}[v_i^{(k)} v_j^{(k)}]=\delta_{ij}$, and $N$ is the number of data samples.

Per probe vector, one Hessian-vector product via autograd is obtained by applying autograd a second time to the gradient:

$$
H v^{(k)} = \nabla_f( g^T v^{(k)} ), \quad \text{with} \quad g = \nabla_f L(f).
$$

The estimate is obtained as the average over $m$ probe vectors $v^{(k)}$:

$$
\hat{h} = \frac{1}{m} \sum_{k=1}^m v^{(k)} \odot (H v^{(k)}),
$$

where $\odot$ denotes elementwise product. This gives the Hutchinson estimate

$$
\mathbb{E}[ v \odot (H v) ] = \text{diag}(H).
$$

**Pros**: Unbiased and variance decreases $\propto 1/m$. Computational cost scales linearly with $O(m)$ (number of probes).

**Cons**: Slow decrease of variance can be a problem.

## Mode: Exact result with full autograd

**Specifier**: `:hessian:exact`

Exact diagonal components obtained via basis-vector Hessian products:

$$
h_i = e_i^T H e_i = [ H e_i ]_i,
$$

where

$$
\quad H e_i = \nabla_f( g^T e_i ), \quad \text{with} \quad g = \nabla_f L(f).
$$

and $e_i$ is the standard $N$-dimensional basis vector. Thus we need to run autograd $N$ times again on the gradient, which we obtained with a single autograd call. No known sublinear exact autograd algorithm exists (or is known), unless the loss has special structure. Only `vmap` batching is possible (memory bounded), but the asymptotic scaling does not change.


**Pros**: Exact result.

**Cons**: Computational cost scales linearly with $O(N)$ (dataset size), which is prohibitively slow with complex loss functions.

## Numerical safeguard (all modes)

For stability, a positivity clamp with a minimum Hessian value is applied to protect against negative curvature:

$$
\hat{h} \leftarrow \text{clip}( |\hat{h}|, h_{min}, h_{max}), \quad \text{with} \quad h_{min} > 0.
$$

Negative Hessian values would make the Newton update ill-posed, if left unprotected. Negative curvature will happen e.g. if negative event weights are utilized.
