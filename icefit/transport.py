# Optimal Transport (Wasserstein) distance measures (tests under construction)
# 
# run tests with: pytest icefit/transport.py -rP -vv
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import pytest
import math

def quantile_function(qs: torch.Tensor, cumweights: torch.Tensor, values: torch.Tensor):
    """
    Computes the quantile function of an empirical distribution
    
    (handles also multiple independent columns, i.e. for vectorized SWD)
    
    Args:
        qs:         quantile positions where the quantile function is evaluated, (n,) or (n,2)
        cumweights: cumulative weights of the 1D empirical distribution, (n,) or (n,2)
        values:     locations of the 1D empirical distribution

    Returns:
        quantiles of the distribution
    """
    n = values.shape[0]
    
    if qs.ndim == 2:
        qs = qs.transpose(0,1).contiguous()
    if cumweights.ndim == 2:
        cumweights = cumweights.transpose(0,1).contiguous()

    idx = torch.searchsorted(cumweights, qs, right=False)

    if idx.ndim == 2:
        idx = idx.transpose(0,1)
    
    return torch.take_along_dim(values, torch.clip(idx, 0, n - 1), dim=0)


def wasserstein_distance_1D(u_values: torch.Tensor, v_values: torch.Tensor,
                            u_weights: torch.Tensor=None, v_weights: torch.Tensor=None,
                            p: int=1, inverse_power: bool=False, norm_weights: bool=True,
                            require_sort: bool=True):
    """
    Wasserstein 1D distance over two empirical samples.
    
    This function computes with quantile functions (not just CDFs as with special case p=1),
    thus compatible with arbitrary p.
    
    Args:
        u_values:       sample U vectors (n, [possibly multiple independent columns, for vectorized])
        v_values:       sample V vectors (m, [as above])
        u_weights:      sample U weights (n,) (if None, assumed to be unit)
        v_weights:      sample V weights (m,)
        p:              p-norm parameter (p = 1 is 'Earth Movers', 2 = is W-2, ...)     
        num_slices:     number of random MC projections (slices) (higher the better)
        inverse_power:  apply final inverse power 1/p
        norm_weights:   normalize per sample (U,V) weights to sum to one
        require_sort:   always by default, unless presorted
    
    Returns:
        distance between the empirical sample distributions
    """

    def zero_pad(a, pad_width, value=0):
        """
        Helper zero-padding function
        """
        how_pad = tuple(element for t in pad_width[::-1] for element in t)
        return torch.nn.functional.pad(a, how_pad, value=value)
    
    n = u_values.shape[0]
    m = v_values.shape[0]
    
    if u_weights is None:
        u_weights = torch.ones((n,), dtype=u_values.dtype, device=u_values.device)
    if v_weights is None:
        v_weights = torch.ones((m,), dtype=v_values.dtype, device=v_values.device)
    
    if norm_weights:
        u_weights = u_weights / torch.sum(u_weights)
        v_weights = v_weights / torch.sum(v_weights)
    
    if u_values.ndim == 2: # need (n,d) view, expand is memory friendly
        u_weights = u_weights.unsqueeze(1).expand(-1, u_values.size(-1))
        v_weights = v_weights.unsqueeze(1).expand(-1, v_values.size(-1))
    
    if require_sort:
        u_sorter  = torch.argsort(u_values, dim=0)
        u_values  = torch.take_along_dim(u_values, u_sorter, dim=0)
        u_weights = torch.take_along_dim(u_weights, u_sorter, dim=0)
        
        v_sorter  = torch.argsort(v_values, dim=0)
        v_values  = torch.take_along_dim(v_values, v_sorter, dim=0)
        v_weights = torch.take_along_dim(v_weights, v_sorter, dim=0)

    u_cumweights = torch.cumsum(u_weights, dim=0)
    v_cumweights = torch.cumsum(v_weights, dim=0)

    # Compute quantile functions
    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), dim=0), dim=0).values
    u_quantiles = quantile_function(qs=qs, cumweights=u_cumweights, values=u_values)
    v_quantiles = quantile_function(qs=qs, cumweights=v_cumweights, values=v_values)
    
    del u_cumweights, v_cumweights
    
    # Boundary conditions
    qs = zero_pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
    
    # Measure and integrand
    delta = qs[1:, ...] - qs[:-1, ...]
    del qs
    
    dq = torch.abs(u_quantiles - v_quantiles)
    del u_quantiles, v_quantiles
    
    if p == 1:
        return (delta * dq).sum(dim=0)
    
    if inverse_power:
        dq.pow_(p)
        return (delta * dq).sum(dim=0).pow(1.0 / p)
    else:
        dq.pow_(p)
        return (delta * dq).sum(dim=0)


@torch.no_grad()
def rand_projections(dim: int, N: int=1000, device: str='cpu', dtype=torch.float32):
    """
    Define N random projection directions on the unit sphere S^{dim-1}
    Normally distributed components and final normalization guarantee (Harm) uniformity.
    """
    projections = torch.randn((N, dim), dtype=dtype, device=device)
    return projections / torch.norm(projections, p=2, dim=1, keepdim=True)

@torch.no_grad()
def qr_projections(
    dim: int,
    N: int=1000,
    *,
    device: str = 'cpu',
    dtype = torch.float32,
    block_size: int | None = None,
    random_sign: bool = True,
):
    """
    Orthogonal blocks of random directions, with optional +-1 sign flip.
    """
    if block_size is None:
        block_size = dim

    n_blocks = math.ceil(N / block_size)
    rays     = []
    
    for _ in range(n_blocks):
        k = min(block_size, N - len(rays))               # last block may be smaller
        g = torch.randn((dim, k), device=device, dtype=dtype)
        q, _ = torch.linalg.qr(g, mode="reduced")        # (dim, k)

        if random_sign:                                  # independent sign per vector
            signs = (torch.randint(0, 2, (1, k), device=q.device) * 2 - 1).to(q.dtype)
            q = q * signs

        rays.append(q.T)                                 # store k vectors as rows

    return torch.cat(rays, dim=0)[:N]                    # (N, dim)

def sliced_wasserstein_distance(u_values: torch.Tensor, v_values: torch.Tensor,
                                u_weights: torch.Tensor=None, v_weights: torch.Tensor=None,
                                p: int=1, num_slices: int=1000, mode: str='SWD', 
                                norm_weights: bool=True,
                                vectorized: bool=True, inverse_power: bool=True):
    """
    Sliced Wasserstein Distance over arbitrary dimensional samples
    
    References:
        https://arxiv.org/abs/1902.00434
        https://arxiv.org/abs/2211.08775
        https://arxiv.org/abs/2304.13586
    
    Notes:
        When using this as a loss function e.g. with neural nets, large
        minibatch sizes may be beneficial or needed.
    
    Args:
        u_values:      sample U vectors (n x dim)
        v_values:      sample V vectors (m x dim)
        u_weights:     sample U weights (n,) (if None, assumed to be unit)
        v_weights:     sample V weights (m,)
        p:             p-norm parameter (p = 1 is 'Earth Movers', 2 = is W-2, ...)       
        num_slices:    number of random MC projections (slices) (higher the better)
        mode:          'SWD'  (basic uniform MC random)
                       'SWD-QR' (orthogonal random blocks, lower estimator variance)
        norm_weights:  normalize per sample (U,V) weights to sum to one
        vectorized:    fully vectorized (may take more GPU/CPU memory, but 10x faster)
        inverse_power: apply final inverse power
    
    Returns:
        distance between the empirical sample distributions
    """
    
    # Generate a random projection direction
    dim        = int(u_values.shape[-1])
    
    if   mode == 'SWD':
        directions = rand_projections(dim=dim, N=num_slices, device=u_values.device)
    elif mode == 'SWD-QR':
        directions = qr_projections(dim=dim, N=num_slices, device=u_values.device)
    else:
        raise Exception(f'Unknown mode parameter: {mode}')
    
    if vectorized:
        
        u_proj = torch.matmul(u_values, directions.T)
        v_proj = torch.matmul(v_values, directions.T)
        
        dist = wasserstein_distance_1D(u_values=u_proj, v_values=v_proj,
                    u_weights=u_weights, v_weights=v_weights, p=p,
                    norm_weights=norm_weights,
                    inverse_power=False)
        
    else:
        dist = torch.zeros(num_slices, device=u_values.device, dtype=u_values.dtype)
        
        for i in range(num_slices):
            
            # Project the distributions on the random direction
            u_proj = torch.matmul(u_values, directions[i,:])
            v_proj = torch.matmul(v_values, directions[i,:])
            
            # Calculate the 1-dim Wasserstein on the direction
            dist[i] = wasserstein_distance_1D(u_values=u_proj, v_values=v_proj,
                            u_weights=u_weights, v_weights=v_weights, p=p,
                            norm_weights=norm_weights,
                            inverse_power=False)
    
    # Average over
    dist = torch.sum(dist) / num_slices
    
    if inverse_power:
        return dist ** (1.0 / p)
    else:
        return dist


def test_1D(EPS=1e-3):
    """
    Test function (fixed reference checked against scikit-learn)
    """
    from scipy.stats import wasserstein_distance
    import numpy as np
    import ot
    
    # -----------------------------------------------
    ## p = 1 case checked against scikit-learn
    
    p = 1
    
    u = torch.tensor([0.0, 1.0, 3.0], requires_grad=True)
    v = torch.tensor([5.0, 6.0, 8.0], requires_grad=True)
    
    res = wasserstein_distance_1D(u, v, p=p)
    res_scikit = wasserstein_distance(u.detach().numpy(), v.detach().numpy())
    
    print(f'1D case 1: p = 1 | {res.item()} {res_scikit}')
    assert res.item() == pytest.approx(res_scikit, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([0., 1.]), torch.tensor([0., 1.]),
                                  torch.tensor([3., 1.]), torch.tensor([2., 2.]), p=p)
    
    res_scikit = wasserstein_distance(np.array([0., 1.]), np.array([0, 1.]),
                                      np.array([3., 1.]), np.array([2, 2.]))
    
    print(f'1D case 2: p = 1 | {res.item()} {res_scikit}')
    assert res.item() == pytest.approx(res_scikit, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([3.4, 3.9, 7.5, 7.8]), torch.tensor([4.5, 1.4]),
                                  torch.tensor([1.4, 0.9, 3.1, 7.2]), torch.tensor([3.2, 3.5]))
    
    res_scikit = wasserstein_distance(np.array([3.4, 3.9, 7.5, 7.8]),     np.array([4.5, 1.4]),
                                      np.array([1.4, 0.9, 3.1, 7.2]),     np.array([3.2, 3.5]))
    
    print(f'1D case 3: p = 1 | res = {res.item()} res_scikit = {res_scikit}')
    assert res.item() == pytest.approx(res_scikit, abs=EPS)
    
    # -----------------------------------------------
    ## p = 1,2 against POT
    
    u_values  = torch.tensor([1.0, 2.0], requires_grad=True)
    v_values  = torch.tensor([3.0, 4.0], requires_grad=True)
    
    u_weights = torch.tensor([0.3, 1.0])
    v_weights = torch.tensor([1.0, 0.5])
    
    # pot library does not normalize
    norm = False
    
    for p in [1,2]:
        
        res = wasserstein_distance_1D(u_values, v_values, u_weights, v_weights, p=p, norm_weights=False)
        pot = ot.wasserstein_1d(u_values, v_values, u_weights, v_weights, p=p)
        
        print(f'1D case 3: p = {p} | res = {res.item()} pot = {pot}')
        
        res_grad = torch.autograd.grad(res, u_values)
        pot_grad = torch.autograd.grad(pot, u_values)
        
        print(f'gradient dW/du:       {res_grad}')
        print(f'gradient dW/du (POT): {pot_grad}')
        
        assert res.detach() == pytest.approx(pot.detach(), abs=EPS)
        assert torch.allclose(res.detach(), pot.detach(), atol=EPS)
    
    # -----------------------------------------------
    ## p = 2 case checked against pre-computed
    
    p = 2
    
    res = wasserstein_distance_1D(torch.tensor([0, 1, 3]), torch.tensor([5, 6, 8]), p=p).item()
    assert res == pytest.approx(25.0, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([0, 1]), torch.tensor([0, 1]),
                                  torch.tensor([3, 1]), torch.tensor([2, 2]), p=p).item()
    assert res == pytest.approx(0.25, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([3.4, 3.9, 7.5, 7.8]), torch.tensor([4.5, 1.4]),
                                  torch.tensor([1.4, 0.9, 3.1, 7.2]), torch.tensor([3.2, 3.5]), p=p).item()
    assert res == pytest.approx(19.09, abs=EPS)


def test_swd():
    """
    Test sliced Wasserstein distance
    """
    
    import ot # POT for reference
    
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        #np.random.seed(seed)  # Numpy module.
        #random.seed(seed)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    from time import time
    
    seed = 42
    
    # ---------------------------------------------------------
    # 2D test
    
    n = 20  # number of samples
    
    # Mean vectors
    mu_u = torch.tensor([0, 0], dtype=torch.float32)
    mu_v = torch.tensor([4, 0], dtype=torch.float32)
    
    # Covariance matrices
    cov_u = torch.tensor([[1.0,    0],
                          [0,    1.0]], dtype=torch.float32)
    cov_v = torch.tensor([[1.0, -0.3],
                          [-0.3, 1.0]], dtype=torch.float32)

    # Function to generate random vectors
    def generate_random_vectors(mu, cov, num_samples):
        distribution = torch.distributions.MultivariateNormal(mu, cov)
        samples = distribution.sample((num_samples,))
        return samples
    
    # Generate random vectors for both distributions
    set_seed(seed)
    u_values = generate_random_vectors(mu_u, cov_u, n).requires_grad_(True)
    v_values = generate_random_vectors(mu_v, cov_v, n).requires_grad_(True)
    
    print(u_values.shape)
    print(v_values.shape)

    
    # ------------------------------------------------------------
    # Versus POT
    
    num_slices = 200
    
    #exact = wasserstein2_distance_gaussian(mu1=mu_u, cov1=cov_u, mu2=mu_v, cov2=cov_v)  
    
    for p in [1,2]:
        
        pot = ot.sliced_wasserstein_distance(X_s=u_values, X_t=v_values,
                                             a=torch.ones(n)/n, b=torch.ones(n)/n,
                                             n_projections=num_slices, p=p)
        
        print(f'gradient dW/du (POT): {torch.autograd.grad(pot, u_values)}')
        
        for mode in ['SWD', 'SWD-QR']:
            for vectorized in [False, True]:
                
                set_seed(seed)
                res = sliced_wasserstein_distance(u_values=u_values, v_values=v_values, p=p,
                                    num_slices=num_slices, mode=mode, vectorized=vectorized)
                print(f'p = {p}: case 2 [{mode}] | res = {res.item()} pot = {pot} (vectorized = {vectorized})')
                print(f'gradient dW/du: {torch.autograd.grad(res, u_values)}')

                assert res.detach() == pytest.approx(pot.detach(), abs=0.3)
        
    # ---------------------------------------------------------
    # Fixed values 1D test
    
    u_values  = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    u_weights = torch.tensor([0.2, 0.5, 0.3]) # event weights
    v_values  = torch.tensor([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    v_weights = torch.tensor([0.3, 0.4, 0.3]) # event weights
    
    p = 2
    num_slices = 200
    
    for mode in ['SWD', 'SWD-QR']:
        
        set_seed(seed)
        res = sliced_wasserstein_distance(u_values=u_values, v_values=v_values,
                                        u_weights=u_weights, v_weights=v_weights,
                                        p=p, num_slices=num_slices, mode=mode).item()
        print(res)
        assert res == pytest.approx(0.683, abs=0.05)
        
    # -------------------------------------------------------
    # Test vectorized versus non-vectorized slicing implementation

    seed = 42
    set_seed(seed)
    
    u_values = torch.randn(100, 4)
    v_values = torch.randn(150, 4)
    
    u_weights = None
    v_weights = None
    
    for p in [1,2]:
        for mode in ['SWD', 'SWD-QR']:
            
            # 'vectorized'
            set_seed(seed)
            tic = time()
            d     = sliced_wasserstein_distance(u_values=u_values, v_values=v_values,
                        u_weights=u_weights, v_weights=v_weights,
                        p=p, num_slices=num_slices, mode=mode, vectorized=True).item()
            toc = time() - tic
            
            # 'non-vectorized'
            set_seed(seed)
            tic_alt = time()
            d_alt = sliced_wasserstein_distance(u_values=u_values, v_values=v_values,
                        u_weights=u_weights, v_weights=v_weights,
                        p=p, num_slices=num_slices, mode=mode, vectorized=False).item()
            toc_alt = time() - tic_alt
            
            print(f'p = {p} ({mode}) || D = {d} (vectorized, {toc:0.2e} sec) | D = {d_alt} (non-vectorized, {toc_alt:0.2e} sec)')
            
            assert d == pytest.approx(d_alt, abs=1e-4)

# ---------------------------------------------------------------------
# Advanced SWD test: value, variance, runtime, memory and gradients
# ---------------------------------------------------------------------
def test_swd_advanced():
    """
    For each (dim, test-case, #rays) the routine compares two samplers
    
        - SWD:    i.i.d. Gaussian directions
        - SWD-QR: QR-orthogonal blocks
    
    Metrics reported:
        
        - Monte-Carlo mean of SWD
        - Variance across 100 replicates
        - Relative mean-absolute error (RMAE) against a high-ray POT ref
        - Mean relative gradient error vs. POT gradient
        - Peak memory during a forward/backward pass
        - Average wall-time per replicate
    """
    
    import ot # POT for reference
    import psutil
    import os
    import time
    import gc

    def set_seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

    def sample_mog(n: int, dim: int, n_components: int = 3):
        """Mixture-of-Gaussians with random means/covariances."""
        
        # draw mixing weights that sum to 1
        weights = torch.distributions.Dirichlet(torch.ones(n_components)).sample()
        comps   = torch.multinomial(weights, n, replacement=True) # (n,)

        means = torch.randn(n_components, dim) * 2.0
        covs  = []
        for _ in range(n_components):
            A = torch.randn(dim, dim)
            covs.append((A @ A.T) / dim + 0.2 * torch.eye(dim))
        covs = torch.stack(covs) # (k, d, d)

        out = torch.empty(n, dim)
        for k in range(n_components):
            idx = (comps == k).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            mvn = torch.distributions.MultivariateNormal(means[k], covs[k])
            out[idx] = mvn.sample((idx.numel(),))
        
        return out

    def sample_ring(n: int, dim: int, radius=4.0, noise=0.2):
        """d-dimensional ring/donut distribution."""
        
        theta  = torch.rand(n) * 2 * math.pi
        base   = torch.stack((radius * torch.cos(theta),
                            radius * torch.sin(theta)), dim=1)
        if dim > 2:
            extra = torch.randn(n, dim - 2) * radius * 0.1
            base  = torch.cat((base, extra), dim=1)
        
        return base + noise * torch.randn_like(base)

    seed            = 123
    ray_grid        = (1, 10, 100, 1000)
    reps            = 100
    n_samples       = 2048
    nrays_ref       = int(1e5)          # POT reference rays
    rtol_grad       = 0.30
    atol_grad       = 1e-4
    device          = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -----------------------------------------------------------------
    # helper to draw two independent clouds with a deterministic seed
    # -----------------------------------------------------------------
    def _draw_pair(f1, f2, *, base_seed):
        set_seed(base_seed)
        X = f1(n_samples, dim).to(device)
        set_seed(base_seed + 999)
        Y = f2(n_samples, dim).to(device)
        return X, Y

    test_cases = [
        ('mog_vs_ring', lambda bs: _draw_pair(sample_mog,  sample_ring, base_seed=bs)),
        ('mog_vs_mog',  lambda bs: _draw_pair(sample_mog,  sample_mog,  base_seed=bs)),
    ]

    for dim in [2, 4, 16, 64]:
        print('\n' + '=' * 70)
        print(f'** Dimension: {dim} **')

        for case_name, make_xy in test_cases:
            print('\n' + '-' * 70)
            print(f'Test case: {case_name}')

            # ----------------------- data --------------------------------
            X_raw, Y_raw = make_xy(seed)

            # POT reference value (no grad needed)
            with torch.no_grad():
                set_seed(seed)
                exact_val = ot.sliced_wasserstein_distance(
                    X_s=X_raw, X_t=Y_raw, a=None, b=None,
                    n_projections=nrays_ref, p=2
                )
            print(f'  POT reference value ({nrays_ref:0.0E} rays): {exact_val:.4f}')

            # POT reference gradient (needs requires_grad)
            set_seed(seed)
            X_ref = X_raw.clone().detach().requires_grad_(True)
            Y_ref = Y_raw.clone().detach()
            pot_val = ot.sliced_wasserstein_distance(
                X_s=X_ref, X_t=Y_ref, n_projections=nrays_ref, p=2
            )
            pot_val.backward()
            grad_ref = X_ref.grad.detach()
            ref_norm = grad_ref.norm().clamp_min(1e-12)

            # ---------------------- table header -------------------------
            header = (f"{'rays':>6} | {'sampler':>12} | {'mean':>10} | "
                      f"{'var':>10} | {'RMAE':>10} | {'grad err':>10} | "
                      f"{'peak MB':>8} | {'ms/rep':>8}")
            print('\n' + header)
            print('-' * len(header))

            # Loop over #slices and samplers
            for num_slices in ray_grid:
                results = {}
                
                for sampler in ('SWD', 'SWD-QR'):
                    # ---------------------------------------------------
                    # (1) PEAK-MEM measurement, run a few fwd/bwd passes
                    # ---------------------------------------------------
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device)
                        start_mem = torch.cuda.memory_allocated(device)
                    else:
                        process   = psutil.Process(os.getpid())
                        gc.collect()
                        start_mem = process.memory_info().rss

                    for _ in range(5):
                        set_seed(seed)
                        X_ = X_raw.clone().detach().requires_grad_(True)
                        Y_ = Y_raw.clone().detach()
                        loss = sliced_wasserstein_distance(
                            X_, Y_, p=2,
                            num_slices=num_slices,
                            mode=sampler, vectorized=True
                        )
                        loss.backward()

                    if device.startswith('cuda'):
                        peak_mb = (torch.cuda.max_memory_allocated(device) - start_mem) / 1024**2
                    else:
                        gc.collect()
                        peak_mb = (psutil.Process(os.getpid()).memory_info().rss - start_mem) / 1024**2

                    # ---------------------------------------------------
                    # (2) Gradient error (single pass, same seed)
                    # ---------------------------------------------------
                    set_seed(seed)
                    X_grad = X_raw.clone().detach().requires_grad_(True)
                    Y_grad = Y_raw.clone().detach()
                    g_loss = sliced_wasserstein_distance(
                        X_grad, Y_grad, p=2,
                        num_slices=num_slices,
                        mode=sampler, vectorized=True
                    )
                    g_loss.backward()
                    grad_err = (X_grad.grad - grad_ref).norm() / ref_norm

                    # ---------------------------------------------------
                    # (3) Monte-Carlo replicates for mean / var / RMAE
                    # ---------------------------------------------------
                    vals = torch.empty(reps, device=device)
                    tic  = time.perf_counter()
                    with torch.no_grad():
                        for r in range(reps):
                            set_seed(seed + r)
                            vals[r] = sliced_wasserstein_distance(
                                X_raw, Y_raw,
                                p=2,
                                num_slices=num_slices,
                                mode=sampler,
                                vectorized=True,
                            )
                    elapsed = (time.perf_counter() - tic) * 1e3 / reps  # ms/rep

                    results[sampler] = dict(
                        mean      = vals.mean().cpu(),
                        var       = vals.var(unbiased=True).cpu(),
                        rmae      = ((vals - exact_val).abs().mean() / exact_val).cpu(),
                        grad_err  = grad_err.cpu(),
                        peak_mb   = peak_mb,
                        elapsed   = elapsed,
                    )

                    print(f"{num_slices:6d} | {sampler:12s} | "
                          f"{results[sampler]['mean']:10.2e} | "
                          f"{results[sampler]['var']:10.2e} | "
                          f"{results[sampler]['rmae']:10.2e} | "
                          f"{results[sampler]['grad_err']:10.2e} | "
                          f"{results[sampler]['peak_mb']:8.2f} | "
                          f"{results[sampler]['elapsed']:8.2f}")
