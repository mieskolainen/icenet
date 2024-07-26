# Optimal Transport (Wasserstein) distance measures (tests under construction)
# 
# run tests with: pytest icefit/transport.py -rP -vv
#
# m.mieskolainen@imperial.ac.uk, 2024

import torch
import pytest

def quantile_function(qs: torch.Tensor, cumweights: torch.Tensor, values: torch.Tensor):
    """
    Computes the quantile function of an empirical distribution
    
    Args:
        qs:         quantile positions where the quantile function is evaluated
        cumweights: cumulative weights of the 1D empirical distribution
        values:     locations of the 1D empirical distribution

    Returns:
        quantiles of the distribution
    """
    n   = values.shape[0]
    qs         = qs.T.contiguous()
    cumweights = cumweights.T.contiguous()
    
    idx = torch.searchsorted(cumweights, qs).T.contiguous()
    
    return torch.take_along_dim(values, torch.clip(idx, 0, n - 1), dim=0)


def wasserstein_distance_1D(u_values: torch.Tensor, v_values: torch.Tensor,
                            u_weights: torch.Tensor=None, v_weights: torch.Tensor=None,
                            p: int=1, inverse_power: bool=False, normalize_weights: bool=True,
                            require_sort: bool=True):
    """
    Wasserstein distance over two empirical samples.
    
    This function computes with quantile functions (not just CDFs as with special case p=1),
    thus compatible with arbitrary p.
    
    Args:
        u_values:          sample U vectors (n x dim)
        v_values:          sample V vectors (m x dim)
        u_weights:         sample U weights (n,) (if None, assumed to be unit)
        v_weights:         sample V weights (m,)
        p:                 p-norm parameter (p = 1 is 'Earth Movers', 2 = is W-2, ...)     
        num_slices:        number of random MC projections (slices) (higher the better)
        inverse_power:     apply final inverse power 1/p
        normalize_weights: normalize per sample (U,V) weights to sum to one  
        require_sort:      always by default, unless presorted
    
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
    
    if normalize_weights and u_weights is not None:
        u_weights = u_weights / torch.sum(u_weights)
    if normalize_weights and v_weights is not None:
        v_weights = v_weights / torch.sum(v_weights)
    
    if u_weights is None:
        u_weights = torch.ones_like(u_values) / n
    elif u_weights.ndim != u_values.ndim:
        u_weights = u_weights[..., None].repeat((1,)*u_weights.ndim + (u_values.shape[-1],))
    
    if v_weights is None:
        v_weights = torch.ones_like(v_values) / m
    elif v_weights.ndim != v_values.ndim:
        v_weights = v_weights[..., None].repeat((1,)*v_weights.ndim + (v_values.shape[-1],))

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
    
    # Boundary conditions
    qs = zero_pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
    
    # Measure and integrand
    delta = qs[1:, ...] - qs[:-1, ...]
    dq = torch.abs(u_quantiles - v_quantiles)

    if p == 1:
        return torch.sum(delta * dq, dim=0)
    
    if inverse_power:
        return torch.sum(delta * torch.pow(dq, p), dim=0)**(1.0 / p)
    else:
        return torch.sum(delta * torch.pow(dq, p), dim=0)


def rand_projections(dim: int, N: int=1000, device: str='cpu', dtype=torch.float32):
    """
    Define N random projection directions on the unit sphere S^{dim-1}
    
    Normally distributed components and final normalization guarantee uniformity.
    """
    projections = torch.randn((N, dim), dtype=dtype, device=device)
    return projections / torch.norm(projections, p=2, dim=1, keepdim=True)


def sliced_wasserstein_distance(u_values: torch.Tensor, v_values: torch.Tensor,
                                u_weights: torch.Tensor=None, v_weights: torch.Tensor=None,
                                p: int=1, num_slices: int=1000, mode: str='SWD', 
                                vectorized: bool=True, inverse_power: bool=True):
    """
    Sliced Wasserstein Distance over arbitrary dimensional samples
    
    References:
        https://arxiv.org/abs/1902.00434
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
        vectorized:    fully vectorized (may take more GPU/CPU memory, but 10x faster)
        inverse_power: apply final inverse power
        
    Returns:
        distance between the empirical sample distributions
    """
    
    # Generate a random projection direction
    dim        = int(u_values.shape[-1])
    directions = rand_projections(dim=dim, N=num_slices, device=u_values.device)
    
    if vectorized:
        
        u_proj = torch.matmul(u_values, directions.T)
        v_proj = torch.matmul(v_values, directions.T)
        
        dist = wasserstein_distance_1D(u_values=u_proj, v_values=v_proj,
                    u_weights=u_weights, v_weights=v_weights, p=p, inverse_power=False)
        
    else:
        dist = torch.zeros(num_slices, device=u_values.device, dtype=u_values.dtype)
        
        for i in range(num_slices):
            
            # Project the distributions on the random direction
            u_proj = torch.matmul(u_values, directions[i,:])
            v_proj = torch.matmul(v_values, directions[i,:])
            
            # Calculate the 1-dim Wasserstein on the direction
            dist[i] = wasserstein_distance_1D(u_values=u_proj, v_values=v_proj,
                            u_weights=u_weights, v_weights=v_weights, p=p, inverse_power=False)
    
    if   mode == 'SWD':  # Standard
        dist = torch.sum(dist) / num_slices
    
    else:
        raise Exception(__name__ + f'.sliced_wasserstein_distance: Unknown "mode" chosen')
    
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
    
    # -----------------------------------------------
    ## p = 1 case checked against scikit-learn
    
    p = 1
    
    res = wasserstein_distance_1D(torch.tensor([0, 1, 3]), torch.tensor([5, 6, 8]), p=p).item()
    res_scikit = wasserstein_distance(np.array([0, 1, 3]),     np.array([5, 6, 8]))
    
    print(f'1D case 1: p = 1 | {res} {res_scikit}')
    assert res == pytest.approx(res_scikit, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([0, 1]), torch.tensor([0, 1]),
                                  torch.tensor([3, 1]), torch.tensor([2, 2]), p=p).item()
    res_scikit = wasserstein_distance(np.array([0, 1]),     np.array([0, 1]),
                                      np.array([3, 1]),     np.array([2, 2]))
    
    print(f'1D case 2: p = 1 | {res} {res_scikit}')
    assert res == pytest.approx(res_scikit, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([3.4, 3.9, 7.5, 7.8]), torch.tensor([4.5, 1.4]),
                                  torch.tensor([1.4, 0.9, 3.1, 7.2]), torch.tensor([3.2, 3.5])).item()
    res_scikit = wasserstein_distance(np.array([3.4, 3.9, 7.5, 7.8]),     np.array([4.5, 1.4]),
                                      np.array([1.4, 0.9, 3.1, 7.2]),     np.array([3.2, 3.5]))
    
    print(f'1D case 3: p = 1 | res = {res} res_scikit = {res_scikit}')
    assert res == pytest.approx(res_scikit, abs=EPS)
    
    # -----------------------------------------------
    ## p = 1,2 against POT
    
    import ot    
    
    u_values = torch.tensor([1.0, 2.0])
    v_values = torch.tensor([3.0, 4.0])
    
    u_weights = torch.tensor([0.3, 1.0])
    v_weights = torch.tensor([1.0, 0.5])
    
    # pot library does not normalize, so do it here
    u_weights /= torch.sum(u_weights)
    v_weights /= torch.sum(v_weights)
    
    for p in [1,2]:
        
        res = wasserstein_distance_1D(u_values, v_values, u_weights, v_weights, p=p).item()
        pot = ot.wasserstein_1d(u_values, v_values, u_weights, v_weights, p=p)
        
        print(f'1D case 3: p = {p} | res = {res} pot = {pot}')
        assert res == pytest.approx(pot, abs=EPS)

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
    
    n = 200  # number of samples
    
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
    u_values = generate_random_vectors(mu_u, cov_u, n)
    v_values = generate_random_vectors(mu_v, cov_v, n)
    
    print(u_values.shape)
    print(v_values.shape)

    
    # ------------------------------------------------------------
    # Versus POT
    
    import ot
    
    num_slices = 200
    
    #exact = wasserstein2_distance_gaussian(mu1=mu_u, cov1=cov_u, mu2=mu_v, cov2=cov_v)  
    
    for p in [1,2]:
        
        pot = ot.sliced_wasserstein_distance(X_s=u_values, X_t=v_values,
                                             a=torch.ones(n)/n, b=torch.ones(n)/n,
                                             n_projections=num_slices, p=p)
        
        for vectorized in [False, True]:
            
            # 'SWD'
            set_seed(seed)
            res = sliced_wasserstein_distance(u_values=u_values, v_values=v_values, p=p,
                                num_slices=num_slices, mode='SWD', vectorized=vectorized).item()
            print(f'p = {p}: case 2 SWD | res = {res} pot = {pot} (vectorized = {vectorized})')
            assert res == pytest.approx(pot, abs=0.3)
    
    # ---------------------------------------------------------
    # Fixed values 1D test
    
    u_values  = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    u_weights = torch.tensor([0.2, 0.5, 0.3]) # event weights
    v_values  = torch.tensor([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    v_weights = torch.tensor([0.3, 0.4, 0.3]) # event weights
    
    p = 2
    num_slices = 200
    
    # 'SWD'
    set_seed(seed)
    res = sliced_wasserstein_distance(u_values=u_values, v_values=v_values,
                                      u_weights=u_weights, v_weights=v_weights,
                                      p=p, num_slices=num_slices, mode='SWD').item()
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
        for mode in ['SWD']:
            
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
