# Optimal Transport (Wasserstein) distance measures
# 
# m.mieskolainen@imperial.ac.uk, 2024

import torch
import pytest

def wasserstein_distance_1D(u_values: torch.Tensor, v_values: torch.Tensor,
                            u_weights: torch.Tensor=None, v_weights: torch.Tensor=None,
                            p: int=1, apply_oop=True):
    """
    1D Wasserstein distance via sort-CDF
    
    (a torch equivalent implementation of scipy.stats.wasserstein_distance)
    
    Args:
        u_values:   samples in U (n x dim)
        v_values:   samples in V (m x dim)
        u_weights:  sample U weights (n,) (if none, assumed to be unity)
        v_weights:  sample V weights (m,)
        p:          p-norm parameter (p = 1 is 'Earth Movers', 2 = is W-2, ...)
        apply_oop:  apply final 1/p   
    
    Returns:
        distance between the empirical sample distributions
    """
    if u_weights is not None:
        assert u_values.size(0) == u_weights.size(0), "wasserstein_distance_1D: u_values and u_weights must be the same size."
    
    if v_weights is not None:
        assert v_values.size(0) == v_weights.size(0), "wasserstein_distance_1D: v_values and v_weights must be the same size."
    
    # Sort values
    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)

    all_values    = torch.cat((u_values, v_values))
    all_values, _ = torch.sort(all_values)

    # Compute the differences between pairs of successive values
    deltas = torch.diff(all_values)

    # Get the respective positions of the values of u and v among the values of both distributions
    u_cdf_indices = torch.searchsorted(u_values[u_sorter], all_values[:-1], right=True)
    v_cdf_indices = torch.searchsorted(v_values[v_sorter], all_values[:-1], right=True)

    # Calculate the CDFs of u and v using their weights, if specified
    if u_weights is None:
        u_cdf = u_cdf_indices.float() / u_values.size(0)
    else:
        zero = torch.tensor([0.0], dtype=u_values.dtype, device=u_values.device)
        rest = torch.cumsum(u_weights[u_sorter], dim=0)
        
        u_sorted_cumweights = torch.cat((zero, rest))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1] # normalize

    if v_weights is None:
        v_cdf = v_cdf_indices.float() / v_values.size(0)
    else:
        zero = torch.tensor([0.0], dtype=v_values.dtype, device=v_values.device)
        rest = torch.cumsum(v_weights[v_sorter], dim=0)
        
        v_sorted_cumweights = torch.cat((zero, rest))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1] # normalize

    # Compute the value of the integral based on the CDFs
    if p == 1:
        integral = torch.sum(torch.abs(u_cdf - v_cdf) * deltas)
    elif p == 2:
        integral = torch.sum((u_cdf - v_cdf).pow(2) * deltas)
    else:
        integral = torch.sum(torch.abs(u_cdf - v_cdf).pow(p) * deltas)

    if apply_oop:
        if p == 1:
            return integral
        if p == 2:
            return torch.sqrt(integral)
        else:
            return torch.pow(integral, 1.0 / p)
    
    # For sliced (multidim) version 1/p is applied after the average
    else:
        return integral

def rand_projections(dim: int, N: int=1000, device: str='cpu', dtype=torch.float32):
    """
    Define N random projection directions on the unit sphere S^{dim-1}
    """
    projections = torch.randn((N, dim), dtype=dtype, device=device)

    return projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))

def sliced_W_vectorized(u_values: torch.Tensor, v_values: torch.Tensor,
                        num_slices: int=2000, p: int=1,
                        u_weights: torch.Tensor=None, v_weights: torch.Tensor=None,
                        directions: torch.Tensor=None):
    """
    Helper function for 'sliced_wasserstein_distance'
    """
    
    device = u_values.device
    
    # Get dimensions
    n_samples_u, dim = u_values.shape
    n_samples_v = v_values.shape[0]
    
    # Generate random projections
    if directions is None:
        directions = rand_projections(dim=dim, num_slices=num_slices, device=device)
    
    # Project the data
    u_proj = u_values @ directions.T
    v_proj = v_values @ directions.T
    
    # Prepare weights
    if u_weights is None:
        u_weights = torch.ones(n_samples_u, device=device) / n_samples_u
    else:
        u_weights = torch.as_tensor(u_weights, dtype=torch.float32, device=device)
        u_weights /= u_weights.sum()
    
    if v_weights is None:
        v_weights = torch.ones(n_samples_v, device=device) / n_samples_v
    else:
        v_weights = torch.as_tensor(v_weights, dtype=torch.float32, device=device)
        v_weights /= v_weights.sum()
    
    # Sort projected data and corresponding weights
    u_sorted, u_indices = torch.sort(u_proj, dim=0)
    v_sorted, v_indices = torch.sort(v_proj, dim=0)
    
    u_weights_sorted = u_weights[u_indices]
    v_weights_sorted = v_weights[v_indices]
    
    # Compute cumulative weights
    u_cum_weights = torch.cumsum(u_weights_sorted, dim=0)
    v_cum_weights = torch.cumsum(v_weights_sorted, dim=0)
    
    # Merge and sort all points
    all_values = torch.cat([u_sorted, v_sorted], dim=0)
    all_cum_weights = torch.cat([u_cum_weights, v_cum_weights], dim=0)
    is_u = torch.cat([torch.ones_like(u_sorted, device=device),
                      torch.zeros_like(v_sorted, device=device)], dim=0)
    
    sorted_idx = torch.argsort(all_values, dim=0)
    all_values = torch.gather(all_values, 0, sorted_idx)
    all_cum_weights = torch.gather(all_cum_weights, 0, sorted_idx)
    is_u = torch.gather(is_u, 0, sorted_idx)
    
    # CDFs
    u_cdf = torch.where(is_u == 1, all_cum_weights, torch.zeros_like(all_cum_weights))
    v_cdf = torch.where(is_u == 0, all_cum_weights, torch.zeros_like(all_cum_weights))
    u_cdf, _ = torch.cummax(u_cdf, dim=0)
    v_cdf, _ = torch.cummax(v_cdf, dim=0)
    
    # Sum up the areas to get the Wasserstein distances
    cdf_diff = torch.abs(u_cdf - v_cdf)
    deltas   = all_values[1:] - all_values[:-1]
    
    if   p == 1:
        return torch.sum(cdf_diff[:-1] * deltas, dim=0)
    elif p == 2:
        return torch.sum(cdf_diff[:-1].pow(2) * deltas, dim=0)
    else:
        return torch.sum(cdf_diff[:-1].pow(p) * deltas, dim=0)

def sliced_wasserstein_distance(u_values: torch.Tensor, v_values: torch.Tensor,
                                u_weights: torch.Tensor=None, v_weights: torch.Tensor=None,
                                p: int=1, num_slices: int=1000, mode='EBSW', vectorized=True):
    """
    Sliced Wasserstein Distance over arbitrary dimensional samples
    
    References:
        https://arxiv.org/abs/1902.00434
        https://arxiv.org/abs/2304.13586 (EBSW)
    
    Notes:
        When using this as a loss function e.g. with neural nets, large
        minibatch sizes may be beneficial or needed.
    
    Args:
        u_values:    sample U vectors (n x dim)
        v_values:    sample V vectors (m x dim)
        u_weights:   sample U weights (n,) (if None, assumed to be unit)
        v_weights:   sample V weights (m,)
        p:           p-norm parameter (p = 1 is 'Earth Movers', 2 = is W-2, ...)       
        num_slices:  number of random MC projections (slices) (higher the better)
        mode:        'SWD'  (basic uniform MC random)
                     'EBSW' (faster convergence and smaller variance)
        vectorized:  fully vectorized (may take more GPU/CPU memory, but 10x faster)
    
    Returns:
        distance between the empirical sample distributions
    """
    
    # Generate a random projection direction
    dim  = int(u_values.shape[-1])
    directions = rand_projections(dim=dim, N=num_slices, device=u_values.device)
    
    if vectorized:
        
        dist = sliced_W_vectorized(u_values  = u_values,  v_values  = v_values,
                                u_weights = u_weights, v_weights = v_weights,
                                p = p, directions = directions)    
    else:
        
        dist = torch.zeros(num_slices, device=u_values.device, dtype=u_values.dtype)
        
        for i in range(num_slices):
            
            # Project the distributions on the random direction
            u_proj = torch.matmul(u_values, directions[i,:])
            v_proj = torch.matmul(v_values, directions[i,:])
            
            # Calculate the 1-dim Wasserstein on the direction
            dist[i] = wasserstein_distance_1D(u_values=u_proj, v_values=v_proj,
                                        u_weights=u_weights, v_weights=v_weights, p=p,
                                        apply_oop=False)
    
    if   mode == 'SWD':  # Standard uniform random
        dist    = dist.mean()

    elif mode == 'EBSW': # "Energy Based" importance sampling via softmax
        dist    = dist.view(1, num_slices)
        weights = torch.softmax(dist, dim=1)
        dist    = torch.sum(weights*dist, dim=1).mean()
    
    else:
        raise Exception(__name__ + f'.sliced_wasserstein_distance: Unknown "mode" chosen')
    
    if   p == 1:
        return dist
    elif p == 2:
        return torch.sqrt(dist)
    else:
        return torch.pow(dist, 1.0 / p)

def test_1D(EPS=1e-3):
    """
    Test function (fixed reference checked against scikit-learn)
    """
    
    # -----------------------------------------------
    # p = 1 case
    p = 1
    
    res = wasserstein_distance_1D(torch.tensor([0, 1, 3]), torch.tensor([5, 6, 8]), p=p).item()
    assert res == pytest.approx(5, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([0, 1]), torch.tensor([0, 1]),
                                  torch.tensor([3, 1]), torch.tensor([2, 2]), p=p).item()
    assert res == pytest.approx(0.25, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([3.4, 3.9, 7.5, 7.8]), torch.tensor([4.5, 1.4]),
                                  torch.tensor([1.4, 0.9, 3.1, 7.2]), torch.tensor([3.2, 3.5]), p=p).item()
    assert res == pytest.approx(4.0781, abs=EPS)

    # -----------------------------------------------
    ## p = 2 case
    p = 2
    
    res = wasserstein_distance_1D(torch.tensor([0, 1, 3]), torch.tensor([5, 6, 8]), p=p).item()
    assert res == pytest.approx(1.91485, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([0, 1]), torch.tensor([0, 1]),
                                  torch.tensor([3, 1]), torch.tensor([2, 2]), p=p).item()
    assert res == pytest.approx(0.25, abs=EPS)
    
    res = wasserstein_distance_1D(torch.tensor([3.4, 3.9, 7.5, 7.8]), torch.tensor([4.5, 1.4]),
                                  torch.tensor([1.4, 0.9, 3.1, 7.2]), torch.tensor([3.2, 3.5]), p=p).item()
    assert res == pytest.approx(1.67402, abs=EPS)

def test_swd():
    """
    Test function
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
    
    u_values  = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    u_weights = torch.tensor([0.2, 0.5, 0.3]) # event weights
    v_values  = torch.tensor([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    v_weights = torch.tensor([0.3, 0.4, 0.3]) # event weights
    p = 2
    num_slices = int(1e3)
    
    # Fixed input tests for 'SWD' and 'EBSW'
    seed = 42
    
    set_seed(seed)
    res = sliced_wasserstein_distance(u_values, v_values, u_weights, v_weights, p, num_slices, 'SWD').item()
    print(res)
    assert res == pytest.approx(0.38, abs=0.05)

    set_seed(seed)
    res_alt = sliced_wasserstein_distance(u_values, v_values, u_weights, v_weights, p, num_slices, 'EBSW').item()
    print(res_alt)
    assert res_alt == pytest.approx(0.38, abs=0.05)

    assert res == pytest.approx(res_alt, abs=0.01)
    
    # -------------------------------------------------------
    # Test vectorized versus non-vectorized implementation

    seed = 4321
    set_seed(seed)
    
    u_values = torch.randn(100, 4)
    v_values = torch.randn(150, 4)
    
    u_weights = torch.rand(100)
    v_weights = torch.rand(150)
    
    # Set the seed
    seed = 1234
    
    for p in [1,2]:
        for mode in ['SWD', 'EBSW']:
            
            set_seed(seed)
            tic = time()
            d     = sliced_wasserstein_distance(u_values, v_values,
                        u_weights, v_weights, p, num_slices, mode, vectorized=True).item()
            toc = time() - tic
            
            set_seed(seed)
            tic_alt = time()
            d_alt = sliced_wasserstein_distance(u_values, v_values,
                        u_weights, v_weights, p, num_slices, mode, vectorized=False).item()
            toc_alt = time() - tic_alt
            
            print(f'p = {p} ({mode}) || D = {d} (vectorized, {toc:0.2e} sec) | D = {d_alt} (non-vectorized, {toc_alt:0.2e} sec)')
            
            assert d == pytest.approx(d_alt, abs=1e-4)
