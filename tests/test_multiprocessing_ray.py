import multiprocessing
import time
import ray
import numpy as np
import os
import numba
import torch


@ray.remote
def np_sum_ray2(obj_ref, start, stop):
    obj_ref['output'][start] = torch.sum(obj_ref['input'][start:stop])
    return obj_ref['output'][start]


def benchmark(NUM_WORKERS):

    N = int(1e7)
    input_data  = torch.tensor(np.random.random((N,1)), dtype=float)
    output_data = N*[None] 
    data        = {'input': input_data, 'output': output_data}

    chunk_size  = int(N / NUM_WORKERS)
    futures     = []
    obj_ref     = ray.put(data)

    start_time  = time.time_ns()

    for i in range(0, NUM_WORKERS):
        
        futures.append(np_sum_ray2.remote(obj_ref, start, i + chunk_size))
    results = ray.get(futures)
    
    print(obj_ref['output'])

    return (time.time_ns() - start_time) / 1_000_000

def main():

    NUM_WORKERS = multiprocessing.cpu_count()
    ray.init(num_cpus=NUM_WORKERS)

    ## Test it
    trials = 10
    t = np.zeros(trials)
    for i in range(len(t)):
        t[i] = benchmark(NUM_WORKERS)

    print(f'time: mean = {np.mean(t):0.1f}, err(mean) = {np.std(t) / len(t):0.1f}')

main()


