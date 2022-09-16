#!/usr/bin/python

import numpy
from multiprocessing import Pool

def fill_array(start_val):
    return range(start_val,start_val+10)

if __name__ == "__main__":
    array_2D = numpy.zeros((20,10))
    pool = Pool(processes = 32)    
    list_start_vals = range(40,60)

    # get the result of pool.map (list of values returned by fill_array)
    # in a pool_result list 
    pool_result = pool.map(fill_array, list_start_vals)

    # the pool is processing its inputs in parallel, close() and join() 
    #can be used to synchronize the main process 
    #with the task processes to ensure proper cleanup.
    pool.close()
    pool.join()

    # Now assign the pool_result to your numpy
    for line,result in enumerate(pool_result):
        print(line)
        array_2D[line,:] = result

    print(array_2D)
