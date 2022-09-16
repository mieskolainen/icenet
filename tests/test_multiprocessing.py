import multiprocessing
import numpy as np

# split a list into evenly sized chunks
def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def do_job(job_id, data_slice):
    print(f"job[{job_id}]: {data_slice}")
    return data_slice

def dispatch_jobs(data, job_number):
    total      = len(data)
    chunk_size = int(total / job_number)
    slice      = chunks(data, chunk_size)
    jobs       = []

    for job_i, s in enumerate(slice):
        j = multiprocessing.Process(target=do_job, args=(job_i, s))
        jobs.append(j)
    
    for j in jobs:
        a = j.start()
        print(f'job: {j}, return: {a}')

if __name__ == "__main__":

    data = np.random.rand(2**16)
    print(data)

    dispatch_jobs(data, 12)
