def split(a, n):
    """
    Generator which returns approx equally sized chunks.
    Args:
        a : Total number
        n : Number of chunks

    Example:
        list(split(range(10), 3))
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def split_start_end(a, n, end_plus=1):
    """
    Returns approx equally sized chunks.

    Args:
        a :       Total number
        n :       Number of chunks
        end_plus: Python/nympy index style (i.e. + 1 for the end)
    
    Examples:
        split_start_end(range(100), 3)  returns [[0, 34], [34, 67], [67, 100]]
        split_start_end(range(5,25), 3) returns [[5, 12], [12, 19], [19, 25]]
    """
    ll  = list(split(a,n))
    out = []

    for i in range(len(ll)):
        out.append([ll[i][0], ll[i][-1] + end_plus])

    return out

def split_size(a, n):
    """
    As split_start_end() but returns only size per chunk
    """
    ll  = list(split(a,n))
    out = [len(ll[i]) for i in range(len(ll))]

    return out



print(list(split(range(35), 3)))

print(split_start_end(range(0,1),1))
