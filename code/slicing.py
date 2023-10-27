import numpy as np

def find_slice(arr,slices,category):
    best = 0
    slice_no = 0
    for x in range(slices):
        mat = np.matrix(arr[:, :, x])
        current = np.count_nonzero(mat == category)

        if current > best:
           
            best = current
            slice_no = x
    return slice_no