import numpy as np

def dense_depth_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    
    # linearindex = np.ravel_multi_index((np.round(Pts[:, 1]).astype(int), np.round(Pts[:, 0]).astype(int)), (n, m))
    # 再看一下！！！
    rounded_x = np.round(Pts[:, 1]).astype(int)
    rounded_y = np.round(Pts[:, 0]).astype(int)
    clipped_x = np.clip(rounded_x, 0, m - 1)
    clipped_y = np.clip(rounded_y, 0, n - 1)
    linearindex = np.ravel_multi_index((clipped_y, clipped_x), (n, m))


    mX = np.full((n, m), np.inf)
    mX.flat[linearindex] = Pts[:, 0] - np.round(Pts[:, 0])
    mY = np.full((n, m), np.inf)
    mY.flat[linearindex] = Pts[:, 1] - np.round(Pts[:, 1])
    mD = np.zeros((n, m))
    mD.flat[linearindex] = Pts[:, 2]
    
    KmX = {}
    KmY = {}
    KmD = {}
    
    for i in range(ng):
        for j in range(ng):
            KmX[(i, j)] = mX[i:n-ng+i, j:m-ng+j] - grid - 1 + i
            KmY[(i, j)] = mY[i:n-ng+i, j:m-ng+j] - grid - 1 + j
            KmD[(i, j)] = mD[i:n-ng+i, j:m-ng+j]
    
    S = np.zeros_like(KmD[(0, 0)])
    Y = np.zeros_like(KmD[(0, 0)])
    
    for i in range(ng):
        for j in range(ng):
            s = 1. / np.sqrt(KmX[(i, j)] ** 2 + KmY[(i, j)] ** 2)
            Y += s * KmD[(i, j)]
            S += s
    
    S[S == 0] = 1
    out = np.zeros((n, m))
    out[grid+1:-grid, grid+1:-grid] = Y / S
    
    return out, mD
