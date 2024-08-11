import numpy as np

# From Github https://github.com/BerensRWU/DenseMap#readme
# From Github https://github.com/balcilar/DenseDepthMap

def dense_map_new(lidarOnImage, n, m, grid):
    # remove out of image size indexes
    mask = (lidarOnImage[:, :2] >= 1).all(axis=1)
    Pts = lidarOnImage[mask].T

    ng = 2 * grid + 1

    mX = np.zeros((m,n)) + float("inf") # inf-matrix
    mY = np.zeros((m,n)) + float("inf") # inf-matrix
    mD = np.zeros((m,n))
    mX[np.int32(np.round(Pts[1])-1),np.int32(np.round(Pts[0])-1)] = Pts[0] - np.round(Pts[0])
    mY[np.int32(np.round(Pts[1])-1),np.int32(np.round(Pts[0])-1)] = Pts[1] - np.round(Pts[1])
    mD[np.int32(np.round(Pts[1])-1),np.int32(np.round(Pts[0])-1)] = Pts[2]

    KmX = np.zeros((ng, ng, m - 2*grid, n - 2*grid))
    KmY = np.zeros((ng, ng, m - 2*grid, n - 2*grid))
    KmD = np.zeros((ng, ng, m - 2*grid, n - 2*grid))

    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - 2*grid + i), j : (n - 2*grid + j)] - grid + j
            KmY[i,j] = mY[i : (m - 2*grid + i), j : (n - 2*grid + j)] - grid + i
            KmD[i,j] = mD[i : (m - 2*grid + i), j : (n - 2*grid + j)]

    S = np.zeros_like(KmD[0,0])
    Y = np.zeros_like(KmD[0,0])

    for i in range(ng):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s

    S[S == 0] = 1
    out = np.zeros((m,n))
    out[grid : -grid, grid  : -grid] = Y/S
    return out