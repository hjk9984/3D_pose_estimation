import torch
import numpy as np

def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_use):
    T = normalized_data.shape[0]  # Batch size
    D = data_mean.shape[0]  # 96

    orig_data = np.zeros((T, D), dtype=np.float32)

    orig_data[:, dimensions_to_use] = normalized_data

    # Multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data

def world2cam(coord, R=None, t=None):
    '''
        coord : (N, 3)
        R : rotation matrix     (3, 3)
        t : transition matrix   (1, 3)
    '''

    if R is None:
        R = 1
    if t is None:
        t = 0

    cam_coord = R @ coord.T + t.T
    return cam_coord.T

def cam2world(coord, R=None, t=None):
    '''
        coord : (N, 3)
        R : rotation matrix     (3, 3)
        t : transition matrix   (1, 3)
    '''

    if R is None:
        R = 1
    if t is None:
        t = 0

    wor_coord = R @ (coord + t).T
    return wor_coord.T

def project_2dto3d(coord, R, t, f, c):
    x = world2cam(coord, R, t)
    f = np.eye(2) * f
    c = c / x[:, 2]
    x = x[:, :2] / x[:, 2]  # (N, 2)
    x = x @ f + c
    return x

    

if __name__ == "__main__":
    tmp = np.array([[1, 1, 0]])
    R = np.array(
        [[np.cos(np.pi / 4), -np.sin(np.pi/4), 0],
        [np.sin(np.pi / 4), np.cos(np.pi/4), 0],
        [0, 0, 1]]
    )
    t = np.array([1, 1, 1]).reshape(1, 3)
    print(world2cam(tmp, R=R, t=t))

    f = np.array([2, 3])
    c = np.array([10, 100])
    print(project_2dto3d(tmp, R, t, f, c))

