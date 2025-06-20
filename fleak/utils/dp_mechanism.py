import numpy as np
import math

def gaussian_simple(epsilon, delta, sensitivity, size):
    noise = np.sqrt(2 * np.log(1.25/delta)) * sensitivity / epsilon
    return np.random.normal(0, noise, size=size)


## calculate sensitivity
def cal_sensitivity_up(lr, clip):
    return 2 * lr * clip

def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)

