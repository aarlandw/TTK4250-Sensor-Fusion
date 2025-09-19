import numpy as np

from mytypes import Measurement2d, MultiVarGauss
from tuning import EKFParams
from solution import initialize as initialize_solu


def get_init_CV_state(meas0: Measurement2d, meas1: Measurement2d,
                      ekf_params: EKFParams) -> MultiVarGauss:
    """This function will estimate the initial state and covariance from
    the two first measurements"""
    dt = meas1.dt
    z0, z1 = meas0.value, meas1.value
    sigma_a = ekf_params.sigma_a
    sigma_z = ekf_params.sigma_z

    mean = None  # TODO
    cov = None  # TODO

    init_state = MultiVarGauss(mean, cov)

    # TODO replace this with own code
    init_state = initialize_solu.get_init_CV_state(meas0, meas1, ekf_params)
    return init_state
