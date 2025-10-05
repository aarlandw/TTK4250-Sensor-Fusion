import numpy as np

from mytypes import Measurement2d, MultiVarGauss
from tuning import EKFParams
from solution import initialize as initialize_solu  # noqa: F401 (not used)


def get_init_CV_state(
    meas0: Measurement2d, meas1: Measurement2d, ekf_params: EKFParams
) -> MultiVarGauss:
    """This function will estimate the initial state and covariance from
    the two first measurements"""
    dt = meas1.dt  # Interpreting this as the sampling time.
    z0, z1 = meas0.value, meas1.value
    print(z0.shape, z1.shape)
    z = np.hstack([z0, z1])
    sigma_a = ekf_params.sigma_a
    sigma_z = ekf_params.sigma_z

    O2 = np.zeros((2, 2))
    I2 = np.eye(2)
    Kp0 = O2
    Kp1 = I2
    K_u1 = (1 / dt) * I2
    K_u0 = -(1 / dt) * I2
    K = np.hstack([[Kp0, K_u0], [Kp1, K_u1]])
    print(f"K.shape: {K.shape}")
    print(f"z.shape: {z.shape}")
    p_hat1 = z1
    u_hat1 = K[1] @ z
    mean = np.hstack([p_hat1, u_hat1])

    # Covariance matrix
    R = (sigma_z**2) * I2
    Qa = (sigma_a**2) * (dt**2) / 3.0 * I2
    cov_top = np.hstack([R, R / dt])
    cov_bottom = np.hstack([R / dt, 2.0 * R / (dt**2) + Qa])
    cov = np.vstack([cov_top, cov_bottom])

    init_state = MultiVarGauss(mean, cov)
    return init_state
