import numpy as np
from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from solution import conditioning as conditioning_solu


def get_cond_state(state: MultiVarGauss2d,
                   sens_modl: LinearSensorModel2d,
                   meas: Measurement2d
                   ) -> MultiVarGauss2d:
    pred_meas = sens_modl.get_pred_meas(state)
    kalman_gain = pred_meas.cov @ sens_modl.H.T @ np.linalg.inv(sens_modl.H @ pred_meas.cov @ sens_modl.H.T + sens_modl.R)
    innovation = meas.value - pred_meas.mean
    cond_mean = state.mean + kalman_gain @ innovation
    cond_cov = state.cov - kalman_gain @ sens_modl.H @ state.cov

    cond_state = MultiVarGauss2d(cond_mean, cond_cov)

    # TODO replace this with own code
    # cond_state = conditioning_solu.get_cond_state(state, sens_modl, meas)

    return cond_state
