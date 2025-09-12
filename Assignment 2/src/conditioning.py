import numpy as np
from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from sensor_model import LinearSensorModel2d
from solution import conditioning as conditioning_solu


def get_cond_state(state: MultiVarGauss2d,
                   sens_modl: LinearSensorModel2d,
                   meas: Measurement2d
                   ) -> MultiVarGauss2d:
    pred_meas = None  # TODO
    kalman_gain = None  # TODO
    innovation = None  # TODO
    cond_mean = None  # TODO
    cond_cov = None  # TODO

    cond_state = MultiVarGauss2d(cond_mean, cond_cov)

    # TODO replace this with own code
    cond_state = conditioning_solu.get_cond_state(state, sens_modl, meas)

    return cond_state
