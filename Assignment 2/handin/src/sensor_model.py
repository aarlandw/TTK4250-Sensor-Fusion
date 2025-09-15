from dataclasses import dataclass
import numpy as np

from gaussian import MultiVarGauss2d
from measurement import Measurement2d
from solution import sensor_model as sensor_model_solu

x_bar = np.array([0, 0])
P = 25 * np.eye(2)
Hc = Hr = np.eye(2)
Rc = np.array([[79, 36], [36, 36]])
Rr = np.array([[28, 4], [4, 22]])
zc = np.array([2, 14])
zr = np.array([-4, 6])


@dataclass
class LinearSensorModel2d:
    """A 2d sensor model"""

    H: np.eye(2)
    R = np.array([[79, 36], [36, 36]])
    # R: np.ndarray

    def get_pred_meas(self, state_est: MultiVarGauss2d) -> MultiVarGauss2d:
        pred_mean = self.H @ state_est.mean
        pred_cov = self.H @ state_est.cov @ self.H.T + self.R

        pred_meas = MultiVarGauss2d(pred_mean, pred_cov)

        # TODO replace this with own code
        # pred_meas = sensor_model_solu.LinearSensorModel2d.get_pred_meas(self, state_est)

        return pred_meas

    def meas_as_gauss(self, meas: Measurement2d) -> MultiVarGauss2d:
        """Get the measurement as a Gaussian distribution."""
        meas_gauss = MultiVarGauss2d(meas.value, self.R)
        return meas_gauss
