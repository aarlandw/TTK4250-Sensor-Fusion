from dataclasses import dataclass
from typing import Sequence
from senfuslib import MultiVarGauss, DynamicModel, SensorModel, TimeSequence
import numpy as np
import tqdm
import logging

from states import StateCV, MeasPos
from models import ModelImm
from sensors import SensorPos
from gaussian_mixture import GaussianMixture
from solution import filter as filter_solu


@dataclass
class OutputEKF:
    x_est_upd: MultiVarGauss[StateCV]
    x_est_pred: MultiVarGauss[StateCV]
    z_est_pred: MultiVarGauss[MeasPos]


@dataclass
class EKF:
    dynamic_model: DynamicModel[StateCV]
    sensor_model: SensorModel[MeasPos]

    def step(
        self, x_est_prev: MultiVarGauss[StateCV], z: MeasPos, dt: float
    ) -> OutputEKF:
        """Perform one EKF update step."""
        x_est_pred = self.dynamic_model.pred_from_est(x_est_prev, dt)
        z_est_pred = self.sensor_model.pred_from_est(x_est_pred)

        H_mat = self.sensor_model.H(x_est_pred.mean)
        P_mat = x_est_pred.cov
        S_mat = z_est_pred.cov

        kalman_gain = P_mat @ H_mat.T @ np.linalg.inv(S_mat)
        innovation = z - z_est_pred.mean

        state_upd_mean = x_est_pred.mean + kalman_gain @ innovation
        state_upd_cov = P_mat - kalman_gain @ H_mat @ P_mat

        x_est_upd = MultiVarGauss(state_upd_mean, state_upd_cov)

        return OutputEKF(x_est_upd, x_est_pred, z_est_pred)


@dataclass
class FilterIMM:
    dynamic_model: ModelImm
    sensor_model: SensorPos

    def calculate_mixings(
        self, x_est_prev: GaussianMixture[StateCV], dt: float
    ) -> np.ndarray:
        """Calculate the mixing probabilities, following step 1 in (6.4.1).

        The output should be on the following from:
        $mixing_probs[s_{k-1}, s_k] = \mu_{s_{k-1}|s_k}$
        """
        pi_mat = self.dynamic_model.get_pi_mat_d(dt)  # the pi in (6.6)
        prev_weights = x_est_prev.weights  # \mathbf{p}_{k-1}

        # Calculate mixing probabilities using (6.32)
        # mixing_probs[i, j] = mu_{i|j} = pi[i,j] * prev_weights[i] / normalization
        mixing_probs = np.zeros((len(prev_weights), len(prev_weights)))

        # First calculate the predicted weights for normalization (6.6)
        weights_pred = pi_mat.T @ prev_weights

        for j in range(len(prev_weights)):  # for each current mode s_k
            for i in range(len(prev_weights)):  # for each previous mode s_{k-1}
                mixing_probs[i, j] = pi_mat[i, j] * prev_weights[i] / weights_pred[j]

        # TODO remove this
        # mixing_probs = filter_solu.FilterIMM.calculate_mixings(self, x_est_prev, dt)

        return mixing_probs

    def mixing(
        self,
        x_est_prev: GaussianMixture[StateCV],
        mixing_probs: np.ndarray,
    ) -> Sequence[MultiVarGauss[StateCV]]:
        """Calculate the moment-based approximations,
        following step 2 in (6.4.1).
        Should return a gaussian with mean=(6.34) and cov=(6.35).

        Hint: Create a GaussianMixture for each mode density (6.33),
        and use .reduce() to calculate (6.34) and (6.35).
        """
        moment_based_preds = []
        for j in range(len(x_est_prev)):  # for each current mode
            # Create weights for this mode j using mixing probabilities
            # mixing_probs[i, j] = probability that previous mode was i given current mode is j
            weights_j = mixing_probs[:, j]  # column j contains mu_{i|j} for all i

            # Create GaussianMixture for mode j using (6.33)
            mixture = GaussianMixture(weights_j, x_est_prev.gaussians)

            # Reduce to get moment-based prediction using (6.34) and (6.35)
            moment_based_pred = mixture.reduce()
            moment_based_preds.append(moment_based_pred)

        # TODO remove this
        # moment_based_preds = filter_solu.FilterIMM.mixing(
        #     self, x_est_prev, mixing_probs
        # )
        return moment_based_preds

    def mode_match_filter(
        self, moment_based_preds: GaussianMixture[StateCV], z: MeasPos, dt: float
    ) -> Sequence[OutputEKF]:
        """Calculate the mode-match filter outputs (6.36),
        following step 3 in (6.4.1).

        Hint: Use the EKF class from the top of this file.
        The last part (6.37) is not part of this
        method and is done later."""
        ekf_outs = []

        for i, x_prev in enumerate(moment_based_preds):
            # Create EKF with the appropriate dynamic model for mode i
            ekf = EKF(self.dynamic_model.models[i], self.sensor_model)

            # Run one EKF step using the moment-based prediction as input
            out_ekf = ekf.step(x_prev, z, dt)
            ekf_outs.append(out_ekf)

        # TODO remove this
        # ekf_outs = filter_solu.FilterIMM.mode_match_filter(
        #     self, moment_based_preds, z, dt
        # )

        return ekf_outs

    def update_probabilities(
        self, ekf_outs: Sequence[OutputEKF], z: MeasPos, dt: float, weights: np.ndarray
    ) -> np.ndarray:
        """Update the mixing probabilities,
        using (6.37) from step 3 and (6.38) from step 4 in (6.4.1).

        Hint: Use (6.6)
        """
        pi_mat = self.dynamic_model.get_pi_mat_d(dt)

        weights_pred = None  # TODO, use (6.6)
        z_probs = None  # TODO

        weights_upd = None  # TODO

        # TODO remove this
        weights_upd = filter_solu.FilterIMM.update_probabilities(
            self, ekf_outs, z, dt, weights
        )
        return weights_upd

    def step(
        self, x_est_prev: GaussianMixture[StateCV], z: MeasPos, dt
    ) -> GaussianMixture[StateCV]:
        """Perform one step of the IMM filter."""
        mixing_probs = None  # TODO
        momend_based_preds = None  # TODO
        ekf_outs = None  # TODO
        weights_upd = None  # TODO
        x_est_upd = None  # TODO
        if ekf_outs is not None:  # You can remove this
            x_est_pred = GaussianMixture(
                x_est_prev.weights, [out.x_est_pred for out in ekf_outs]
            )

            z_est_pred = GaussianMixture(
                x_est_prev.weights, [out.z_est_pred for out in ekf_outs]
            )

        # TODO remove this
        x_est_upd, x_est_pred, z_est_pred = filter_solu.FilterIMM.step(
            self, x_est_prev, z, dt
        )

        return x_est_upd, x_est_pred, z_est_pred

    def run(
        self, x0_est: GaussianMixture[StateCV], zs: TimeSequence[MeasPos]
    ) -> TimeSequence[GaussianMixture[StateCV]]:
        """Run the IMM filter."""
        logging.info("Running IMM filter")
        x_est_upds = TimeSequence()
        x_est_preds = TimeSequence()
        z_est_preds = TimeSequence()
        x_est_upds.insert(0, x0_est)
        t_prev = 0
        for t, z in tqdm.tqdm(zs.items(), total=len(zs)):
            t_prev, x_est_prev = x_est_upds[-1]
            dt = np.round(t - t_prev, 8)

            x_est_upd, x_est_pred, z_est_pred = self.step(x_est_prev, z, dt)
            x_est_upds.insert(t, x_est_upd)
            x_est_preds.insert(t, x_est_pred)
            z_est_preds.insert(t, z_est_pred)
        return x_est_upds, x_est_preds, z_est_preds
