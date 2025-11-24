from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import numpy as np
import torch
from kooplearn._src.linalg import cov
from kooplearn._src.operator_regression import primal
from kooplearn._src.operator_regression.utils import (
    contexts_to_markov_train_states,
    parse_observables,
)
from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import ShapeError, check_contexts_shape, check_is_fitted
from kooplearn.abc import BaseModel, FeatureMap
from kooplearn.models.feature_maps import IdentityFeatureMap
from kooplearn.models import ExtendedDMD
import functorch
logger = logging.getLogger("kooplearn")


class GenExtendedDMD(ExtendedDMD):
    """
    Extended Dynamic Mode Decomposition (ExtendedDMD) Model.
    Implements the ExtendedDMD estimators approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator following the approach described in :footcite:t:`Kostic2022`.

    Args:
        feature_map (callable): Dictionary of functions used for the ExtendedDMD algorithm. Should be a subclass of ``kooplearn.abc.FeatureMap``.
        reduced_rank (bool): If ``True`` initializes the reduced rank estimator introduced in :footcite:t:`Kostic2022`, if ``False`` initializes the classical principal component estimator.
        rank (int): Rank of the estimator. ``None`` returns the full rank estimator.
        tikhonov_reg (float): Tikhonov regularization coefficient. ``None`` is equivalent to ``tikhonov_reg = 0``, and internally calls specialized stable algorithms to deal with this specific case.
        svd_solver (str): Solver used to perform the internal SVD calcuations. Currently supported: `full`, uses LAPACK solvers, `arnoldi`, uses ARPACK solvers, `randomized`, uses randomized SVD algorithms as described in :guilabel:`TODO - ADD REF`.
        iterated_power (int): Number of power iterations when using a randomized algorithm (``svd_solver == 'randomized'``).
        n_oversamples (int): Number of oversamples when using a randomized algorithm (``svd_solver == 'randomized'``).
        rng_seed (int): Random Number Generator seed. Only used when ``svd_solver == 'randomized'``.  Defaults to ``None``, that is no explicit seed is setted.

    .. tip::

        A powerful DMD variation proposed by :footcite:t:`Arbabi2017`, known as Hankel-DMD, evaluates the Koopman/Transfer estimators by stacking consecutive snapshots together in a Hankel matrix. When this model is fitted context windows of length > 2, the lookback window length is automatically set to ``context_len - 1``. Upon fitting, the whole lookback window is passed through the feature map and the results are then flattened and *concatenated* together, realizing an Hankel-EDMD estimator.

    Attributes:
        data_fit : Training data: array of context windows of shape ``(n_samples, context_len, *features_shape)``.
        cov_X : Covariance matrix of the feature map evaluated at the initial states, that is ``self.data_fit[:, :self.lookback_len, ...]``.
        cov_Y : Covariance matrix of the feature map evaluated at the evolved states, , that is ``self.data_fit[:, 1:self.lookback_len + 1, ...]``.
        cov_XY : Cross-covariance matrix between initial and evolved states.
        U : Projection matrix of shape (n_out_features, rank). The Koopman/Transfer operator is approximated as :math:`U U^T \mathrm{cov_{XY}}`.
    """

    def __init__(
        self,
        feature_map: FeatureMap = IdentityFeatureMap(),
        reduced_rank: bool = True,
        rank: Optional[int] = None,
        tikhonov_reg: Optional[float] = None,
        svd_solver: str = "full",
        iterated_power: int = 1,
        n_oversamples: int = 5,
        rng_seed: Optional[int] = None,
    ):
        # Perform checks on the input arguments:
        super().__init__(feature_map,reduced_rank,rank,tikhonov_reg,svd_solver,iterated_power,n_oversamples,rng_seed)




    def _init_covs(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initializes the covariance matrices `cov_X`, `cov_Y`, and `cov_XY`.

        Args:
            stacked (np.ndarray): Training data of shape ``(n_samples, 2,  *features_shape)``. It should be the result of the function :func:`stack_lookback`.

        Returns:
            A tuple containing:
                - ``cov_X`` (np.ndarray): Covariance matrix of the feature map evaluated at X, shape ``(n_features, n_features)``.
                - ``cov_Y`` (np.ndarray): Covariance matrix of the feature map evaluated at Y, shape ``(n_features, n_features)``.
                - ``cov_XY`` (np.ndarray): Cross-covariance matrix of the feature map evaluated at X and Y, shape ``(n_features, n_features)``.
        """

        #X = torch.tensor(X,dtype=torch.float32)
        n_samples = X.shape[0]

        feature_X, encoded_Y  = self._feature_map(X.reshape(n_samples,-1),Y.reshape(n_samples,-1))
        cov_Y = cov(encoded_Y,encoded_Y)
        cov_XY = cov(feature_X,encoded_Y)
        cov_X = cov(feature_X,feature_X)
        return cov_X, cov_Y, cov_XY




class DMD(ExtendedDMD):
    """
    Dynamic Mode Decomposition (DMD) Model.
    Implements the classical DMD estimator approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator. This model just a minimal wrapper around ``ExtendedDMD`` setting the feature map to the identity function.
    """

    def __init__(
        self,
        reduced_rank: bool = True,
        rank: Optional[int] = 5,
        tikhonov_reg: float = 0,
        svd_solver: str = "full",
        iterated_power: int = 1,
        n_oversamples: int = 5,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(
            reduced_rank=reduced_rank,
            rank=rank,
            tikhonov_reg=tikhonov_reg,
            svd_solver=svd_solver,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            rng_seed=rng_seed,
        )
