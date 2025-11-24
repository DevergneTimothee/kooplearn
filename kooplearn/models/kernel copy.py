from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import DotProduct, Kernel

from kooplearn._src.operator_regression import dual
from kooplearn._src.operator_regression.utils import (
    contexts_to_markov_train_states,
    parse_observables,
)
from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn._src.utils import ShapeError, check_contexts_shape, check_is_fitted
from kooplearn.abc import BaseModel


class KernelDMD(BaseModel, RegressorMixin):
    """
    Kernel Dynamic Mode Decomposition (KernelDMD) Model.
    Implements the KernelDMD estimators approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator following the approach described in :footcite:t:`Kostic2022`.

    Args:
        kernel (sklearn.gaussian_process.kernels.Kernel): sklearn Kernel object. Defaults to `DotProduct`.
        reduced_rank (bool): If ``True`` initializes the reduced rank estimator introduced in :footcite:t:`Kostic2022`, if ``False`` initializes the classical principal component estimator.
        rank (int): Rank of the estimator. Defaults to 5.
        tikhonov_reg (float): Tikhonov regularization coefficient. ``None`` is equivalent to ``tikhonov_reg = 0``, and internally calls specialized stable algorithms to deal with this specific case.
        svd_solver (str): Solver used to perform the internal SVD calcuations. Currently supported: `full`, uses LAPACK solvers, `arnoldi`, uses ARPACK solvers, `randomized`, uses randomized SVD algorithms as described in :guilabel:`TODO - ADD REF`.
        iterated_power (int): Number of power iterations when using a randomized algorithm (``svd_solver == 'randomized'``).
        n_oversamples (int): Number of oversamples when using a randomized algorithm (``svd_solver == 'randomized'``).
        optimal_sketching (bool): Sketching strategy for the randomized solver. If `True` performs optimal sketching (computaitonally expensive but more accurate).
        rng_seed (int): Random Number Generator seed. Only used when ``svd_solver == 'randomized'``.  Defaults to ``None``, that is no explicit seed is setted.

    .. tip::

        A powerful DMD variation proposed by :footcite:t:`Arbabi2017`, known as Hankel-DMD, evaluates the Koopman/Transfer estimators by stacking consecutive snapshots together in a Hankel matrix. When this model is fitted context windows of length > 2, the lookback window length is automatically set to ``context_len - 1``. Upon fitting, the whole lookback window is flattened and *concatenated* together before evaluating the kernel function, thus realizing an Hankel-KDMD estimator.

    Attributes:
        data_fit : Training data: array of context windows of shape ``(n_samples, context_len, *features_shape)``.
        kernel_X : Kernel matrix evaluated at the initial states, that is ``self.data_fit[:, :self.lookback_len, ...]`. Shape ``(n_samples, n_samples)``
        kernel_Y : Kernel matrix evaluated at the evolved states, that is ``self.data_fit[:, 1:self.lookback_len + 1, ...]``. Shape ``(n_samples, n_samples)``
        kernel_XY : Cross-kernel matrix between initial and evolved states. Shape ``(n_samples, n_samples)``.
        U : Projection matrix of shape (n_samples, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)` (see :footcite:t:`Kostic2022`).
        V : Projection matrix of shape (n_samples, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)` (see :footcite:t:`Kostic2022`).

    """

    def __init__(
        self,
        kernel: Kernel = DotProduct(),
        reduced_rank: bool = True,
        rank: int = 5,
        tikhonov_reg: Optional[float] = None,
        svd_solver: str = "full",
        iterated_power: int = 1,
        n_oversamples: int = 5,
        optimal_sketching: bool = False,
        rng_seed: Optional[int] = None,
        fit_generator: bool = False,
        mean_features: bool=False
    ):
        # Initial checks
        if svd_solver not in ["full", "arnoldi", "randomized"]:
            raise ValueError(
                "Invalid svd_solver. Allowed values are 'full', 'arnoldi' and 'randomized'."
            )
        if svd_solver == "randomized" and iterated_power < 0:
            raise ValueError("Invalid iterated_power. Must be non-negative.")
        if svd_solver == "randomized" and n_oversamples < 0:
            raise ValueError("Invalid n_oversamples. Must be non-negative.")

        self.rng_seed = rng_seed
        self._kernel = kernel
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer.")

        self.rank = rank
        if tikhonov_reg is None:
            self.tikhonov_reg = 0.0
        else:
            self.tikhonov_reg = tikhonov_reg
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.optimal_sketching = optimal_sketching

        self.reduced_rank = reduced_rank
        self._is_fitted = False
        self._lookback_len = -1
        self.fit_generator = fit_generator
        self.mean_features=mean_features

    @property
    def lookback_len(self) -> bool:
        return self._lookback_len

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = X.reshape(X.shape[0], -1)
        if Y is not None:
            Y = Y.reshape(Y.shape[0], -1)
        return self._kernel(X, Y)
    


    def fit(self, data: np.ndarray, verbose: bool = True,  forces=None, friction=None) -> KernelDMD:
        """
        Fits the KernelDMD model using either a randomized or a non-randomized algorithm, and either a full rank or a reduced rank algorithm,
        depending on the parameters of the model.


        .. attention::

            If ``context_len = data.shape[1] > 2``, the attribute :attr:`lookback_len` will be automatically set to ``context_len - 1``. The data will be first flattened along the trailing dimensions before computing the kernel function. The pseudo-code of this operation is

            .. code-block:: python

                X = data[:, :self.lookback_len, ...] #Y = data[:, 1:self.lookback_len + 1, ...]
                #Flatten the trailing dimensions
                X = X.reshape(data.shape[0], -1)
                self.kernel_X = self.kernel(X)

        Args:
            data (np.ndarray): Batch of context windows of shape ``(n_samples, context_len, *features_shape)``.

        Returns:
            The fitted estimator.
        """
        self._pre_fit_checks(data)
        if self.fit_generator:
            if forces is not None and friction is not None:
                self.build_matrices(data, forces, friction)
        if self.reduced_rank:
            if self.svd_solver == "randomized":
                if self.tikhonov_reg == 0.0:
                    raise ValueError(
                        "tikhonov_reg must be specified when solver is randomized."
                    )
                else:
                    U, V = dual.fit_rand_reduced_rank_regression(
                        self.kernel_X,
                        self.kernel_Y,
                        self.tikhonov_reg,
                        self.rank,
                        self.n_oversamples,
                        self.optimal_sketching,
                        self.iterated_power,
                        rng_seed=self.rng_seed,
                    )
                    if self.fit_generator:
                        U_gen, V_gen = dual.fit_rand_reduced_rank_regression(
                            self.kernel_X,
                            self.dXdY,
                            self.tikhonov_reg,
                            self.rank,
                            self.n_oversamples,
                            self.optimal_sketching,
                            self.iterated_power,
                            rng_seed=self.rng_seed,
                    )
            else:
                print("Fitting transfer operator")
                U, V = dual.fit_reduced_rank_regression(
                    self.kernel_X,
                    self.kernel_Y,
                    self.tikhonov_reg,
                    self.rank,
                    self.svd_solver,
                )
                print("Fitting generator")
                if self.fit_generator:
                    U_gen, V_gen = dual.fit_reduced_rank_regression(
                        self.kernel_X,
                        self.dXdY,
                        self.tikhonov_reg,
                        self.rank,
                        self.svd_solver,
                )
        else:
            if self.svd_solver == "randomized":
                U, V = dual.fit_rand_principal_component_regression(
                    self.kernel_X,
                    self.tikhonov_reg,
                    self.rank,
                    self.n_oversamples,
                    self.iterated_power,
                    rng_seed=self.rng_seed,
                )
            else:
                U, V = dual.fit_principal_component_regression(
                    self.kernel_X, self.tikhonov_reg, self.rank, self.svd_solver
                )
                if self.fit_generator:
                    U_gen, V_gen = U, V
        self.U = U
        self.V = V
        if self.fit_generator:
            self.U_gen = U_gen
            self.V_gen = V_gen
            if self.mean_features:
                self.U_gen = (np.eye(self.U_gen.shape[0]) - np.ones(self.U_gen.shape[0])/U_gen.shape[0])@self.U_gen
                self.V_gen = (np.eye(self.V_gen.shape[0]) - np.ones(self.V_gen.shape[0])/V_gen.shape[0])@self.U_gen
                self.trace = np.trace(np.linalg.inv((self.kernel_X + self.tikhonov_reg))@ self.kernel_YX)

        check_is_fitted(
            self,
            ["U", "V", "kernel_X", "kernel_Y", "kernel_YX", "data_fit", "lookback_len"],
        )
        self._is_fitted = True
        if verbose:
            print(
                f"Fitted {self.__class__.__name__} model. Lookback length set to {self.lookback_len}"
            )
        return self

    def risk(self, data: Optional[np.ndarray] = None) -> float:
        """Risk of the estimator on the validation ``data``.

        Args:
            data (np.ndarray or None): Batch of context windows of shape ``(n_samples, context_len, *features_shape)``. If ``None``, evaluates the risk on the training data.

        Returns:
            Risk of the estimator, see Equation 11 of :footcite:p:`Kostic2022` for more details.
        """
        if data is not None:
            check_contexts_shape(data, self.lookback_len)
            data = np.asanyarray(data)
            if data.shape[1] - 1 != self.lookback_len:
                raise ShapeError(
                    f"The  context length ({data.shape[1]}) of the validation data does not match the context length of the training data ({self.lookback_len + 1})."
                )

            X_val, Y_val = contexts_to_markov_train_states(data, self.lookback_len)
            X_train, Y_train = contexts_to_markov_train_states(
                self.data_fit, self.lookback_len
            )
            kernel_Yv = self.kernel(Y_val)
            kernel_XXv = self.kernel(X_train, X_val)
            kernel_YYv = self.kernel(Y_train, Y_val)
        else:
            kernel_Yv = self.kernel_Y
            kernel_XXv = self.kernel_X
            kernel_YYv = self.kernel_Y
        return dual.estimator_risk(
            kernel_Yv, self.kernel_Y, kernel_XXv, kernel_YYv, self.U, self.V
        )

    def predict(
        self,
        data: np.ndarray,
        t: int = 1,
        observables: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Predicts the state or, if the system is stochastic, its expected value :math:`\mathbb{E}[X_t | X_0 = X]` after ``t`` instants given the initial conditions ``X = data[:, self.lookback_len:, ...]`` being the lookback slice of ``data``.

        .. attention::

            ``data.shape[1]`` must match the lookback length ``self.lookback_len``. Otherwise, an error is raised.

        If ``observables`` are not ``None``, returns the analogue quantity for the observable instead.

        Args:
            data (numpy.ndarray): Initial conditions to predict. Array of context windows with shape ``(n_init_conditions, self.lookback_len, *self.data_fit.shape[2:])`` (see the note above).
            t (int): Number of steps in the future to predict (returns the last one).
            observables (callable or None): Callable or ``None``. If callable should map batches of states of shape ``(batch, *self.data_fit.shape[2:])`` to batches of observables ``(batch, *obs_features_shape)``.

        Returns:
           The predicted (expected) state/observable at time :math:`t`, shape ``(n_init_conditions, *obs_features_shape)``.
        """
        check_is_fitted(
            self,
            ["U", "V", "kernel_X", "kernel_Y", "kernel_YX", "data_fit", "lookback_len"],
        )

        _obs, expected_shape, X_inference, X_fit = parse_observables(
            observables, data, self.data_fit, self.lookback_len
        )

        K_Xin_X = self.kernel(X_inference, X_fit)
        return dual.predict(t, self.U, self.V, self.kernel_YX, K_Xin_X, _obs).reshape(
            expected_shape
        )

    def eig(
        self,
        eval_left_on: Optional[np.ndarray] = None,
        eval_right_on: Optional[np.ndarray] = None,
    ):
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Args:
            eval_left_on (numpy.ndarray or None): Array of context windows on which the left eigenfunctions are evaluated, shape ``(n_samples, *self.data_fit.shape[1:])``.
            eval_right_on (numpy.ndarray or None): Array of context windows on which the right eigenfunctions are evaluated, shape ``(n_samples, *self.data_fit.shape[1:])``.

        Returns:
            (eigenvalues, left eigenfunctions, right eigenfunctions) - Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. If ``eval_left_on`` or ``eval_right_on``  are not ``None``, returns the left/right eigenfunctions evaluated at ``eval_left_on``/``eval_right_on``: shape ``(n_samples, rank)``.
        """

        check_is_fitted(
            self,
            ["U", "V", "kernel_X", "kernel_Y", "kernel_YX", "lookback_len", "data_fit"],
        )

        if hasattr(self, "_eig_cache"):
            if self.fit_generator:
                w, vl, vr, w_gen, vl_gen,vr_gen = self._eig_cache
            else:
                w, vl, vr = self._eig_cache
        else:
            w, vl, vr = dual.estimator_eig(
                self.U, self.V, self.kernel_X, self.kernel_YX
            )
            if self.fit_generator:
                w_gen, vl_gen, vr_gen = dual.estimator_eig(
                    self.U_gen, self.V_gen, self.kernel_X, self.dXY
            )
            self._eig_cache = (w, vl, vr,w_gen,vl_gen,vr_gen)

        X_fit, Y_fit = contexts_to_markov_train_states(self.data_fit, self.lookback_len)
        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            if self.fit_generator:
                return w, w_gen
            else:
                return w
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            check_contexts_shape(
                eval_right_on, self.lookback_len, is_inference_data=True
            )
            kernel_Xin_X_or_Y = self.kernel(eval_right_on, X_fit)
            if self.fit_generator:
                return w, w_gen, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vr), dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vr_gen)
            else:
                return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vr)
        elif eval_left_on is not None and eval_right_on is None:
            # (eigenvalues, left eigenfunctions)
            check_contexts_shape(
                eval_left_on, self.lookback_len, is_inference_data=True
            )
            kernel_Xin_X_or_Y = self.kernel(eval_left_on, Y_fit)
            return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vl)
        elif eval_left_on is not None and eval_right_on is not None:
            # (eigenvalues, left eigenfunctions, right eigenfunctions)

            check_contexts_shape(
                eval_right_on, self.lookback_len, is_inference_data=True
            )
            check_contexts_shape(
                eval_left_on, self.lookback_len, is_inference_data=True
            )

            kernel_Xin_X_or_Y_left = self.kernel(eval_left_on, Y_fit)
            kernel_Xin_X_or_Y_right = self.kernel(eval_right_on, X_fit)
            return (
                w,
                dual.evaluate_eigenfunction(kernel_Xin_X_or_Y_left, vl),
                dual.evaluate_eigenfunction(kernel_Xin_X_or_Y_right, vr),
            )

    def modes(
        self,
        data: np.ndarray,
        observables: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Computes the mode decomposition of arbitrary observables of the Koopman/Transfer operator at the states defined by ``data``.

        Informally, if :math:`(\\lambda_i, \\xi_i, \\psi_i)_{i = 1}^{r}` are eigentriplets of the Koopman/Transfer operator, for any observable :math:`f` the i-th mode of :math:`f` at :math:`x` is defined as: :math:`\\lambda_i \\langle \\xi_i, f \\rangle \\psi_i(x)`. See :footcite:t:`Kostic2022` for more details.

        Args:
            data (numpy.ndarray): Initial conditions to compute the modes on. See :func:`predict` for additional details.
            observables (callable or None): Callable or ``None``. If callable should map batches of states of shape ``(batch, *self.data_fit.shape[2:])`` to batches of observables ``(batch, *obs_features_shape)``.

        Returns:
            Modes of the system at the states defined by ``data``. Array of shape ``(self.rank, n_states, ...)``.
        """
        check_is_fitted(
            self, ["U", "V", "kernel_X", "kernel_YX", "lookback_len", "data_fit"]
        )

        _obs, expected_shape, X_inference, X_fit = parse_observables(
            observables, data, self.data_fit, self.lookback_len
        )

        if hasattr(self, "_eig_cache"):
            _, lv, rv = self._eig_cache
        else:
            _, lv, rv = dual.estimator_eig(
                self.U, self.V, self.kernel_X, self.kernel_YX
            )

        K_Xin_X = self.kernel(X_inference, X_fit)
        _gamma = dual.estimator_modes(K_Xin_X, rv, lv)

        expected_shape = (self.rank,) + expected_shape
        return np.matmul(_gamma, _obs).reshape(
            expected_shape
        )  # [rank, num_initial_conditions, num_observables]

    def svals(self):
        """Singular values of the Koopman/Transfer operator.

        Returns:
            The estimated singular values of the Koopman/Transfer operator. Array of shape `(rank,)`.
        """
        check_is_fitted(self, ["U", "V", "kernel_X", "kernel_Y"])
        return dual.svdvals(self.U, self.V, self.kernel_X, self.kernel_Y)

    def _init_kernels(self, X: np.ndarray, Y: np.ndarray):
        K_X = self.kernel(X)
        K_Y = self.kernel(Y)
        K_YX =  self.kernel(Y, X)
        return K_X, K_Y, K_YX

    
    def return_dk(self, data, forces: np.ndarray, friction: np.float):
        """ Computes the dot product between the second order derivatives"""
        print("hello")
        sigma = self._kernel.length_scale
        print(sigma)
        X, Y_fit = contexts_to_markov_train_states(data, self.lookback_len)
        X = X.reshape(X.shape[0],-1)
        kern=self._kernel(X,X)
        forces, Y_fit = contexts_to_markov_train_states(forces, self.lookback_len)
        forces = forces.reshape(forces.shape[0],-1)

        difference = (X[:, np.newaxis, :] - X[np.newaxis, :, :])
        n =  difference.shape[2]
        off_diag = np.ones((n,n)) - np.eye(n)
        print(off_diag)
        #Computation of the first term of the dot product
        dk_1 = (-np.einsum('ijk,ijl,ik,jl->ij', difference, difference, forces, forces) / sigma**4 
                + np.einsum("ik,jk->ij",forces,forces)/sigma**2) #first term in the dot product the /sigma**2 comes from the case i=j
        
        #Computation of the second "square" term in the dot product, had to use a trick to treat the special case when i=j (see last appendix Houe 2023)
        
        first_term = (np.einsum('ijk,ijl,kl->ij', difference**2,(difference/ sigma)**2 - 1,off_diag)
                      +np.einsum('ijk,ijk->ij', (difference/ sigma)**2 - 6, difference**2)) / sigma**6 # add the diagonal term
        

        second_term = n*(n+2)/sigma**4
        third_term = np.einsum('ijk,lk->ij',difference**2/sigma**6,off_diag) #same here
        
        #Computation of the cross term:
        #ct_1 =  -0.5 * (np.einsum("ijk,ijl,il, kl->ij", (difference/sigma)**2-1,difference , forces, off_diag) 
        #              + np.einsum("ijk,ijk,ik->ij",(difference/sigma)**2-3, difference, forces)) * friction / sigma**4
        

        ct_2 =  0.5 * (np.einsum("ijk,ijl,jl, kl->ij",(difference/sigma)**2-1, difference, forces, off_diag) 
                      + np.einsum("ijk,ijk,jk->ij",(difference/sigma)**2-3, difference, forces)) * friction / sigma**4 
        #print(ct_1-ct_2)
        #dk = np.zeros((X.shape[0],X.shape[0]))
        #for i in range(X.shape[0]):
         #   for j in range(X.shape[0]):
         #       dk[i,j] = (forces[i,0]*forces[j,0])*(-(X[i,0]-X[j,0])**2 / sigma**4 +1/sigma**2) + \
         #       friction**2  * ((X[i,0]-X[j,0])**2 /sigma**2 - 6) * (X[i,0]-X[j,0]) **2 / (4*sigma**6) + friction**2 * 3/(4*sigma**4) + \
         #       + (X[i,0]-X[j,0])*((X[i,0]-X[j,0])**2 /sigma**2 -3) * forces[j,0]*friction/sigma**4 \
                #- 0.5 *  (X[i,0]-X[j,0])*((X[i,0]-X[j,0])**2 /sigma**2 -3) * forces[i,0]*friction/sigma**4
        #print(dk - dk_1 - (first_term + second_term - third_term)*friction**2/4 - 2* ct_2)
        #return (dk_1 + (first_term + second_term - third_term)*friction**2/4 + 2* ct_2)*self.kernel_X
        return (dk_1 + (first_term + second_term - third_term)*friction**2/4 + 2* ct_2)*self.kernel_X
    
    def return_mixed_term(self,data, forces:np.ndarray, friction:np.float):

        sigma = self._kernel.length_scale
        
        X, Y_fit = contexts_to_markov_train_states(data, self.lookback_len)
        X = X.reshape(X.shape[0],-1)
        kern=self._kernel(X,X)
        forces, Y_fit = contexts_to_markov_train_states(forces, self.lookback_len)
        forces = forces.reshape(forces.shape[0],-1)
        difference = (X[:, np.newaxis, :] - X[np.newaxis, :, :])
        n =  difference.shape[2]

        dk=  (np.einsum('ik,ijk->ij', -forces, difference)/sigma**2 + 0.5*friction * np.einsum('ijk->ij', difference**2)/sigma**4 - friction * n *0.5 / sigma**2) * self.kernel_X        #return self.gaussianG00G10(X,forces,friction,sigma)*self.kernel_X
        if self.mean_features:
            return dk - np.einsum("ij->i",dk)/dk.shape[0]
        else:
            return dk
    def _pre_fit_checks(self, data: np.ndarray) -> None:
        """Performs pre-fit checks on the training data.

        Use :func:`check_contexts_shape` to check and sanitize the input data, initialize the kernel matrices and saves the training data.

        Args:
            data (np.ndarray): Batch of context windows of shape ``(n_samples, context_len, *features_shape)``.
        """
        lookback_len = data.shape[1] - 1
        check_contexts_shape(data, lookback_len)
        data = np.asanyarray(data)
        # Save the lookback length as a private attribute of the model
        self._lookback_len = lookback_len
        X_fit, Y_fit = contexts_to_markov_train_states(data, self.lookback_len)

        self.kernel_X, self.kernel_Y, self.kernel_YX = self._init_kernels(X_fit, Y_fit)
        self.data_fit = data

        if hasattr(self, "_eig_cache"):
            del self._eig_cache
    def build_matrices(self, data, forces, friction):
        self.dXdY = self.return_dk(data, forces, friction) 
        self.dXY  = self.return_mixed_term(data, forces,friction) 
    def save(self, filename):
        """Serialize the model to a file.

        Args:
            filename (path-like or file-like): Save the model to file.
        """
        pickle_save(self, filename)

    @classmethod
    def load(cls, filename):
        """Load a serialized model from a file.

        Args:
            filename (path-like or file-like): Load the model from file.

        Returns:
            KernelDMD: The loaded model.
        """
        return pickle_load(cls, filename)
