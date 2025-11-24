from __future__ import annotations

import logging
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
from kooplearn.models.kernel import KernelDMD
from jax import vmap, jacrev, hessian, grad
import jax.numpy as jnp
logger = logging.getLogger("kooplearn")

class KernelGenerator(KernelDMD):

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
    ):
        super().__init__(kernel,reduced_rank,rank,tikhonov_reg,svd_solver,iterated_power,n_oversamples,optimal_sketching,rng_seed)
    def fit(self, data: np.ndarray, verbose: bool = True,  forces=None, friction=None,bias=None, beta=1.0) -> KernelGenerator:
        #self.weights = weights
        self.bias = bias
        self.beta = beta
        self._pre_fit_checks(data)
        #if weights is not None:
        #    w_matrix = self._build_weights(weights)
        #    self.kernel_X*=w_matrix
        #print(w_matrix)
        self._build_matrices(data,forces,friction)

        super().fit(data, verbose)

        #if weights is not None:
        #    w_matrix = self._build_weights(weights)
        #    self.kernel_X*=w_matrix
        return self



    
    def _init_kernels(self, X: np.ndarray, Y: np.ndarray):
        print("hello")
        K_X = self.kernel(X)
        return K_X
    def kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = X.reshape(X.shape[0], -1)
        if Y is not None:
            Y = Y.reshape(Y.shape[0], -1)
        if self.bias is not None:
            if Y is not None:
                exp = 1.0 #np.exp(self.beta * (self.bias(X).reshape(-1,1)+self.bias(Y).reshape(1,-1))/2).reshape(X.shape[0],Y.shape[0])
            else:
                exp = np.exp(self.beta * (self.bias(X)[:,np.newaxis]+self.bias(X))/2).reshape(X.shape[0],X.shape[0])
        else:
            exp =1.0
        return self._kernel(X, Y)*exp
    def _build_weights(self,weights):
        weights_b, weights_a = contexts_to_markov_train_states(weights, self.lookback_len)
        weights_b=weights_b.reshape(-1)
        return np.einsum("i,j->ij",weights_b,weights_b)

    
    def _build_matrices(self, data, forces, friction):
        self.kernel_Y = self.return_dk(data, forces, friction) 
        self.kernel_YX  = self.return_mixed_term(data, forces,friction)
    
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

        self.kernel_X = self._init_kernels(X_fit,Y_fit)
        self.data_fit = data

        if hasattr(self, "_eig_cache"):
            del self._eig_cache

    def return_dk(self, data, forces: np.ndarray, friction: np.float):
        """ Computes the dot product between the second order derivatives"""
        sigma = self._kernel.length_scale
        X, Y_fit = contexts_to_markov_train_states(data, self.lookback_len)
        X = X.reshape(X.shape[0],-1)
        kern=self._kernel(X,X)
        forces, Y_fit = contexts_to_markov_train_states(forces, self.lookback_len)
        forces = forces.reshape(forces.shape[0],-1)

        difference = (X[:, np.newaxis, :] - X[np.newaxis, :, :])
        n =  difference.shape[2]
        off_diag = np.ones((n,n)) - np.eye(n)

        #Computation of the first term of the dot product
        dk_1 = (-np.einsum('ijk,ijl,ik,jl->ij', difference, difference, forces, forces) / sigma**4 
                + np.einsum("ik,jk->ij",forces,forces)/sigma**2) #first term in the dot product the /sigma**2 comes from the case i=j
        
        #Computation of the second "square" term in the dot product, had to use a trick to treat the special case when i=j (see last appendix Houe 2023)
        
        first_term = (np.einsum('ijk,ijl,kl->ij', difference**2,(difference/ sigma)**2 - 1,off_diag)
                      +np.einsum('ijk,ijk->ij', (difference/ sigma)**2 - 6, difference**2)) / sigma**6 # add the diagonal term
        

        second_term = n*(n+2)/sigma**4
        third_term = np.einsum('ijk,lk->ij',difference**2/sigma**6,off_diag) #same here
        

        ct_2 =  0.5 * (np.einsum("ijk,ijl,jl, kl->ij",(difference/sigma)**2-1, difference, forces, off_diag) 
                      + np.einsum("ijk,ijk,jk->ij",(difference/sigma)**2-3, difference, forces)) * friction / sigma**4 

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

        return dk


    def compute_prediction(self,index,train_data, t, bin_edge1, bin_edge2):
        evs, ul, ur = dual.estimator_eig(
                self.U, self.V, self.kernel_X, self.kernel_YX
        )
        n = self.kernel_YX.shape[0]

        dphi = self.kernel_YX[index]
        #uv_t = (self.U@ur).T

        h = self.kernel_YX[index]

        interval = np.where(np.logical_and(train_data > bin_edge1, train_data < bin_edge2), 1 ,0)
        s_identity = interval[:,0] / np.sqrt(n)

        g = (self.V@ul).T @ s_identity
    #print(np.exp(evs*t))
        pred = ((np.exp(evs*t) * (g*h)).sum(axis=-1))
        return pred



class KernelInverseGenerator(KernelDMD):

    def __init__(
        self,
        kernel: Kernel = DotProduct(),
        reduced_rank: bool = True,
        rank: int = 5,
        tikhonov_reg: Optional[float] = None,
        eta = 1e-3,
        svd_solver: str = "full",
        iterated_power: int = 1,
        n_oversamples: int = 5,
        optimal_sketching: bool = False,
        rng_seed: Optional[int] = None,
        transform=False
    ):
        self.eta = eta
        self.transform = transform
        super().__init__(kernel,reduced_rank,rank,tikhonov_reg,svd_solver,iterated_power,n_oversamples,optimal_sketching,rng_seed)
    def fit(self, data: np.ndarray, verbose: bool = True,  forces=None, friction=None,weights=None) -> KernelGenerator:

        self._pre_fit_checks(data)
        if weights is not None:
            w_matrix = self._build_weights(weights)
            self.kernel_X *= w_matrix
        self._build_matrices(data,forces,friction)
   

        U, V = dual.fit_principal_component_regression_generator(     
                 self.kernel_X, self.return_mixed_term(data, forces,friction), self.tikhonov_reg, self.eta, self.rank, self.svd_solver
                )

        self.U = U
        self.V = V

        

        # Final Checks
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
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = dual.estimator_eig(
                self.U, self.V, self.kernel_X, self.kernel_X
            )
            self._eig_cache = (w, vl, vr)

        X_fit, Y_fit = contexts_to_markov_train_states(self.data_fit, self.lookback_len)
        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return w
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            check_contexts_shape(
                eval_right_on, self.lookback_len, is_inference_data=True
            )
            kernel_Xin_X_or_Y = self.kernel(eval_right_on, X_fit)
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

    
    def _init_kernels(self, X: np.ndarray, Y: np.ndarray):

        K_X = self.kernel(X)
        return K_X
    def kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        X = X.reshape(X.shape[0], -1)
        if Y is not None:
            Y = Y.reshape(Y.shape[0], -1)
        exp =1.0

        return self._kernel(X, Y)*exp
    def _build_weights(self,weights):
        weights_b, weights_a = contexts_to_markov_train_states(weights, self.lookback_len)
        weights_b=weights_b.reshape(-1)
        return np.einsum("i,j->ij",weights_b,weights_b)

    
    def _build_matrices(self, data, forces, friction):
        #self.kernel_Y = self.return_dk(data, forces, friction) 
        if self.transform ==False:
            self.kernel_Y  = self.return_mixed_term(data, forces,friction)
            self.kernel_YX = self.return_mixed_term(data, forces, friction)  # no importnance here
        else:
            self.kernel_Y  = self.return_mixed_term_descriptors(data, forces,friction)
            self.kernel_YX = self.return_mixed_term_descriptors(data, forces, friction)
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
        if self.transform:
            X_fit = self.transform_to_distances(X_fit)
        self.kernel_X = self._init_kernels(X_fit,Y_fit)
        self.data_fit = data

        if hasattr(self, "_eig_cache"):
            del self._eig_cache
    
    def transform_to_distances(self,X):
        X = X.reshape(X.shape[0],-1)
        n_samples = X.shape[0]
        positions_velocities = X.reshape(n_samples,-1,6)
        velocities = positions_velocities[:,:,:3]
        positions = positions_velocities[:,:,3:]
        distances = self.compute_distances(positions)
        jacobian = vmap(jacrev(self.compute_distances_one))(positions)
        new_velocities = jnp.einsum("ijkl,ikl->ij",jacobian,velocities)
        return jnp.concatenate((new_velocities,distances)).reshape(n_samples,-1)

    def compute_kernel_one(self,X,Y):

        positions_velocities = X.reshape(-1,6)
        velocities = positions_velocities[:,:3]
        positions = positions_velocities[:,3:]
        length_scale = self._kernel.length_scale
        distances = self.compute_distances_one(positions)
        jacobian = jacrev(self.compute_distances_one)(positions)
        new_velocities = jnp.einsum("jkl,kl->j",jacobian,velocities)
        features = jnp.concatenate((new_velocities,distances)).reshape(-1)  
        return jnp.exp(-jnp.linalg.norm((features - Y)/length_scale,axis=-1)**2)
    
    def compute_kernel_hessian(self,X):
        n_samples = X.shape[0]
        kernel_jacobian = np.zeros((n_samples,n_samples, X.shape[1]))
        kernel_hessian = np.zeros((n_samples,n_samples, X.shape[1],X.shape[1]))
        x_transformed = self.transform_to_distances(X)
        print("here")

        kernel_jacobian = vmap(jacrev(self.compute_kernel_one,argnums=0),in_axes=(0,None))(X,x_transformed)
        print(kernel_jacobian)
        
        kernel_hessian = vmap(hessian(self.compute_kernel_one,argnums=0),in_axes=(0,None))(X,x_transformed)
        return kernel_jacobian, kernel_hessian
    
    def return_mixed_term(self,data, forces:np.ndarray, friction:np.float):

        sigma = self._kernel.length_scale
        
        X, Y_fit = contexts_to_markov_train_states(data, self.lookback_len)
        X = X.reshape(X.shape[0],-1)
        kern=self._kernel(X,X)
        forces, Y_fit = contexts_to_markov_train_states(forces, self.lookback_len)
        forces = forces.reshape(forces.shape[0],-1)
        difference = (X[:, np.newaxis, :] - X[np.newaxis, :, :])
        n =  difference.shape[2]

        dk=  (np.einsum('ik,ijk,k->ij', -forces, difference,1/sigma**2) + 0.5* np.einsum('ijk,k,k->ij', difference**2,friction,1/sigma**4) - 0.5 * np.einsum("k,k->",friction, 1/ sigma**2) )* self.kernel_X        #return self.gaussianG00G10(X,forces,friction,sigma)*self.kernel_X

        return dk
    
    def return_mixed_term_descriptors(self,data, forces:np.ndarray, friction:np.float):

        sigma = self._kernel.length_scale
        
        X, Y_fit = contexts_to_markov_train_states(data, self.lookback_len)
        X = X.reshape(X.shape[0],-1)
        n_samples = X.shape[0]

        forces, Y_fit = contexts_to_markov_train_states(forces, self.lookback_len)
        forces = forces.reshape(forces.shape[0],-1)
        difference = (X[:, np.newaxis, :] - X[np.newaxis, :, :])
        n =  difference.shape[2]
        jacobian, hessian = self.compute_kernel_hessian(X)
        print("here2")
        dk=  (np.einsum('ik,ijk->ij', -forces, jacobian) + 0.5* np.einsum('ijkk,k->ij', hessian,friction) )* self.kernel_X        #return self.gaussianG00G10(X,forces,friction,sigma)*self.kernel_X

        return dk
    
    def compute_distances(self,positions):
        distance_tensor = jnp.zeros((positions.shape[0], 45))
        idx = 0
        for i in range(positions.shape[1]):
            for j in range(i + 1, positions.shape[1]):
                distance_ij = jnp.sqrt(jnp.sum((positions[:, i, :] - positions[:, j, :]) ** 2,axis=-1))
            # Use out-of-place operation to assign values
                distance_tensor= distance_tensor.at[:, idx].set(distance_ij)
                idx += 1
        return distance_tensor
    
    def compute_distances_one(self,positions):   
        distance_tensor = jnp.zeros(( 45))
        idx = 0
        for i in range(positions.shape[0]):
            for j in range(i + 1, positions.shape[0]):
                distance_ij = jnp.sqrt(jnp.sum((positions[ i, :] - positions[ j, :]) ** 2,axis=-1))
            # Use out-of-place operation to assign values
                distance_tensor= distance_tensor.at[idx].set(distance_ij)
                idx += 1
        return distance_tensor
    

    
