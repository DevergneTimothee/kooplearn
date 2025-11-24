import logging
import weakref
from copy import deepcopy
from typing import Optional

import lightning
import numpy as np
import torch
import functorch

from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn.abc import TrainableFeatureMap
from kooplearn.nn.functional import (
    log_fro_metric_deformation_loss,
    relaxed_projection_score,
    vamp_score,
)
from kooplearn.models.feature_maps.dpnets import DPNet, DPModule
logger = logging.getLogger("kooplearn")

def generator_score(cov_x, cov_xy):
    return torch.trace(torch.mm(torch.inverse(cov_x+1e-5*torch.eye(cov_x.size(0),device=cov_x.device)),cov_xy))


class DPGen(DPNet):
    """Implements the DPNets :footcite:p:`Kostic2023DPNets` feature map, which learn an invariant representation of time-homogeneous stochastic dynamical systems. Can be used in conjunction to :class:`kooplearn.models.DeepEDMD` to learn a Koopman/Transfer operator from data. The DPNet feature map is trained using the :class:`lightning.LightningModule` API, and can be trained using the :class:`lightning.Trainer` API. See the `PyTorch Lightning documentation <https://pytorch-lightning.readthedocs.io/en/latest/>`_ for more information.

    Args:
        encoder (torch.nn.Module): Encoder network. Should be a subclass of :class:`torch.nn.Module`. Will be initialized as ``encoder(**encoder_kwargs)``.
        optimizer_fn (torch.optim.Optimizer): Any optimizer from :class:`torch.optim.Optimizer`.
        trainer (lightning.Trainer): An initialized `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ object used to train the DPNet feature map.
        use_relaxed_loss (bool, optional): Whether to use the relaxed projection score introduced in :footcite:t:`Kostic2023DPNets`. Might be slower to convergence, but is much more stable in ill-conditioned problems.  Defaults to False.
        metric_deformation_loss_coefficient (float, optional): Coefficient of the metric deformation loss. Defaults to 1.0.
        encoder_kwargs (dict, optional): Dictionary of keyword arguments passed to the encoder network upon initialization. Defaults to ``{}``.
        optimizer_kwargs (dict): Dictionary of keyword arguments passed to the optimizer at initialization. Defaults to ``{}``.
        encoder_timelagged (Optional[torch.nn.Module], optional): Encoder network for the time-lagged data. Defaults to None. If None, the encoder network is used for time-lagged data as well. If not None, it will be initialized as ``encoder_timelagged(**encoder_timelagged_kwargs)``.
        encoder_timelagged_kwargs (dict, optional): Dictionary of keyword arguments passed to `encoder_timelagged` upon initialization. Defaults to ``{}``.
        center_covariances (bool, optional): Wheter to compute the VAMP score with centered covariances. Defaults to False.
        seed (int, optional): Seed of the internal random number generator. Defaults to None.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        optimizer_fn: torch.optim.Optimizer,
        trainer: lightning.Trainer,
        use_relaxed_loss: bool = False,
        metric_deformation_loss_coefficient: float = 1.0,  # That is, the parameter γ in the paper.
        encoder_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        encoder_timelagged: Optional[torch.nn.Module] = None,
        encoder_timelagged_kwargs: dict = {},
        center_covariances: bool = False,
        seed: Optional[int] = None,
        friction=0.0,
        forces = None
    ):
        super().__init__(encoder,optimizer_fn,trainer,use_relaxed_loss,metric_deformation_loss_coefficient,encoder_kwargs,optimizer_kwargs,encoder_timelagged,encoder_timelagged_kwargs,center_covariances,seed)
        self.lightning_module = DPModuleGen(
            encoder,
            optimizer_fn,
            optimizer_kwargs,
            use_relaxed_loss=use_relaxed_loss,
            metric_deformation_loss_coefficient=metric_deformation_loss_coefficient,
            encoder_kwargs=encoder_kwargs,
            encoder_timelagged=encoder_timelagged,
            encoder_timelagged_kwargs=encoder_timelagged_kwargs,
            center_covariances=center_covariances,
            kooplearn_feature_map_weakref=weakref.ref(self),
            friction=friction,
        )
        self.friction=friction
    def __call__(self, X: np.ndarray, Y: np.ndarray=None) -> np.ndarray:
        X = torch.from_numpy(X.copy(order="C")).float()
        self.lightning_module.eval()
        embedded_X = self.lightning_module.encoder(
            X.to(self.lightning_module.device)
        )
        if Y is not None:
            Y = torch.from_numpy(Y.copy(order="C")).float()
            
            compute_batch_jacobian = functorch.vmap(functorch.jacrev(lambda t: self.lightning_module.encoder(t)))
            gradient = compute_batch_jacobian(X.to(self.lightning_module.device)).squeeze(1)
            compute_batch_hessian = functorch.vmap(functorch.hessian(lambda t: self.lightning_module.encoder(t)))

            hessian = compute_batch_hessian(X.to(self.lightning_module.device)).squeeze(1)

            force_part = torch.einsum("ijk,ik->ij",gradient,Y.to(self.lightning_module.device))


            encoded_Y = (force_part + 0.5 * self.friction **2 * torch.einsum("ijkk->ij",hessian)).cpu().detach().numpy()
            embedded_X = embedded_X.detach().cpu().numpy()
            return embedded_X, encoded_Y
        else:
            return embedded_X.detach().cpu().numpy()




class DPModuleGen(DPModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        optimizer_fn: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        use_relaxed_loss: bool = False,
        metric_deformation_loss_coefficient: float = 1.0,  # That is, the parameter γ in the paper.
        encoder_kwargs: dict = {},
        encoder_timelagged: Optional[torch.nn.Module] = None,
        encoder_timelagged_kwargs: dict = {},
        center_covariances: bool = True,
        kooplearn_feature_map_weakref=None,
        friction = 0.0,
    ):
        super().__init__(encoder, optimizer_fn,optimizer_kwargs,use_relaxed_loss,metric_deformation_loss_coefficient,encoder_kwargs,encoder_timelagged,encoder_timelagged_kwargs,center_covariances,kooplearn_feature_map_weakref)

        self.friction = friction

    def training_step(self, train_batch, batch_idx):
        X, Y = train_batch[:, 0, ...], train_batch[:, 1, ...]#Positions in 0, forces in 1

        encoded_X= self.forward(X)
        gradient = torch.ones_like(encoded_X,dtype=torch.float32)

        #j= torch.autograd.functional.jacobian(lambda t: self.forward(t),X, create_graph=True)
        #gradient_X = gradient_X.permute(2, 0, 1)

        #print(j.size())
        #gradient_X = torch.diagonal(j, offset=0, dim1=0, dim2=2) #see https://stackoverflow.com/questions/54754153/autograd-grad-for-tensor-in-pytorch
        #gradient_X = gradient_X.permute(2, 0, 1)

        #hessian = torch.zeros((gradient_X.size(0),gradient_X.size(1),gradient_X.size(2),gradient_X.size(2)))
# Compute the Hessian for each element in the output with respect to each pair of elements in the input
        
        #for i in range(gradient_X.size(0)):
        #    for j in range(gradient_X.size(1)):
        #        h= torch.autograd.functional.hessian(lambda t: self.forward(t)[i,j],X, create_graph=True)
        #        dydx2 = torch.diagonal(h, offset=0, dim1=0, dim2=2) #dydx2: (output_dim, input_dim, batch_size)
        #        dydx2 = dydx2.permute(2, 0, 1) #dydx2: (batch_size, output_dim, input_dim)
        #        hessian[i,j] = dydx2[i]
        compute_batch_jacobian = functorch.vmap(functorch.jacrev(lambda t: self.encoder(t)))
        gradient = compute_batch_jacobian(X).squeeze(1)

        compute_batch_hessian = functorch.vmap(functorch.hessian(lambda t: self.encoder(t)))

        hessian = compute_batch_hessian(X).squeeze(1)

        force_part = torch.einsum("ijk,ik->ij",gradient,Y)

        encoded_Y = force_part + 0.5 * self.friction **2 * torch.einsum("ijkk->ij",hessian)

        _norm = torch.rsqrt(torch.tensor(encoded_X.shape[0]))
        encoded_X *= _norm
        encoded_Y *= _norm

        cov_X = torch.mm(encoded_X.T, encoded_X)
        cov_XY = torch.mm(encoded_X.T, encoded_Y)

        metrics = {}
        # Compute the losses

        svd_loss = -1 * generator_score(cov_X, cov_XY)
        metrics["train/generator_score"] = -1.0 * svd_loss.item()

        if self.hparams.metric_deformation_loss_coefficient > 0.0:
            metric_deformation_loss = 0.5 * (
                log_fro_metric_deformation_loss(cov_X)
            )
            print(self.hparams.metric_deformation_loss_coefficient)
            #metric_deformation_loss *= self.hparams.metric_deformation_loss_coefficient
            metric_deformation_loss *= 50
            metrics["train/metric_deformation_loss"] = metric_deformation_loss.item()
            svd_loss += metric_deformation_loss
        metrics["train/total_loss"] = svd_loss.item()
        self.log_dict(metrics, on_step=True, prog_bar=True, logger=True)
        return svd_loss
    def forward(self, X: torch.Tensor, time_lagged: bool = False) -> torch.Tensor:
        # Caution: this method is designed only for internal calling by the DPNet feature map.
        batch_size = X.shape[0]
        trail_dims = X.shape[1:]
        X = X.view(batch_size, *trail_dims)
        if time_lagged:
            encoded_X = self.encoder_timelagged(X)
        else:
            encoded_X = self.encoder(X)
        trail_dims = encoded_X.shape[1:]
        encoded_X = encoded_X.view(batch_size, *trail_dims)
        return encoded_X.view(batch_size, -1)

