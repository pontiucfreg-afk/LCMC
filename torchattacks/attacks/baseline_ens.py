import math
import os
import cv2
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import ToPILImage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image


class LCMC:
    """
    Ensemble-based adversarial attack method
    """

    def __init__(self, models, eps=8 / 255, alpha=1 / 255, steps=8, momentum=0.9, targeted=False):
        self.models = models
        self.eps = eps  # Perturbation bound
        self.steps = steps  # Number of iterations
        self.momentum = momentum
        self.targeted = targeted
        self.alpha = -alpha if self.targeted else alpha  # Step size direction

        # Feature layers for different model architectures
        self.feature_layers = {
            'resnet': ['layer2', 'layer3'],
            'alexnet': ['4', '7']
        }
        self.mid_outputs = []  # Store intermediate layer outputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _get_mid_output(self, module, input, output):
        """Hook function to capture intermediate outputs"""
        self.mid_outputs.append(output)

    def forward(self, images, labels, batch_size):
        # Set models to evaluation mode
        for model in self.models:
            model.eval()

        # Initialize variables on GPU
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        modifier = torch.nn.Parameter(torch.full_like(images, 0.01 / 255, device='cuda'), requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.alpha)

        # Register hooks for feature extraction
        hooks = []
        # ResNet features
        for layer in self.feature_layers['resnet']:
            hooks.append(self.models[0][1]._modules[layer].register_forward_hook(self._get_mid_output))
        self.models[0](images)
        mid_originals_res = [o.detach().clone() for o in self.mid_outputs]
        self.mid_outputs.clear()

        # AlexNet features
        for layer in self.feature_layers['alexnet']:
            hooks.append(
                self.models[1][1]._modules['features']._modules[layer].register_forward_hook(self._get_mid_output))
        self.models[1](images)
        mid_originals_alex = [o.detach().clone() for o in self.mid_outputs]
        self.mid_outputs.clear()

        # Iterative perturbation update
        for _ in range(self.steps):
            adv_images = torch.clamp(images + torch.clamp(modifier, -self.eps, self.eps), 0, 1)
            total_exp_loss = 0.0
            losses = []

            # Calculate loss for ResNet
            self.models[0](adv_images)
            res_loss = self._calculate_model_loss(mid_originals_res, self.mid_outputs, batch_size)
            total_exp_loss += torch.exp(res_loss)
            losses.append(res_loss)
            self.mid_outputs.clear()

            # Calculate loss for AlexNet
            self.models[1](adv_images)
            alex_loss = self._calculate_model_loss(mid_originals_alex, self.mid_outputs, batch_size)
            total_exp_loss += torch.exp(alex_loss)
            losses.append(alex_loss)
            self.mid_outputs.clear()

            # Ensemble loss with dynamic weights
            weights = [torch.exp(l) / total_exp_loss for l in losses]
            ens_loss = sum(l * w for l, w in zip(losses, weights))

            # Update perturbation
            optimizer.zero_grad()
            ens_loss.backward()
            optimizer.step()

        # Cleanup hooks
        for hook in hooks:
            hook.remove()

        return torch.clamp(images + torch.clamp(modifier, -self.eps, self.eps), 0, 1).detach()

    def _calculate_model_loss(self, original_features, adv_features, batch_size):
        """Calculate weighted loss for intermediate features"""
        mid_losses = []
        adj_losses = []
        total_mid_exp = 0.0
        total_adj_exp = 0.0

        # Compute individual layer losses
        for orig, adv in zip(original_features, adv_features):
            orig_flat = orig.reshape(batch_size, -1)
            adv_flat = adv.reshape(batch_size, -1)

            mid_loss = F.cosine_similarity(orig_flat, adv_flat).mean()
            adj_loss = 0.1 * F.cosine_similarity(adv_flat[:-1], adv_flat[1:]).mean()

            mid_losses.append(mid_loss)
            adj_losses.append(adj_loss)
            total_mid_exp += torch.exp(mid_loss)
            total_adj_exp += torch.exp(adj_loss)

        # Weighted sum of losses
        mid_weights = [torch.exp(l) / total_mid_exp for l in mid_losses]
        adj_weights = [torch.exp(l) / total_adj_exp for l in adj_losses]

        weighted_mid = sum(l * w for l, w in zip(mid_losses, mid_weights))
        weighted_adj = sum(l * w for l, w in zip(adj_losses, adj_weights))

        return weighted_mid + weighted_adj