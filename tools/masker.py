
import torch
import numpy as np


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

class Masker:
    def __init__(self, mask_type, mask_ratio, grid_size, **kwargs):
        self.mask_ratio = mask_ratio
        if mask_type == "random_roi":
            self.masking_strategy = self.random_masking_within_roi
        elif mask_type == "random_roi_blockwise":
            self.masking_strategy = self.random_masking_within_roi_blockwise
        elif mask_type == "random":
            self.masking_strategy = self.random_masking
        else:
            raise NotImplementedError
    
    def __call__(self, x, roi_mask=None):
        """
        x: [N, L, D], sequence
        x_masked: [N, L * mask_ratio, D], masked sequence
        mask: [N, L], binary mask
        ids_restore: [N, L], indices to restore the original order
        """
        mask, ids_restore, ids_keep = self.masking_strategy(input_size=x.shape, device=x.device, roi_mask=roi_mask)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        return x_masked, mask, ids_restore, ids_keep
    
    def call_masking_fctn(self, x, fctn_name, **kwargs):
        fctn = eval(f"self.{fctn_name}")
        mask, ids_restore, ids_keep = fctn(input_size=x.shape, device=x.device, **kwargs)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        return x_masked, mask, ids_restore
        
    def random_masking(self, input_size, device, **kwargs):
        """
        # Reference: https://github.com/facebookresearch/mae.git
        
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        
        input_size: [N, L, D], sequence
        device: torch.device
        mask: [N, L], binary mask
        ids_restore: [N, L], indices to restore the original order
        """
        #print("Ranodm Masking")
        N, L, D = input_size  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore, ids_keep

    def random_masking_plus_given_index(self, input_size, device, given_index, **kwargs):
        """
        Perform per-sample random masking by per-sample shuffling and also mask out the given index.
        input_size: [N, L, D], sequence
        given_index: [L_g], given index to be masked out
        device: torch.device
        mask: [N, L], binary mask
        ids_restore: [N, L], indices to restore the original order
        """
        N, L, D = input_size
        maskout_index_total = np.union1d(given_index, np.random.choice(L, int(L * self.mask_ratio), replace=False))
        len_keep = L - len(maskout_index_total)
        mask = torch.zeros([N, L], device=device)
        mask[:, maskout_index_total] = 1
        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        return mask, ids_restore, ids_keep

    def random_masking_within_roi(self, input_size, device, roi_mask, **kwargs):
        """
        Perform random masking only within the ROI region defined by `roi_mask`.

        Args:
            input_size: Tuple[int, int, int], dimensions of the input sequence (e.g., [N, L, D]).
            device: torch.device.
            roi_mask: [N, L], binary mask indicating ROI patches after patchification.

        Returns:
            mask: [N, L], binary mask (0 is keep, 1 is masked).
            ids_restore: [N, L], indices to restore the original order.
            ids_keep: [N, len_keep], indices of kept patches.
        """
        #print("Random Masking within ROI")

        num_roi_patches = torch.sum(roi_mask, dim=1)

        N, L, D = input_size  # Extract shape directly

        # Ensure roi_mask is binary and matches the input
        assert roi_mask.shape == (N, L), "ROI mask must have shape [N, L]"
        #print(f"ROI mask binary shape: {roi_mask.shape}")  # Expected: [4, 5376]

        # Calculate the total number of spatches to mask
        #num_to_mask = int(L * self.mask_ratio)  # Fixed number of patches to mask per sample
        num_to_mask = int(torch.min(num_roi_patches).item() * self.mask_ratio)  # Mask a fraction of ROI patches
        #num_to_mask = num_roi_patches * self.mask_ratio

        # Initialize mask as all zeros (unmasked)
        mask = torch.zeros([N, L], device=device)

        # For each sample in the batch
        for i in range(N):
            # Get indices of ROI patches
            roi_indices = torch.nonzero(roi_mask[i], as_tuple=True)[0]  # Indices of ROI patches for sample `i`

            if len(roi_indices) == 0:
                raise ValueError(f"Sample {i} has no ROI patches to mask.")

            # Shuffle ROI indices
            shuffled_indices = roi_indices[torch.randperm(len(roi_indices))]

            # Mask the first `num_to_mask` ROI patches
            num_to_mask_i = min(num_to_mask, len(roi_indices))  # Limit to the number of available ROI patches
            mask[i, shuffled_indices[:num_to_mask_i]] = 1  # Set these patches as masked

        # Generate ids_restore for reordering patches
        noise = torch.rand(N, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Recompute ids_keep: Indices where mask == 0 (patches to keep)
        ids_keep = torch.nonzero(mask == 0, as_tuple=True)[1]
        ids_keep = ids_keep.reshape(N, -1)  # Reshape to [N, len_keep]

        return mask, ids_restore, ids_keep
    
    def random_masking_within_roi_blockwise(self, input_size, device, roi_mask, **kwargs):
        """
        Perform random masking only within the ROI region defined by `roi_mask`.
        Args:
            input_size: Tuple[int, int, int], dimensions of the input sequence (e.g., [N, L, D]).
            device: torch.device.
            roi_mask: [N, 2 * L], binary mask indicating ROI patches for both contrasts.

        Returns:
            mask: [N, 2 * L], binary mask (0 is keep, 1 is masked).
            ids_restore: [N, 2 * L], indices to restore the original order.
            ids_keep: [N, len_keep], indices of kept patches.
        """
        print("Random Masking within ROI blockwise")
        N, L, D = input_size  # Batch size, total patches, embedding dimension
        num_patches_per_channel = L // 2  # Divide total patches equally for two channels

        # Reshape the mask to separate channels
        roi_mask = roi_mask.reshape(N, 2, num_patches_per_channel)  # [N, 2, num_patches_per_channel]


        # Verify that both channels have identical masks
        assert torch.all(roi_mask[:, 0, :] == roi_mask[:, 1, :]), "Masks for both channels must be identical."
        roi_mask_collapsed = roi_mask[:, 0, :]  # Use only one channel's mask

        # Number of ROI patches per sample
        num_roi_patches = torch.sum(roi_mask_collapsed, dim=1)  # [N]

        # Determine the number of patches to mask
        num_to_mask = (num_roi_patches * self.mask_ratio).long()

        # Initialize mask as all zeros (unmasked)
        mask_per_channel = torch.zeros([N, num_patches_per_channel], device=device)

        for i in range(N):
            # Get indices of ROI patches
            roi_indices = torch.nonzero(roi_mask_collapsed[i], as_tuple=True)[0]

            if len(roi_indices) == 0:
                raise ValueError(f"Sample {i} has no ROI patches to mask.")

            # Shuffle ROI indices
            shuffled_indices = roi_indices[torch.randperm(len(roi_indices))]

            # Mask the first `num_to_mask` ROI patches
            num_to_mask_i = min(num_to_mask[i].item(), len(roi_indices))
            mask_per_channel[i, shuffled_indices[:num_to_mask_i]] = 1

        # Repeat the mask for both channels
        mask = mask_per_channel.unsqueeze(1).repeat(1, 2, 1)  # [N, 2, num_patches_per_channel]
        assert mask[:, 0, :].equal(mask[:, 1, :]), "Masks for both channels must be identical."
        mask = mask.reshape(N, L)  # Flatten back to [N, L]

        # Generate ids_restore for reordering patches
        noise = torch.rand(N, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Recompute ids_keep: Indices where mask == 0 (patches to keep)
        ids_keep = torch.nonzero(mask == 0, as_tuple=True)[1]
        ids_keep = ids_keep.reshape(N, -1)  # Reshape to [N, len_keep]

        return mask, ids_restore, ids_keep