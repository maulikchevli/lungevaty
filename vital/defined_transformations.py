import torch.nn.functional as F
from einops import rearrange
import torch
from monai import transforms
from monai.transforms.intensity.array import ScaleIntensityRange
import SimpleITK as sitk
import numpy as np


def pad_tensor(tensor, desired_shape, pad_value=None):
    """
    Pads the input tensor at the end of each spatial dimension to reach the desired shape.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [C, D, H, W] or [D, H, W].
        desired_shape (tuple): Desired spatial shape as (D_new, H_new, W_new).
        pad_value (float or bool, optional): Value to pad with. Defaults to 0 for numeric tensors and False for boolean tensors.
    
    Returns:
        torch.Tensor: Padded tensor.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # Add channel dimension: [1, D, H, W]
    
    C, D, H, W = tensor.shape
    D_new, H_new, W_new = desired_shape
    
    # Calculate padding sizes: (W_left, W_right, H_left, H_right, D_left, D_right)
    # Since padding is at the end, lefts are 0
    pad_depth = D_new - D
    pad_height = H_new - H
    pad_width = W_new - W
    
    if pad_depth < 0 or pad_height < 0 or pad_width < 0:
        raise ValueError("Desired shape must be greater than or equal to the tensor's current shape in all dimensions.")
    
    # Determine default pad_value based on tensor dtype if not provided
    if pad_value is None:
        if torch.is_floating_point(tensor):
            pad_value = 0.0
        elif torch.is_bool(tensor):
            pad_value = False
        else:
            raise ValueError("Unsupported tensor dtype. Provide a pad_value for non-floating and non-boolean tensors.")
    
    # Apply padding
    padded_tensor = F.pad(
        tensor, 
        (0, pad_width, 0, pad_height, 0, pad_depth), 
        mode='constant', 
        value=pad_value
    )
    
    # If original tensor was 3D, remove the added channel dimension
    # if padded_tensor.size(0) == 1:
    #     padded_tensor = padded_tensor.squeeze(0)
    
    return padded_tensor


def extract_patches(image_batch, patch_size):
    """
    Extracts non-overlapping patches from a batch of 3D images.
    
    Args:
        image_batch (torch.Tensor): Padded images tensor of shape [B, D, H, W]
        patch_size (tuple): Patch size (pD, pH, pW)
    
    Returns:
        torch.Tensor: Extracted patches of shape [B, num_patches, pD*pH*pW]
    """
    C, D, H, W = image_batch.shape
    pD, pH, pW = patch_size
    # Rearrange to extract patches
    patches = rearrange(image_batch, 
                       'b (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3)', 
                       p1=pD, p2=pH, p3=pW)
    return patches

def reconstruct_from_patches(patches, image_shape, patch_size):
    """
    Reconstructs a batch of 3D images from non-overlapping patches.
    
    Args:
        patches (torch.Tensor): Tensor of patches with shape [B, num_patches, pD*pH*pW]
        image_shape (tuple): Original image shape (B, D, H, W)
        patch_size (tuple): Patch size (pD, pH, pW)
        
    Returns:
        torch.Tensor: Reconstructed image batch of shape [B, D, H, W]
    """
    B, D, H, W = image_shape
    pD, pH, pW = patch_size
    
    # Rearrange to reconstruct the original grid
    reconstructed_images = rearrange(patches, 
                                     'b (d h w) (p1 p2 p3) -> b (d p1) (h p2) (w p3)', 
                                     d=D // pD, h=H // pH, w=W // pW, 
                                     p1=pD, p2=pH, p3=pW)
    
    return reconstructed_images

class CalculatePhysicalSize(transforms.MapTransform):
    def __init__(self, keys, spacing):
        self.spacing = spacing
    
    def __call__(self, data):
        img_shape = data['image'].shape
        physical_shape = (self.spacing[0] * img_shape[0], self.spacing[1] * img_shape[1], self.spacing[2] * img_shape[2]) 
        data['size'] = physical_shape
        return data

class Patchify(transforms.MapTransform):
    def __init__(self, keys, patch_size):
        super().__init__(keys)
        self.patch_size = patch_size

    def __call__(self, data):
        image = data["image"]

        patched_image = extract_patches(image,self.patch_size)
        data["image"] = patched_image.squeeze()

        return data
    
class Lateralityd(transforms.MapTransform):
    def __init__(self, keys, patch_size):
        super().__init__(keys)
        self.patch_size = patch_size

    def __call__(self, data):
        annotation_mask = data["mask"]
        lobes = False
        sides = False
        #data['annotation'] = torch.zeros_like(annotation_mask)

        laterality = data["cancer_laterality"]

        if laterality[1]:
            lobes = True
            label = laterality[1] - 1
        elif laterality[0] == 3:
            label = 1 # label 1, lobes False means right lung
            sides = True
        elif laterality[0] == 4:
            label = 0 # label 0, lobes False means left lung
            sides = True
        else:
            label = -1 # label -1 whole lung

        patched_annotation_mask = extract_patches(annotation_mask, self.patch_size).squeeze()
        has_positive = (patched_annotation_mask > 0).any(dim=1)
        result = torch.zeros(patched_annotation_mask.size(0), dtype=patched_annotation_mask.dtype, device=patched_annotation_mask.device)

        # Get all unique values > 0 across the tensor
        unique_pos = torch.unique(patched_annotation_mask[patched_annotation_mask > 0])

        # For rows with positive elements, compute mode among > 0
        pos_rows = patched_annotation_mask[has_positive]
        expanded = pos_rows.unsqueeze(-1) == unique_pos
        counts = expanded.sum(dim=1)
        max_idx = counts.argmax(dim=1)
        result[has_positive] = unique_pos[max_idx]

        patched_annotation = extract_patches(data["annotation"], self.patch_size)
        data["annotation"] = patched_annotation.squeeze()

        data["laterality"] = result
        data['laterality_label'] = label
        data['lobes'] = lobes
        data['sides'] = sides

        data.pop("mask")
        data.pop("series")
        data.pop("study")
        data.pop("pid")
        data.pop("screen_timepoint")
        data.pop("institution")
        data.pop("cancer_laterality")
        return data
    
class Permuted(transforms.MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        image = data["image"]
        # Permute the image dimensions for 3d conv patch embedding
        permuted_image = image.permute(0, 3, 1, 2)
        data['image'] = permuted_image
        if 'mask' in self.keys:
            permuted_mask = data["mask"].permute(0, 3, 1, 2)
            data['mask'] = permuted_mask

        if 'annotation' not in self.keys:
            return data
        if data.get('annotation', None) is None:
            # create a dummy annotation if it does not exist
            data['annotation'] = torch.zeros_like(permuted_image)
        else:
            annotation = data['annotation'].permute(0, 3, 1, 2)
            data['annotation'] = annotation
        return data


class PermuteLongid(transforms.MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            # Permute the image dimensions for 3d conv patch embedding
            permuted_image = image.permute(0, 3, 1, 2)
            data[key] = permuted_image
        return data



class MaskPatchesd(transforms.MapTransform):
    def __init__(self, keys, patch_size, hull_only=False, use_annotations=True):
        super().__init__(keys)
        self.patch_size = patch_size
        self.hull_only = hull_only
        self.use_annotations = use_annotations

    def __call__(self, data):
        image = data["image"]

        patched_image = extract_patches(image,self.patch_size)
        data["image"] = patched_image.squeeze()

        annotation = data.get('annotation', None)
        if annotation is not None and self.use_annotations:
            patched_annotation = extract_patches(annotation, self.patch_size)
            data["annotation"] = patched_annotation.squeeze()
            data['has_annotation'] = True
        else:
            annotation_mask = data["mask"].clone()
            laterality = data["cancer_laterality"]

            if self.hull_only:
                annotation_mask[annotation_mask > 0] = 1 #select only hull
            elif laterality[1]:
                annotation_mask[annotation_mask != laterality[1]] = 0
            elif laterality[0] == 3:
                annotation_mask[annotation_mask < 4] = 0 #select right lung only
            elif laterality[0] == 4:
                annotation_mask[annotation_mask == 1] = 0 #get rid of hull
                annotation_mask[annotation_mask > 3] = 0 #select left lung only

            annotation_mask[annotation_mask > 0] = 1
            patched_annotation_mask = extract_patches(annotation_mask, self.patch_size).squeeze()
            data["annotation"] = patched_annotation_mask
            data['has_annotation'] = False

        data.pop("mask")
        data.pop("series")
        data.pop("study")
        data.pop("pid")
        data.pop("screen_timepoint")
        data.pop("institution")
        data.pop("cancer_laterality")

        return data

class MaskPatchesNoPopd(transforms.MapTransform):
    def __init__(self, keys, patch_size, hull_only=False, use_annotations=True):
        super().__init__(keys)
        self.patch_size = patch_size
        self.hull_only = hull_only
        self.use_annotations = use_annotations

    def __call__(self, data):
        image = data["image"]

        patched_image = extract_patches(image,self.patch_size)
        data["image"] = patched_image.squeeze()

        annotation = data.get('annotation', None)
        if annotation is not None and self.use_annotations:
            patched_annotation = extract_patches(annotation, self.patch_size)
            data["annotation"] = patched_annotation.squeeze()
            data['has_annotation'] = True
        else:
            annotation_mask = data["mask"].clone()
            laterality = data["cancer_laterality"]

            if self.hull_only:
                annotation_mask[annotation_mask > 0] = 1 #select only hull
            elif laterality[1]:
                annotation_mask[annotation_mask != laterality[1]] = 0
            elif laterality[0] == 3:
                annotation_mask[annotation_mask < 4] = 0 #select right lung only
            elif laterality[0] == 4:
                annotation_mask[annotation_mask == 1] = 0 #get rid of hull
                annotation_mask[annotation_mask > 3] = 0 #select left lung only

            annotation_mask[annotation_mask > 0] = 1
            patched_annotation_mask = extract_patches(annotation_mask, self.patch_size).squeeze()
            data["annotation"] = patched_annotation_mask
            data['has_annotation'] = False

        return data

class NoNodulesNoPopd(transforms.MapTransform):
    def __init__(self, keys, patch_size, hull_only=False, use_annotations=True):
        super().__init__(keys)
        self.patch_size = patch_size
        self.hull_only = hull_only
        self.use_annotations = use_annotations

    def __call__(self, data):
        image = data["image"]
        annotation = data.get('annotation', None)
        if annotation is not None and self.use_annotations:
            annotation_mask = (annotation > 0)
            image[annotation_mask] = -1

            patched_annotation = extract_patches(annotation, self.patch_size)
            data["annotation"] = patched_annotation.squeeze()
            data['has_annotation'] = True
        else:
            annotation_mask = data["mask"].clone()
            laterality = data["cancer_laterality"]

            if self.hull_only:
                annotation_mask[annotation_mask > 0] = 1 #select only hull
            elif laterality[1]:
                annotation_mask[annotation_mask != laterality[1]] = 0
            elif laterality[0] == 3:
                annotation_mask[annotation_mask < 4] = 0 #select right lung only
            elif laterality[0] == 4:
                annotation_mask[annotation_mask == 1] = 0 #get rid of hull
                annotation_mask[annotation_mask > 3] = 0 #select left lung only

            annotation_mask[annotation_mask > 0] = 1
            patched_annotation_mask = extract_patches(annotation_mask, self.patch_size).squeeze()
            data["annotation"] = patched_annotation_mask
            data['has_annotation'] = False


        patched_image = extract_patches(image,self.patch_size)
        data["image"] = patched_image.squeeze()
        return data


class AnnotationToNegativeImaged(transforms.MapTransform):
    def __init__(self, keys, annotation_key="annotation", threshold=0.5, fill_value=-1.0):
        super().__init__(keys)
        self.annotation_key = annotation_key
        self.threshold = threshold
        self.fill_value = fill_value

    def __call__(self, data):
        annotation = data.get(self.annotation_key)
        if annotation is None:
            return data

        target_device = None
        for key in self.keys:
            img = data.get(key)
            if torch.is_tensor(img):
                target_device = img.device
                break

        if torch.is_tensor(annotation):
            annotation_tensor = annotation
        else:
            annotation_tensor = torch.as_tensor(annotation)

        if target_device is not None and annotation_tensor.device != target_device:
            annotation_tensor = annotation_tensor.to(target_device)

        if target_device is None:
            target_device = annotation_tensor.device

        mask = annotation_tensor > self.threshold
        if not mask.any():
            return data

        data[self.annotation_key] = annotation_tensor

        for key in self.keys:
            image = data.get(key)
            if image is None:
                continue

            if torch.is_tensor(image):
                image_tensor = image
                if image_tensor.device != target_device:
                    image_tensor = image_tensor.to(target_device)
                    data[key] = image_tensor
            else:
                image_tensor = torch.as_tensor(image, device=target_device)
                data[key] = image_tensor

            broadcast_mask = mask
            if broadcast_mask.shape != image_tensor.shape:
                while broadcast_mask.dim() < image_tensor.dim():
                    broadcast_mask = broadcast_mask.unsqueeze(0)
                broadcast_mask = broadcast_mask.expand_as(image_tensor)

            image_tensor.masked_fill_(broadcast_mask, self.fill_value)

        return data

class PatchifyLongi(transforms.MapTransform):
    def __init__(self, keys, patch_size, spatial_size):
        super().__init__(keys)
        self.patch_size = patch_size
        self.size = (1, spatial_size[0], spatial_size[1], spatial_size[2])

    def __call__(self, data):
        image = data.get("image0")
        patched_image = extract_patches(image,self.patch_size)
        data["image0"] = patched_image.squeeze()

        image = data.get("image1")
        patched_image = extract_patches(image,self.patch_size)
        data["image1"] = patched_image.squeeze()

        image = data.get("image2")
        patched_image = extract_patches(image,self.patch_size)
        data["image2"] = patched_image.squeeze()

        data.pop("mask0")
        data.pop("mask1")
        data.pop("mask2")

        data.pop("institution")
        return data

class ScaleIntensityCondition(transforms.MapTransform):
    """
    Scale intensity of the input image based on a condition.
    
    Args:
        keys (list): List of keys to apply the transformation to.
        scale (float): Scaling factor for the intensity.
        condition (callable): A function that takes the image and returns a boolean mask.
    """
    def __init__(self,
                 keys,
                 a_min=0.0,
                 a_max=1.0,
                 b_min=0.0,
                 b_max=1.0,
                 clip=True,
                 dtype=np.float32):
        super().__init__(keys)
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip, dtype)

    def __call__(self, data):
        if data['src'] == "NLST":
            return data
        data['image'] = self.scaler(data['image'])
        return data
        
class Load(transforms.MapTransform):
    """
    A custom MONAI transform to efficiently load NIfTI (.nii, .nii.gz)
    and NumPy (.npy) files from a dictionary of file paths.

    This loader is designed to be a faster alternative to MONAI's `LoadImaged`.
    It uses SimpleITK for NIfTI files (which is generally faster for I/O)
    and numpy for .npy files directly. NIfTI images are transposed to
    (X, Y, Z) axis order upon loading.

    Args:
        keys (list): A list of keys in the input dictionary that correspond
                     to the file paths to be loaded.
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        """
        Loads the image data for the specified keys in the data dictionary.

        Args:
            data (dict): A dictionary where keys specified in `self.keys`
                         contain file paths.

        Returns:
            dict: The dictionary with file paths replaced by the loaded image data
                  as NumPy arrays.
        """
        for key in self.keys:
            filepath = data[key]
            if filepath.endswith(('.nii', '.nii.gz')):
                # Use SimpleITK to load NIfTI files
                img_obj = sitk.ReadImage(filepath)
                array = sitk.GetArrayFromImage(img_obj)
                # Transpose from SimpleITK's (Z, Y, X) to (X, Y, Z)
                data[key] = array.transpose(2, 1, 0)
            elif filepath.endswith('.npy'):
                # Use numpy to load .npy files
                data[key] = np.load(filepath)
            else:
                raise ValueError(f"Unsupported file format for {filepath}. "
                                    "Only .nii, .nii.gz, and .npy are supported.")
            # Ensure channel dimension is added if missing
            if data[key].ndim == 3:
                data[key] = np.expand_dims(data[key], axis=0)
        return data

class Repeat(transforms.MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = data[key].repeat(3, 1, 1, 1) 

        return data