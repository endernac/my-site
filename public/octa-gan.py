import onnxruntime as ort  # Add ONNX Runtime dependency
import torch
import numpy as np
import torchio as tio
from torch.nn import functional as F
import pandas as pd


def normalize(volume: np.ndarray) -> np.ndarray:
    """
    Rescales intensities to 0.5 and 99.5 percentiles, then
    normalizes them between -1 and 1.
    """

    # Check that volume is a 3D numpy array
    if len(volume.shape) != 3:
        raise Exception('Expected input volume to be a 3D numpy array, ' \
                        f'but got a {len(volume.shape)}D array instead!')

    # Define rescale and normalization operation
    normalize = tio.RescaleIntensity(percentiles=(0.5, 99.5), out_min_max=(-1, 1))

    # Create a TorchIO subject from the 3D volume
    subj = to_tio_subject(volume)

    # Apply rescaling and normalization
    processed = normalize(subj)
    return processed['image'].numpy()[0]

def crop_volume(volume: np.ndarray, tol: float = 0.0):
    """
    Crops out zero-intensity axial slices near the upper and 
    lower bounds of the y-axis (height).
    """

    # Check that volume is a 3D numpy array
    if len(volume.shape) != 3:
        raise Exception('Expected input volume to be a 3D numpy array, ' \
                        f'but got a {len(volume.shape)}D array instead!')

    # Mask of non-black pixels (assuming volume has a single channel).
    mask = volume > tol

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = volume[:,y0:y1,:]

    return cropped

def resize(
    volume: np.ndarray,
    target_dim: tuple[int, int, int]
    # device: optional[torch.device] = None
) -> np.ndarray:
    """
    Resize volume with trilinear interpolation to the target dimensions
    using PyTorch's implementation of torch.nn.functional.interpolate.
    """

    # Check that volume is a 3D numpy array
    if len(volume.shape) != 3:
        raise Exception('Expected input volume to be a 3D numpy array, ' \
                        f'but got a {len(volume.shape)}D array instead!')

    # Check that target dimensions is a 3D tuple
    if not isinstance(target_dim, tuple) or len(target_dim) != 3:
        raise ValueError('Expected target volume dimensions to be a 3D tuple, ' \
                            f'but got a {len(target_dim)}D tuple instead!')

    # Nothing to do if volume size already matches the target dimensions
    if volume.shape == target_dim:
        return volume

    # Cast numpy array to a PyTorch tensor
    vol_t = torch.from_numpy(volume).view(1, 1, *volume.shape)

    # if device is not None:
    #     vol_t = vol_t.to(device)

    # Interpolate the volume to the target dimensions
    with torch.inference_mode():
        resized = F.interpolate(vol_t, size=target_dim, mode='trilinear', align_corners=False)
    
    # if device is not None:
    #     resized = resized.cpu()

    return resized.numpy()[0,0]

def to_tio_subject(volume: np.ndarray, phys_dim: np.ndarray = np.array([3.0, 2.0, 3.0])) -> tio.Subject:
    """
    Creates a TorchIO subject of an OCTA volume from a 3D numpy array.
    """

    # Check that volume is a 3D numpy array
    if len(volume.shape) != 3:
        raise Exception('Expected input volume to be a 3D numpy array, ' \
                        f'but got a {len(volume.shape)}D array instead!')

    # Create a TorchIO subject from a 3D numpy array
    aff = np.diag([
        phys_dim[0] / volume.shape[0],
        phys_dim[1] / volume.shape[1],
        phys_dim[2] / volume.shape[2], 1
    ])
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=torch.from_numpy(volume[None,].copy()), affine=aff)
    )
    
    return subject

# Pre-processing function
def preprocess(input_array: np.ndarray) -> torch.Tensor:
    """
    Pre-process the input tensor to prepare it for the model.
    Args:
        input_tensor (np.ndarray): The input tensor to be processed.
    Returns:
        torch.Tensor: The pre-processed input tensor with shape (1, 1, 128, 128, 128).
    """
    # Create an instance of OCTADataset

    # Normalize the input volume
    vol = np.flip(input_array.copy().astype(np.float32), 1)
    vol = normalize(vol)
    vol = crop_volume(vol, tol=0.75)
    vol = resize(vol, target_dim=(128, 128, 128))
    vol = normalize(vol)

    # Convert to PyTorch tensor and reshape
    tensor = torch.from_numpy(vol).view(1, 1, 128, 128, 128)
    return tensor

# Inference function
def run_inference(preprocessed_tensor: torch.Tensor, model_path: str) -> torch.Tensor:
    """
    Run inference on the input volume using ONNX Runtime.
    Args:
        preprocessed_tensor (torch.Tensor): The pre-processed input tensor.
        model_path (str): Path to the ONNX model file.
    Returns:
        torch.Tensor: The output tensor from the ONNX model.
    """
    # Convert PyTorch tensor to NumPy array
    input_array = preprocessed_tensor.cpu().numpy()

    # Initialize ONNX Runtime session
    session = ort.InferenceSession(model_path)

    # Get input and output names for the ONNX model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    output = session.run([output_name], {input_name: input_array})[0]

    # Convert the output back to a PyTorch tensor
    return torch.tensor(output)

# Post-processing function
def postprocess(output: torch.Tensor) -> str:
    """
    Post-process the output tensor to get the final result.
    Args:
        output (torch.Tensor): The output tensor from the model.
    Returns:
        str: The final result after post-processing. Either "Good" or "Suboptimal".
    """
    score = output.item()  # Extract the scalar value from the tensor
    return score, "Good" if score >= 3.8675 else "Suboptimal"

if __name__ == "__main__":
    # Example usage
    input_path = "data/size_1024/P1318393_Angiography 3x3 mm_7-22-2016_9-29-21_OD_sn0085_FlowCube_z.img"  # Replace with the actual path to the input volume
    model_path = "weights/OCTA-GAN/D_A_iter280000.onnx"  # Replace with the actual path to the ONNX model

    # Load the input volume (assume .img file with vol_dim [245, 1024, 245])
    vol_dim = (245, 1024, 245)
    input_array = np.fromfile(input_path, dtype="uint8").reshape(vol_dim).astype(np.float32)

    # Preprocess the input volume
    preprocessed_tensor = preprocess(input_array)

    # Run inference using ONNX Runtime
    output = run_inference(preprocessed_tensor, model_path)

    # Post-process the output tensor
    score, label = postprocess(output)
    