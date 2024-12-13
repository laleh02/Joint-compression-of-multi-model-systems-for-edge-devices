import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from typing import Any
from tqdm import tqdm
from .eval import eval_restoration
def forward_pass_callback(model: torch.nn.Module, params) -> None:
    """
    NOTE: This is intended to be the user-defined model calibration function.
    AIMET requires the above signature. So if the user's calibration function does not
    match this signature, please create a simple wrapper around this callback function.

    A callback function for model calibration that simply runs forward passes on the model to
    compute encoding (delta/offset). This callback function should use representative data and should
    be subset of entire train/validation dataset (~1000 images/samples).

    :param model: PyTorch model.
    :param _: Argument(s) of this callback function. Up to the user to determine the type of this parameter.
    E.g. could be simply an integer representing the number of data samples to use. Or could be a tuple of
    parameters or an object representing something more complex.
    """
    # User action required
    # User should create data loader/iterable using representative dataset and simply run
    # forward passes on the model.
    dataloader, device = params
    for i in range(100):
        representative_sample = torch.rand(1,3,256,256).to(device=device)
        forward_pass = model(representative_sample)

def eval_callback(model: torch.nn.Module, params) -> float:
    """
    NOTE: This is intended to be the user-defined model evaluation function.
    AIMET requires the above signature. So if the user's calibration function does not
    match this signature, please create a simple wrapper around this callback function.

    A callback function for model evaluation that determines model performance. This callback function is
    expected to return scalar value representing the model performance evaluated against entire
    test/evaluation dataset.

    :param model: PyTorch model.
    :param _: Argument(s) of this callback function. Up to the user to determine the type of this parameter.
    E.g. could be simply an integer representing the number of data samples to use. Or could be a tuple of
    parameters or an object representing something more complex.
    :return: Scalar value representing the model performance.
    """
    # User action required
    # User should create data loader/iterable using entire test/evaluation dataset, perform forward passes on
    # the model and return single scalar value representing the model performance.
    eval_dataloader, device, task = params
    if task == "Restoration":
        results = eval_restoration_callback(model, eval_dataloader, device)
    else:
        results = eval_detection_callback(model, eval_dataloader, device)
    return results

def eval_detection_callback(model, eval_dataloader, device):
    return -1 # not implemented

def eval_restoration_callback(model, eval_dataloader, device, calculate_lpips = False, save_img = False, results_dir = 'debug'):
    results = eval_restoration(model, eval_dataloader, device, calculate_lpips, save_img, results_dir)
    return results['PSNR']

def calibration_callback(model: torch.nn.Module, params, n_samples = None) -> float:
    dataloader, device, n_samples, task = params
    model.eval()
    if n_samples is None:
        n_samples = len(dataloader)
    assert n_samples is not None

    sample_count = 0
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for idx, input in tqdm(enumerate(dataloader)):
            if task == 'Restoration':
                model(input['lq'].to(device)/255)   
            else:
                model(input['img'].to(device)/255)   

            sample_count += batch_size
            if sample_count > n_samples:
                break

def calibration_callback_detector(model: torch.nn.Module, params, n_samples = None) -> float:
    dataloader, device, n_samples, task = params
    model.eval()
    if n_samples is None:
        n_samples = len(dataloader)
    assert n_samples is not None

    sample_count = 0
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for idx, input in tqdm(enumerate(dataloader)):
            if task == 'Restoration':
                model(input['lq'].to(device))   
            else:
                model(input['img'].to(device))   

            sample_count += batch_size
            if sample_count > n_samples:
                break