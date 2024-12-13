import argparse
import onnxruntime as ort
import numpy as np
import torch 
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from torchmetrics.detection.iou import IntersectionOverUnion 

from .metrics import calculate_nme_torch

def run_evaluation(model_path, eval_dataloader, calculate_lpips = True):
    # Load the ONNX model
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name

    # Initialize metrics
    psnr_f = PeakSignalNoiseRatio()
    ssim_f = StructuralSimilarityIndexMeasure()
    if calculate_lpips:
        lpips_f = LearnedPerceptualImagePatchSimilarity(normalize=True)
    
    # Accumulators for metrics
    psnr = 0
    ssim = 0
    lpips = 0
    num_samples = 0
    
    for idx, val_data in tqdm(enumerate(eval_dataloader)):
        # Convert torch tensors to numpy arrays
        target_np = val_data['gt'].numpy()
        input_np = val_data['lq'].numpy()
        if len(input_np.shape) == 3:
            input_np = val_data['lq'].unsqueeze(0).numpy()

        # Perform inference with ONNX Runtime
        output_np = ort_session.run(None, {input_name: input_np})[0]
        output_np = np.round(np.clip(output_np, 0, 255))

        # Convert numpy arrays back to torch tensors
        target = torch.from_numpy(target_np)
        output = torch.from_numpy(output_np)
        if len(output.shape) == 5:
            output = output.squeeze()

        # Calculate metrics
        psnr += psnr_f(output, target)
        ssim += ssim_f(output, target)

        if calculate_lpips:
            lpips += lpips_f(output/255, target/255)  # Normalized to [0, 1]
        
        num_samples += target.shape[0]  # Assuming target has shape (N, C, H, W)
    # Compute averages
    psnr /= num_samples
    ssim /= num_samples
    if calculate_lpips:
        lpips /= num_samples

    # Return results
    return {'PSNR': psnr.tolist(), 'SSIM': ssim.tolist(), 'LPIPS': lpips.tolist() if calculate_lpips else None}




def run_evaluation_detector(model_path, eval_dataloader, ref_model):
    # Load the ONNX model
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name

    # Initialize metrics
    iou_f = IntersectionOverUnion()
    
    for idx, val_data in tqdm(enumerate(eval_dataloader)):
        # Convert torch tensors to numpy arrays
        input_np = val_data['img'].numpy()
        input_np = val_data['lq'].numpy()
        if len(input_np.shape) == 3:
            input_np = val_data['lq'].unsqueeze(0).numpy()

    with torch.no_grad():
        for idx, data in tqdm(enumerate(eval_dataloader)):            
            metas = [{
            'img_shape': (256, 256, 3),
            'ori_shape': (256, 256, 3),
            'pad_shape': (256, 256, 3),
            'scale_factor': (1, 1, 1,1)
            }]

            output_np = ort_session.run(None, {input_name: input_np})[0]
            result = torch.Tensor(output_np)
            #TODO refactor this cursed tensor-list struct inherited from dataset/dataloader
            if len(result[0]) > 0:

                result = ref_model.bbox_head.get_bboxes(*result) #monkey patch just to get get_boxes() func without refactoring code
                result = {'bbox': [result[0][0]], 'labels' : [result[1][0]],'kps' : [result[2][0]]} #TODO : OMG the ground truth is [y1 x1 y2 x2] and model outputs were [x1 y1 x2 y2]......
            if len(result['bbox']) == 1 and len(result['bbox'][0]) > 0:  #TODO clean this block

                bbox_gt = torch.flatten(torch.stack(
                    [torch.stack(data['label']['person_0']["face_bbox_coordinates"]["2d_corners"][0]),
                    torch.stack(data['label']['person_0']["face_bbox_coordinates"]["2d_corners"][1])]
                )).unsqueeze(0)/4

                bbox_pred = result['bbox'][0][0,:-1].unsqueeze(0)
                nme += calculate_nme_torch(data['label']['person_0'], result)


                sample_iou = iou_f( 
                            [{"boxes" : bbox_pred ,
                            "labels": result['labels'][0][0].unsqueeze(0)}], # The last element is the confidence, thus -1,
                            [{"boxes": bbox_gt,
                            "labels" : torch.tensor([0])}])['iou']
                

                iou += sample_iou
                count_detections += 1
            else:
                missed_detections += 1
        if count_detections != 0:
            nme /= count_detections
            iou /= count_detections

        return {'NME' : nme.tolist(), 'IoU' : iou.tolist(), 'missed_detections' : missed_detections, 'n_detections' : count_detections}