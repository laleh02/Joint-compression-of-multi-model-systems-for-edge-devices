
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
from torchmetrics.detection.iou import IntersectionOverUnion 
import torch
from torchvision.utils import make_grid
from .metrics import calculate_nme, calculate_nme_torch
import os
import cv2
import math
import numpy as np
def eval_restoration(model, eval_dataloader, device, calculate_lpips = True, save_img = False, results_dir = 'debug'):
    psnr_f = PeakSignalNoiseRatio().to(device)
    ssim_f = StructuralSimilarityIndexMeasure().to(device)
    if calculate_lpips:
        lpips_f = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    model.eval()
    psnr = 0
    ssim = 0
    lpips = 0
    with torch.no_grad():
        for idx, val_data in tqdm(enumerate(eval_dataloader)):
            #print(model(val_data['lq'].to(device)))
            target = val_data['gt'].to(device)/255
            img_name = val_data['lq_path'][0].split("/")[-1].split(".")[0]
            output = torch.clip(model(val_data['lq'].to(device)/255),min=0,max=1)
            psnr += psnr_f(output, target)
            ssim += ssim_f(output, target)

            if calculate_lpips:
        
                lpips += lpips_f(output, target) #In range [0,1] to be normalized by metric to [-1,1]
        
            if save_img:
                output_img = tensor2img(output[0], rgb2bgr=True, min_max=(0, 1))
                gt_img = tensor2img(target[0], rgb2bgr=True, min_max=(0, 1))
                lq_img = tensor2img(val_data['lq'].to(device)/255, rgb2bgr=True, min_max=(0, 1))
                imwrite(output_img, os.path.join(results_dir, f'{img_name}_cleaned.png'))
                imwrite(gt_img, os.path.join(results_dir, f'{img_name}_gt.png'))
                imwrite(lq_img, os.path.join(results_dir, f'{img_name}_lq.png'))

                print(f"Saving image {os.path.join(results_dir, f'{img_name}_cleaned.png')}")
        
        psnr /= (len(eval_dataloader.dataset))
        ssim /= (len(eval_dataloader.dataset))
        lpips /= (len(eval_dataloader.dataset))
            
        if not calculate_lpips:

            return {'PSNR' : psnr.tolist(), 'SSIM' : ssim.tolist(), 'LPIPS' : 0}
        else:
            return {'PSNR' : psnr.tolist(), 'SSIM' : ssim.tolist(), 'LPIPS' : lpips.tolist()}
    
def eval_detection(model, eval_dataloader, device,ref_model, save_img = False, results_dir = None):
    iou_f = IntersectionOverUnion().to(device)
    model.eval()
    nme = torch.zeros(1).to(device)
    missed_detections = 0
    iou = torch.zeros(1).to(device)
    count_detections = 0
    with torch.no_grad():
        for idx, data in tqdm(enumerate(eval_dataloader)): 
            img_name = data['img_path'][0].split("/")[-1].split(".")[0]           
            metas = [{
            'img_shape': (256, 256, 3),
            'ori_shape': (256, 256, 3),
            'pad_shape': (256, 256, 3),
            'scale_factor': (1, 1, 1,1)
            }]

            result = model(data['img'].to('cuda'))
            #TODO refactor this cursed tensor-list struct inherited from dataset/dataloader
            if len(result[0]) > 0:

                result = ref_model.bbox_head.get_bboxes(*result) #monkey patch just to get get_boxes() func without refactoring code
                result = {'bbox': [result[0][0]], 'labels' : [result[1][0]],'kps' : [result[2][0]]} #TODO : OMG the ground truth is [y1 x1 y2 x2] and model outputs were [x1 y1 x2 y2], corrected in the lines below......
            if len(result['bbox']) == 1 and len(result['bbox'][0]) > 0:  #TODO clean this block

                bbox_gt = torch.flatten(torch.stack(
                    [torch.stack(data['label']['person_0']["face_bbox_coordinates"]["2d_corners"][0]),
                    torch.stack(data['label']['person_0']["face_bbox_coordinates"]["2d_corners"][1])]
                )).unsqueeze(0).to(device)/4

                bbox_pred = result['bbox'][0][0,:-1].unsqueeze(0)
                nme += calculate_nme_torch(data['label']['person_0'], result, device='cuda:0')


                sample_iou = iou_f( 
                            [{"boxes" : bbox_pred ,
                            "labels": result['labels'][0][0].unsqueeze(0)}], # The last element is the confidence, thus -1,
                            [{"boxes": bbox_gt,
                            "labels" : torch.tensor([0],device=device)}])['iou']
                
                if save_img:
                    
                    # enhanced_image = ref_model.output_enhancers_result(data['img'].to('cuda'))
                    # enhanced_frame = enhanced_image.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
                    # enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
                    # imwrite(enhanced_frame, os.path.join(results_dir, f'{img_name}_enhanced.png'))

                    # Extract rectangle coordinates
                    frame = data['img'].squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    try:
                        x1, y1, x2, y2 = bbox_pred[0].cpu().numpy().astype(np.uint8)
                        kps = result['kps'][0][0].cpu().numpy().astype(np.uint8)
                        # Draw the rectangle
                        cv2.rectangle(frame, (y1, x1), (y2, x2), (0, 255, 0), 2)

                        # Loop through the remaining values in pairs and draw the points
                        for i in range(0, len(kps), 2):
                            x, y = kps[i], kps[i + 1]
                            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                    except:
                        raise
                    imwrite(frame, os.path.join(results_dir, f'{img_name}_detected.png'))
                iou += sample_iou
                count_detections += 1
            else:
                missed_detections += 1
        if count_detections != 0:
            nme /= count_detections
            iou /= count_detections

        return {'NME' : nme.tolist(), 'IoU' : iou.tolist(), 'missed_detections' : missed_detections, 'n_detections' : count_detections}


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            elif img_np.shape[2] == 3:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result