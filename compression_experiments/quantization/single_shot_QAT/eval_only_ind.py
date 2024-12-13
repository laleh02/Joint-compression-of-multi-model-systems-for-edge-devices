import argparse
import torch
import os
import time 
import gc
import logging 
import cv2
import json
from aimet_torch.arch_checker.arch_checker import ArchChecker
from aimet_torch.model_preparer import prepare_model
from aimet_common.defs import QuantScheme
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.quant_analyzer import QuantAnalyzer, CallbackFunc
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.utils import AimetLogger


### PTQ imports

from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.cross_layer_equalization import equalize_model
import aimet_torch.quantsim as quantsim

from archs.PMRID import PMRID
from archs.PMRIDd2 import PMRIDd2
from archs.e2e_yunet import PipelinedYuNet 
from utils.callbacks_rescaled import forward_pass_callback, eval_callback, calibration_callback, calibration_callback_detector
from utils.train_rescaled import train
from utils.eval_rescaled import eval_restoration, eval_detection
from utils.config_handler import parse_yaml
from utils.data import create_dataloader, Unlabeled_Dataset, Restoration_Dataset, Detection_Dataset
from utils.onnx_runtime_evaluation_rescaled import run_evaluation
from utils.builder import init_detector
from utils.image_processing import resize_img



def parse_args():
    """argument parser"""
    parser = argparse.ArgumentParser(
        description="QAT pipeline for restoration and detection models"
    )
    parser.add_argument(
        "--aimet_config",
        default=None,
        help="Config for QAT",
    )
    parser.add_argument(
        "--experiment_config",
        default=None,
        help="Config for quantization experiments",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--dataroot_lq",
        type=str,
        default=None,
        help="A folder containing the degraded training data for restoration models.",
    )
    parser.add_argument(
        "--dataroot_gt",
        type=str,
        default=None,
        help="A folder containing the ground truth training data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="training seed",
    )
    parser.add_argument(
        "--adaround_output_path",
        type=str,
        default="dummy_ptq_run/",
    )
    parser.add_argument(
        "--gt_suffix",
        type=str,
        default=None,
        help="The suffix identifying the ground truth data.",
    )
    parser.add_argument(
        "--lq_suffix",
        type=str,
        default=None,
        help="The suffix identifying the degraded data.",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=[1,3,256,256],
        help="The suffix identifying the degraded data.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="dummy_ptq_run",
        help="Folder for AIMET quantization and eval results.",
    )
    args = parser.parse_args()
    for arg in vars(args):
        print("{:30s} : {}".format(arg, getattr(args, arg)))
    experiment_config = parse_yaml(args.experiment_config)
    args.input_shape = tuple(args.input_shape)

    return args, experiment_config


class Patched_IndependentYuNet(torch.nn.Module):

    def __init__(self,
                 denoiser,
                 lle_enhancer,
                 detector,
                ):
        super(Patched_IndependentYuNet, self).__init__()
        self.enhancers = torch.nn.Sequential()
        self.denoiser = denoiser
        self.lle_enhancer = lle_enhancer
        self.detector = detector
    def set_enhancers(self, enhancer_list) : 
        assert len(enhancer_list) != 0
        self.enhancers = torch.nn.Sequential(*enhancer_list)
    def forward(self,
                      img,
                      #img_metas,
                      #gt_bboxes,
                      #gt_labels,
                      #gt_keypointss=None,
                      #gt_bboxes_ignore=None, 
                      rescale=False):
        if isinstance(img,list):
            img = img[0]
        img = self.lle_enhancer(self.denoiser(img))
        cls_preds, bbox_preds, obj_preds, kps_preds = self.detector(img)

        return cls_preds, bbox_preds, obj_preds, kps_preds#, bbox_list


    def renormalize_input(self, input):
        # Enhancers work on images in [0,1] while detector work on images in [0, 255]
        return torch.round(torch.clip(input,0,255))
    def output_enhancers_result(self, input):
        return self.renormalize_input(self.enhancers(input))

def main():
    results_dict = {}
    args, experiment_config = parse_args()
    if experiment_config['model_name'] == 'PMRID':
        model = PMRID()
    elif experiment_config['model_name'] == 'PMRIDd2':
        model = PMRIDd2()
    else:
        assert experiment_config['model_name'] == 'YuNET' \
        or experiment_config['model_name'] == 'PipelinedYuNET' \
        or experiment_config['model_name'] == 'indPipelinedYuNET'

    AimetLogger.set_level_for_all_areas(logging.CRITICAL)
    device = experiment_config['device']
    results_dir = os.path.join("experiment_results",experiment_config['experiment_name'])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Folder '{results_dir}' created successfully.")
    else:
        print(f"Experiment '{experiment_config['experiment_name']}' already exists.")
    import sys

    sys.setrecursionlimit(5000)

    if 'pretrain_network_g' in experiment_config['path']:

        if experiment_config['model_name'] == 'YuNET' \
            or experiment_config['model_name'] == 'PipelinedYuNET' \
            or experiment_config['model_name'] == 'indPipelinedYuNET':
            ref_model = init_detector(config=experiment_config['yunet_config'],
                              checkpoint = experiment_config['path']['pretrain_network_g'])
            print(f"Loaded model weights from {experiment_config['path']['pretrain_network_g']}")
            
            #Yunet requires metadata of the image for training
            img = torch.rand(args.input_shape)*255
            numpy_image = img[0].permute(1, 2, 0).numpy()
            # Convert the numpy array to a cv2 image
            cv2_image = cv2.cvtColor(numpy_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite("test2.png", cv2_image)

            det_img, det_scale = resize_img(cv2_image, 'AUTO')
            image_metas = [{
            'img_shape': det_img.shape,
            'ori_shape': cv2_image.shape,
            'pad_shape': det_img.shape,
            }]
            dummy_input = (img , image_metas)
        
            if experiment_config['model_name'] == 'indPipelinedYuNET':

                detector = quantsim.load_checkpoint(experiment_config['enhancers']['detector']['pretrain_network_g'])

                lle_enhancer = quantsim.load_checkpoint(experiment_config['enhancers']['LLE']['pretrain_network_g'])
                denoising_enhancer = quantsim.load_checkpoint(experiment_config['enhancers']['denoise']['pretrain_network_g'])

                model = Patched_IndependentYuNet(detector=detector.model.to(device),denoiser=denoising_enhancer.model.to(device),lle_enhancer=lle_enhancer.model.to(device))
                     
            elif experiment_config['model_name'] == 'YuNET':
                state_dict = torch.load(experiment_config['model']['pretrain_network_g'])['state_dict']

                backbone_state_dict = {".".join(k.split(".")[1:]) : v for k,v in state_dict.items() if 'backbone' in k}
                neck_state_dict = {".".join(k.split(".")[1:]) : v for k,v in state_dict.items() if 'neck' in k}
                bbox_head_state_dict = {".".join(k.split(".")[1:]) : v for k,v in state_dict.items() if 'bbox_head' in k}  

                model.backbone.load_state_dict(backbone_state_dict)
                model.neck.load_state_dict(neck_state_dict)
                model.bbox_head.load_state_dict(bbox_head_state_dict)
            elif experiment_config['model_name'] == 'PipelinedYuNET':
                model = torch.load(experiment_config['detector']['pretrain_network_g'])
        else: 
            model.load_state_dict(torch.load(experiment_config['path']['pretrain_network_g'])['params'])
            print(f"Loaded model weights from {experiment_config['path']['pretrain_network_g']}")
            dummy_input = torch.rand(args.input_shape)*255

        if experiment_config['original_model_onnx_export']:
            onnx_model_path = os.path.join(results_dir,"original_model.onnx")
            torch.onnx.export(
                model,             # model being run
                dummy_input,       # model input (or a tuple for multiple inputs)
                onnx_model_path,   # where to save the model (can be a file or file-like object)
                export_params=True,         # store the trained parameter weights inside the model file
                #opset_version=12,           # the ONNX version to export the model to
                #do_constant_folding=True,   # whether to execute constant folding for optimization
                input_names=['input'],      # the model's input names
                output_names=['output'],    # the model's output names
                #dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                #            'output': {0: 'batch_size'}}
                )

            print(f"Model exported to {onnx_model_path}")

    model.to(device)
    if experiment_config['task'] == "Restoration":
        dummy_input = dummy_input.to(device=device)
    else:
        dummy_input = [torch.rand(args.input_shape).to(device)]
    ArchChecker.check_model_arch(model, dummy_input=dummy_input)
    print("start preparing model")
    start = time.time()

    if experiment_config['task'] == 'Restoration':
        eval_dataset = Restoration_Dataset(config = experiment_config['datasets']['eval'])
        eval_dataloader = create_dataloader(eval_dataset,config = {'datasets':
                                                                {'dataloader': {'batch_size' : 1}}})
        eval_res = eval_restoration(model, eval_dataloader,device, calculate_lpips=True)

    else:
        eval_dataset = Detection_Dataset(config = experiment_config['datasets']['eval'])
        eval_dataloader = create_dataloader(eval_dataset,config = {'datasets':
                                                                {'dataloader': {'batch_size' : 1}}})    
        eval_res = eval_detection(model, eval_dataloader,device,ref_model)


    print(f"EVAL RESULT - Independent model eval results : {eval_res}")
    results_dict['after pipelining'] = eval_res
    end = time.time()
    print(f"Evaluation on {len(eval_dataloader) * eval_dataloader.batch_size} time : {end - start}")

    if experiment_config['task'] == 'Restoration':
        original_model_evaluation = run_evaluation(os.path.join(results_dir,"original_model.onnx"), eval_dataloader)
        print(f'Original Model Evaluation Metric: {original_model_evaluation}')
        results_dict['original_model_onnx'] = original_model_evaluation
        # Evaluate the quantized model
        quantized_model_evaluation = run_evaluation(os.path.join(results_dir,"quant_" + experiment_config['experiment_name'] + "_embedded.onnx"), eval_dataloader)
        print(f'Quantized Model Evaluation Metric: {quantized_model_evaluation}')
        results_dict['quantized_model_onnx'] = quantized_model_evaluation

    with open(os.path.join(results_dir,"eval_results.json"), 'w') as json_file:
        json.dump(results_dict, json_file, indent = 4)
if __name__ == '__main__':
    main()