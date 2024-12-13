import argparse
import torch
import os
import time 
import gc
import logging 
import cv2
import json

### PTQ imports
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


from archs.e2e_yunet import End2EndYuNet
from utils.eval import eval_restoration
from utils.config_handler import parse_yaml
from utils.callbacks import forward_pass_callback, eval_callback, calibration_callback

from utils.data import create_dataloader, Restoration_Dataset, Detection_Dataset
from utils.onnx_runtime_evaluation import run_evaluation
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


def main():
    results_dict = {}
    args, experiment_config = parse_args()
    # if experiment_config['model_name'] == 'PMRID':
    #     model = PMRID()
    # elif experiment_config['model_name'] == 'PMRIDd':
    #     model = PMRIDd()

    if experiment_config['model_name'] == 'PMRIDd2':
        model = PMRIDd2()
    else:
        assert experiment_config['model_name'] == 'YuNET'

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

        if experiment_config['model_name'] == 'PipelinedYuNET':
            model = init_detector(config=experiment_config['yunet_config'],
                              checkpoint = experiment_config['path']['pretrain_network_g'])
            print(f"Loaded model weights from {experiment_config['path']['pretrain_network_g']}")
            
            #Yunet requires metadata of the image for training
            img = torch.rand(args.input_shape)
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

        else: 
            # model = torch.load(experiment_config['path']['pretrain_network_g'])
            # print(model)
            model.load_state_dict(torch.load(experiment_config['path']['pretrain_network_g'])['params'])
            #print(f"Loaded model weights from {experiment_config['path']['pretrain_network_g']}")
            dummy_input = torch.rand(args.input_shape)   

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
    ArchChecker.check_model_arch(model, dummy_input=dummy_input.to(device))
    print("start preparing model")
    prepared_model = prepare_model(model)

    ModelValidator.validate_model(prepared_model, model_input=dummy_input.to(device))
    
    sim = QuantizationSimModel(model=prepared_model.to(device=experiment_config['device']),
                            quant_scheme=QuantScheme.post_training_tf,
                            config_file="configs/debug_config.json",
                            dummy_input=dummy_input.to(device),
                            default_output_bw=8,
                            default_param_bw=8)

    eval_dataset = Restoration_Dataset(config = experiment_config['datasets']['eval'])
    eval_dataloader = create_dataloader(eval_dataset,config = {'datasets':
                                                            {'dataloader': {'batch_size' : 1}}})

    sim.compute_encodings(forward_pass_callback=calibration_callback,
                            forward_pass_callback_args=(eval_dataloader, device, 10,'Restoration')) 

    start = time.time()
    # postptq_analyzer = QuantAnalyzer(sim.model, 
    #                         dummy_input=dummy_input.to(device),
    #                         forward_pass_callback=CallbackFunc(forward_pass_callback,(eval_dataloader,device)),
    #                         eval_callback=CallbackFunc(eval_callback,(eval_dataloader, device, 'Restoration')))

    # postptq_analyzer.enable_per_layer_mse_loss(unlabeled_dataset_iterable=analyzer_data_loader, num_batches=4)
    # postptq_analyzer.analyze(quant_scheme=QuantScheme.post_training_tf_enhanced,
    #                     default_param_bw=8,
    #                     default_output_bw=8,
    #                     config_file="configs/debug_config.json",
    #                     results_dir= os.path.join(results_dir, "post_ptq_results"))
    # end = time.time()
    # print(f"Post-PTQ Analyzer time: {end - start}")
    
    
    if experiment_config['task'] == "Restoration":
        dummy_input = dummy_input.to(device=device)
    else:
        dummy_input = [torch.rand(args.input_shape).to(device)]

    eval_res = eval_restoration(sim.model, eval_dataloader,device, calculate_lpips=True, save_img = True)

    for i, data in enumerate(eval_dataloader):
        print(data.keys())
        print(data['lq_path'])
        print(data['gt_path'])
        break

    print(f"EVAL RESULT - Eval results before PTQ : {eval_res}")
    #results_dict['before PTQ'] = eval_res


if __name__ == '__main__':
    main()