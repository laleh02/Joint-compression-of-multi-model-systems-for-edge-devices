import argparse
import torch
import os
import time 
import gc
import logging 
import cv2
import json

import pdb

### Pruning imports
import torch_pruning as tp


from archs.PMRID import PMRID
from utils.callbacks import forward_pass_callback, eval_callback, calibration_callback
from utils.train import train
from utils.eval import eval_restoration, eval_detection
from utils.config_handler import parse_yaml
from utils.data import create_dataloader, Unlabeled_Dataset, Restoration_Dataset, Detection_Dataset
from utils.onnx_runtime_evaluation import run_evaluation
from utils.builder import init_detector
from utils.image_processing import resize_img
from utils.losses import PSNRLoss
from archs.pipelined_yunet import PipelinedYuNet
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
    if experiment_config['model_name'] == 'PMRID':
        model = PMRID()
    elif experiment_config['model_name'] == 'PMRIDd2':
        model = PMRIDd2()
    elif experiment_config['model_name'] == 'PMRIDu3':
        model = PMRIDu3()
    else:
        assert experiment_config['model_name'] == 'YuNET' or experiment_config['model_name'] == 'PipelinedYuNET'
        task = "detection"

    device = experiment_config['device']
    device_id = experiment_config['gpu_id'] #torch.cuda.current_device()
    torch.cuda.set_device(device_id)
    results_dir = os.path.join("experiment_results","pruning_" + experiment_config['experiment_name'])

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Folder '{results_dir}' created successfully.")
    else:
        print(f"Experiment '{results_dir}' already exists.")
    
    import sys

    sys.setrecursionlimit(5000)
    if 'pretrain_network_g' in experiment_config['path']:

        if experiment_config['model_name'] == 'YuNET' or experiment_config['model_name'] == 'PipelinedYuNET':
            model = init_detector(config=experiment_config['yunet_config'],
                              checkpoint = experiment_config['path']['pretrain_network_g'])
            print(f"Loaded model weights from {experiment_config['path']['pretrain_network_g']}")
            
            #Yunet requires metadata of the image for training
            img = torch.rand(args.input_shape).squeeze()
            numpy_image = img.permute(1, 2, 0).numpy()
            # Convert the numpy array to a cv2 image
            cv2_image = cv2.cvtColor(numpy_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite("test2.png", cv2_image)

            det_img, det_scale = resize_img(cv2_image, 'AUTO')
            image_metas = [{
            'img_shape': det_img.shape,
            'ori_shape': cv2_image.shape,
            'pad_shape': det_img.shape,
            'scale_factor': tuple([int(det_scale) for _ in range(3)])
            }]
            dummy_input = (img , image_metas)
        
            if experiment_config['model_name'] == 'PipelinedYuNET':
                lle_enhancer = PMRIDd2()
                denoising_enhancer = PMRIDd2()
                lle_enhancer.load_state_dict(torch.load(experiment_config['enhancers']['LLE']['pretrain_network_g'])['params'])
                denoising_enhancer.load_state_dict(torch.load(experiment_config['enhancers']['denoise']['pretrain_network_g'])['params'])
                model.set_enhancers([denoising_enhancer.to(device), lle_enhancer.to(device)])

        else: 
            model.load_state_dict(torch.load(experiment_config['path']['pretrain_network_g'])['params'])
            print(f"Loaded model weights from {experiment_config['path']['pretrain_network_g']}")
            dummy_input = torch.rand(args.input_shape)   
        if experiment_config['original_model_onnx_export']:
            onnx_model_path = os.path.join(results_dir,"original_model.onnx")
            torch.onnx.export(
                model,             # model being run
                dummy_input,       # model input (or a tuple for multiple inputs)
                onnx_model_path,   # where to save the model (can be a file or file-like object)
                export_params=True,         # store the trained parameter weights inside the model file
                opset_version=12,           # the ONNX version to export the model to
                #do_constant_folding=True,   # whether to execute constant folding for optimization
                input_names=['input'],      # the model's input names
                output_names=['output'],    # the model's output names
                #dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                #            'output': {0: 'batch_size'}}
                )

            print(f"Model exported to {onnx_model_path}")


    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters in the model: {num_params}')
    model.to(device)

    if experiment_config['task'] == "Restoration":
        dummy_input = dummy_input.to(device=device)
        ignored_layers = [model.conv0, model.out1]
    else:
        dummy_input = [torch.rand(args.input_shape).to(device)]
        if experiment_config['model_name'] == 'PipelinedYuNET':
            ignored_layers = [model.bbox_head, model.neck, model.backbone, model.enhancers.denoiser.conv0, model.enhancers.denoiser.out1,
                            model.enhancers.light_enhancer.conv0, model.enhancers.light_enhancer.out1]
        else:
            ignored_layers = [model.bbox_head]
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
        eval_res = eval_detection(model, eval_dataloader,device, model)


    print(f"EVAL RESULT - Eval results before channel prunning : {eval_res}")
    results_dict['before_pruning'] = eval_res
    end = time.time()
    print(f"Evaluation on {len(eval_dataloader) * eval_dataloader.batch_size} time : {end - start}")

    example_inputs = torch.randn(1, 3, 256, 256).to(device)

    importance =  tp.importance.MagnitudeImportance(p=1, group_reduction='mean') 

    pruning_ratio = experiment_config.get('pruning_ratio', 0.9)
    iterative_steps = 45

    print(f"Pruning ratio : {pruning_ratio}")
    print(f"Total Iterative steps : {iterative_steps}")

    pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=importance,
            iterative_steps=iterative_steps,
            pruning_ratio=pruning_ratio,
            global_pruning=True,
            #round_to=round_to,
            #unwrapped_parameters=unwrapped_parameters,
            ignored_layers=ignored_layers,
            #channel_groups=channel_groups,
        )
    loss_f = PSNRLoss()
    # 3. Prune & finetune the model
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, dummy_input)
    if isinstance(importance, tp.importance.GroupTaylorImportance):
        # Taylor expansion requires gradients for importance estimation
        loss_value = loss_f(model(dummy_input), dummy_input) # A dummy loss, please replace this line with your loss function and data!
        loss_value.backward()
    
    for i in range(iterative_steps):
        
        
        pruner.step()

        macs, nparams = tp.utils.count_ops_and_params(model, dummy_input)

        print(f"Original model : MACS {base_macs}, Params: {base_nparams}")
        print(f"Pruned model, step {i} : MACS {macs}, Params: {nparams}")


        if experiment_config['task'] == 'Restoration':
            eval_res = eval_restoration(model,  eval_dataloader,device, calculate_lpips=True)

        else:   
            eval_res = eval_detection(model, eval_dataloader,device, model) 
            
        print(f"EVAL RESULT - Eval results after channel prunning step {i}: {eval_res}")
        results_dict[f'after_pruning_step_{i}'] = eval_res


        ## finetuning

        train(model, experiment_config, results_dir, device)


        if experiment_config['task'] == 'Restoration':
            eval_res = eval_restoration(model,  eval_dataloader,device, calculate_lpips=True)
            torch.save(model, os.path.join(results_dir,"pruning_" + experiment_config['experiment_name'] + "_step_" + str(i) + ".pth"))

        else:   
            eval_res = eval_detection(model, eval_dataloader,device, model) 
            
        print(f"EVAL RESULT - Eval results after finetuning step {i}: {eval_res}")
        results_dict[f'after_finetuning_step_{i}'] = eval_res

    if experiment_config['task'] == 'Restoration':
        eval_res = eval_restoration(model,  eval_dataloader,device, calculate_lpips=True)

    else:   
        eval_res = eval_detection(model, eval_dataloader,device, model)  

           
    print(f"EVAL RESULT - Eval results after finetuning : {eval_res}")
    results_dict['after_finetuning'] = eval_res





    #Saving models both as .pth and .onnx
    torch.save(model, os.path.join(results_dir,  "pruned_model.pth"))
    if experiment_config['task'] == 'Restoration':
        torch.onnx.export(model, dummy_input,  os.path.join(results_dir, "pruned_model.onnx"), export_params=True,opset_version=11)
    else:
        torch.onnx.export(model, dummy_input[0],  os.path.join(results_dir, "pruned_model.onnx"), export_params=True,opset_version=11)
    

    #### ONNX Runtime evaluation
    # Evaluate the original model
    if experiment_config['task'] == 'Restoration':
        original_model_evaluation = run_evaluation(os.path.join(results_dir,"original_model.onnx"), eval_dataloader)
        print(f'Original Model Evaluation Metric: {original_model_evaluation}')
        results_dict['original_model_onnx'] = original_model_evaluation

        # Evaluate the quantized model
        quantized_model_evaluation = run_evaluation(os.path.join(results_dir, "pruned_model.onnx"), eval_dataloader)
        print(f'Quantized Model Evaluation Metric: {quantized_model_evaluation}')
        results_dict['quantized_model_onnx'] = quantized_model_evaluation


    with open(os.path.join(results_dir,"pruning_eval_results.json"), 'w') as json_file:
        json.dump(results_dict, json_file, indent = 4)

if __name__ == '__main__':
    main()