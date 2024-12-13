import torch
import argparse
import os
from collections import OrderedDict
from archs.PMRID import PMRID
from archs.pipelined_yunet import PipelinedYuNet #import is needed to add module to registry
from utils.builder import init_detector
import utils.model_factory as model_factory
from utils.config_handler import parse_options





def main():

    opt, output_path, _ = parse_options()
    output_path = os.path.join(output_path,opt['experiment_name'] + ".onnx")
    model = model_factory.build_model(opt)
    model.eval()
    input = torch.randn(1, 3, 128, 128, requires_grad=True)

    torch.onnx.export(model,               
                    input,                         
                    output_path,  
                    export_params=True,        
                    opset_version=12,         
                    do_constant_folding=True, 
                    input_names = ['input'],   
                    output_names = ['output'], 
                    )

if __name__ == '__main__':

    main()