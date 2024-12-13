import torch
import argparse
import os
import numpy as np
from PIL import Image
from collections import OrderedDict
from archs.PMRID import PMRID
from archs.pipelined_yunet import PipelinedYuNet #import is needed to add module to registry
from utils.builder import init_detector
import utils.model_factory as model_factory
from utils.config_handler import parse_options
from utils.image_processing import img2tensor, tensor2img, imwrite



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='Path to experiment configuration YAML file.')
    parser.add_argument(
        '--img', 
        type=str, 
        required=False, 
        help='The image used in inference.', 
        default="assets\\example.png")
    parser.add_argument(
        '--output_folder', 
        type=str, 
        required=False, 
        help='The output folder for the onnx model.', 
        default="model_artifacts\\inference_examples")

    model_opt, output_path, args = parse_options(parser)
    output_path = os.path.join(output_path,model_opt['experiment_name'] + ".png")

    model = model_factory.build_model(model_opt)
    model.eval()

    # Load the image into a NumPy array
    numpy_img = np.array(Image.open(args.img))

    # Convert the NumPy array to a torch tensor
    tensor_img = img2tensor(numpy_img)

    # Run inference using your model
    inference_result = model(tensor_img)

    # Convert the inference result back to a NumPy array
    result_numpy_img = tensor2img(inference_result)

    # Save the image to disk
    imwrite(result_numpy_img, output_path)
    

if __name__ == '__main__':

    main()