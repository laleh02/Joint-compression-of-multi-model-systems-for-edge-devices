import torch
import argparse
import os
import cv2
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

    opt, output_path, args = parse_options(parser)
    output_path = os.path.join(output_path,opt['experiment_name'] + ".png")

    model = model_factory.build_model(opt)
    model.eval()

    # Load the image into a NumPy array
    numpy_img = np.array(Image.open(args.img))

    # Convert the NumPy array to a torch tensor
    tensor_img = img2tensor(numpy_img)

    # Run inference using your model
    cls_preds, bbox_preds, obj_preds, kps_preds = model(tensor_img)


    result_list, _,  kps_list = model.bbox_head.get_bboxes(cls_preds, bbox_preds, obj_preds, kps_preds)
    if opt['model_name'] == 'PipelinedYuNet':
        enhanced_frame = model.output_enhancers_result(tensor_img)
        enhanced_frame = tensor2img(enhanced_frame)
    else : 
        enhanced_frame = numpy_img
    try:
        result = result_list[0][0]
        kps = kps_list[0][0].cpu().numpy().astype(np.uint8)

                    
        # Extract rectangle coordinates
        x1, y1, x2, y2 = result[:4].cpu().numpy().astype(np.uint8)
        # Draw the rectangle
        cv2.rectangle(enhanced_frame, (y1, x1), (y2, x2), (0, 255, 0), 2)

        # Loop through the remaining values in pairs and draw the points
        for i in range(0, len(kps), 2):
            x, y = kps[i], kps[i + 1]
            cv2.circle(enhanced_frame, (x, y), 3, (0, 0, 255), -1)
    except: 
        pass


    # Save the image to disk
    imwrite(enhanced_frame, output_path)
    

if __name__ == '__main__':

    main()