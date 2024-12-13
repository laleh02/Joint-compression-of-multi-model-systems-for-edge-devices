from archs.detectors.yunet import YuNet
from mmcv.runner import load_checkpoint
import mmcv 
import warnings
from utils.builder import build_detector
from pathlib import Path
from archs.PMRIDd2 import PMRIDd2

from utils.data import create_dataloader, Detection_Dataset
import torch
from utils.image_processing import resize_img, draw
import cv2
from tqdm import tqdm
from utils.metrics import calculate_nme
from archs.e2e_yunet import End2EndYuNet

###debug imports



def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            raise Exception("You need a checkpoints to run experiments in AIMET.")
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


yunet = init_detector(config= "experiment_config/pipelined_yunet_n.py",checkpoint="weights/yunet_n_retrained.pth")


eval_dataset = Detection_Dataset(config = {'dataroot': 'logynthetic/test',
                                            'gt_suffix': '1_noisy',
                                            'labels_path': 'logynthetic/test_labels'})
eval_dataloader = create_dataloader(eval_dataset,config = {'datasets':
                                                               {'dataloader': {'batch_size' : 1}}})

for idx, data in tqdm(enumerate(eval_dataloader)):

    with torch.no_grad():
        import matplotlib.pyplot as plt
        img = data['img'].squeeze()
        numpy_image = img.permute(1, 2, 0).numpy()
        # Convert the numpy array to a cv2 image
        cv2_image = cv2.cvtColor(numpy_image,cv2.COLOR_RGB2BGR)
        cv2.imwrite("test2_lq.png", cv2_image)

        det_img, det_scale = resize_img(cv2_image, 'AUTO')
        metas = [{
        'img_shape': det_img.shape,
        'ori_shape': cv2_image.shape,
        'pad_shape': det_img.shape,
        'scale_factor': tuple([det_scale for _ in range(4)])
        }]
        if True: 
                lle_enhancer = PMRIDd2()
                denoising_enhancer = PMRIDd2()
                lle_enhancer.load_state_dict(torch.load("./weights/PMRIDd2_denoise.pth")['params'])
                denoising_enhancer.load_state_dict(torch.load("./weights/PMRIDd2_LLE.pth")['params'])
                yunet.set_enhancers([denoising_enhancer.to('cuda'), lle_enhancer.to('cuda')])
        result = yunet(img = data['img'].to('cuda'))
        result = yunet.bbox_head.get_bboxes(*result)
        result = {'bbox': [result[0][0]], 'labels' : [result[1][0]],'kps' : [result[2][0]]} #TODO : OMG the ground truth is [y1 x1 y2 x2] and model outputs were [x1 y1 x2 y2], corrected in the lines below......
        print("PRED")
        print(len(result['bbox'][0]))
        print(result['bbox'][0].size())
        print(result['bbox'][0][0:2])

        print("GT")
        print(len(data['label']['person_0']))
        #print(calculate_nme(data['label']['person_0'], result, device='cuda:0'))

        import time
        time.sleep(1)
    if idx == 1: 
        draw(cv2_image, result['bbox'], result['kps'], "test_lq.png", True)
        break
