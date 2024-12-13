import torch
import os
from collections import OrderedDict
from archs.PMRID import PMRID
from archs.pipelined_yunet import PipelinedYuNet #import is needed to add module to registry
from utils.builder import init_detector


def build_model(opt):

    
    if opt['task'] == 'Detection':
        model = init_detector(opt['yunet_config']).to("cpu")
        if opt['model_name'] == 'Pipelined_YuNET':
            ll_enhancer = torch.load(opt[''],map_location='cpu')
            denoiser = torch.load(opt[''],map_location='cpu')
            if 'params' in ll_enhancer: #enhancer is a state_dict and not a model:
                ll_enhancer = PMRID()
                ll_enhancer.load_state_dict(torch.load(opt['enhancers']['LLE']['pretrain_network_g'],map_location='cpu')['params'])
            if 'params' in denoiser: #enhancer is a state_dict and not a model:
                denoiser = PMRID()
                denoiser.load_state_dict(torch.load(opt['enhancers']['denoise']['pretrain_network_g'],map_location='cpu')['params'])
            model.set_enhancers([denoiser,ll_enhancer])

    elif opt['task'] == "Restoration":
        model = PMRID()
        model.load_state_dict(torch.load(opt['path']['pretrain_network_g'])['params'])

    
    return model