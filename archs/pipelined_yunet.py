import torch

from mmdet.core import bbox2result
from .builder import DETECTORS
from .detectors.single_stage import SingleStageDetector
from .detectors.yunet import YuNet

@DETECTORS.register_module()
class PipelinedYuNet(YuNet):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PipelinedYuNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)
        self.enhancers = torch.nn.Sequential()

    def set_enhancers(self, enhancer_list) : 
        assert len(enhancer_list) != 0
        self.enhancers = torch.nn.Sequential(*enhancer_list)
    def forward(self,
                      img,
                      rescale=False):
        if isinstance(img,list):
            img = img[0]
        img = self.enhancers(img)
        x = self.extract_feat(self.renormalize_input(img))
        cls_preds, bbox_preds, obj_preds, kps_preds = self.bbox_head(x)

        return cls_preds, bbox_preds, obj_preds, kps_preds#, bbox_list


    def renormalize_input(self, input):
        # Enhancers work on images in [0,1] while detector work on images in [0, 255]
        return input
    def output_enhancers_result(self, input):
        return torch.clip(self.renormalize_input(self.enhancers(input)),0,255)
@DETECTORS.register_module()
class End2EndYuNet(YuNet):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(End2EndYuNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)


    def forward(self,
                      img,
                      rescale=False):
        if isinstance(img,list):
            img = img[0]
        if rescale:
            x = self.extract_feat(self.renormalize_input(img))
        else :         
            x = self.extract_feat(img)

        cls_preds, bbox_preds, obj_preds, kps_preds = self.bbox_head(x)
        result = self.bbox_head.get_bboxes(cls_preds, bbox_preds, obj_preds, kps_preds)
        return result


    def renormalize_input(self, input):
        # Enhancers work on images in [0,255] while detector work on images in [0, 255]
        return torch.round(torch.clip(input,0,255))
