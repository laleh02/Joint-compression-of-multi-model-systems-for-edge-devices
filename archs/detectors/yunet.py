import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

@DETECTORS.register_module()
class YuNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YuNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)

    def forward(self,
                      img,
                      #img_metas,
                      #gt_bboxes,
                      #gt_labels,
                      #gt_keypointss=None,
                      #gt_bboxes_ignore=None, 
                      rescale=False):
        #img_metas
        #super(SingleStageDetector, self).forward(img)
        x = self.extract_feat(img)
        cls_preds, bbox_preds, obj_preds, kps_preds = self.bbox_head(x)
        #bbox_list = []#self.bbox_head.get_bboxes(cls_preds, bbox_preds, obj_preds, kps_preds)
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore)
        return cls_preds, bbox_preds, obj_preds, kps_preds#, bbox_list

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if torch.onnx.is_in_onnx_export():
            return outs
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        # if torch.onnx.is_in_onnx_export():
        #    return bbox_list


        #bbox_results = [
        #    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #    for det_bboxes, det_labels in bbox_list['bbox']
        #]
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'
        print('aug-test:', len(imgs))
        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
