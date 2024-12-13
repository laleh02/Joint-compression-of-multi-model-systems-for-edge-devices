import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.core.anchor.builder import build_prior_generator
from ..builder import HEADS, build_loss
from ..utils.yunet_layer import ConvDPUnit
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from collections import namedtuple


@HEADS.register_module()
class YuNet_Head(BaseDenseHead, BBoxTestMixin):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        shared_stacked_convs=2,
        stacked_convs=2,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        use_kps=False,
        kps_num=5,
        loss_kps=None,
        prior_generator=None,
        train_cfg=None,
        test_cfg=None,
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
    ):

        super().__init__()
        self.num_classes = num_classes
        self.NK = kps_num
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_sigmoid_cls = True
        self.use_kps = use_kps
        self.shared_stack_convs = shared_stacked_convs

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        if self.use_kps:
            self.loss_kps = build_loss(loss_kps)
        self.loss_obj = build_loss(loss_obj)
        self.prior_generator = build_prior_generator(prior_generator)
        self.strides = self.prior_generator.strides
        self.strides_num = len(self.strides)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        if self.shared_stack_convs > 0:
            self.multi_level_share_convs = nn.ModuleList()
        if self.stacked_convs > 0:
            self.multi_level_cls_convs = nn.ModuleList()
            self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_cls = nn.ModuleList()
        self.multi_level_bbox = nn.ModuleList()
        self.multi_level_obj = nn.ModuleList()
        if self.use_kps:
            self.multi_level_kps = nn.ModuleList()
        for _ in self.strides:
            if self.shared_stack_convs > 0:
                single_level_share_convs = []
                for i in range(self.shared_stack_convs):
                    chn = self.in_channels if i == 0 else self.feat_channels
                    single_level_share_convs.append(
                        ConvDPUnit(chn, self.feat_channels))
                self.multi_level_share_convs.append(
                    nn.Sequential(*single_level_share_convs))

            if self.stacked_convs > 0:
                single_level_cls_convs = []
                single_level_reg_convs = []
                for i in range(self.stacked_convs):
                    chn = self.in_channels if i == 0 and \
                        self.shared_stack_convs == 0 else self.feat_channels
                    single_level_cls_convs.append(
                        ConvDPUnit(chn, self.feat_channels))
                    single_level_reg_convs.append(
                        ConvDPUnit(chn, self.feat_channels))
                self.multi_level_reg_convs.append(
                    nn.Sequential(*single_level_reg_convs))
                self.multi_level_cls_convs.append(
                    nn.Sequential(*single_level_cls_convs))

            chn = self.in_channels if self.stacked_convs == 0 and \
                self.shared_stack_convs == 0 else self.feat_channels
            self.multi_level_cls.append(
                ConvDPUnit(chn, self.num_classes, False))
            self.multi_level_bbox.append(ConvDPUnit(chn, 4, False))
            if self.use_kps:
                self.multi_level_kps.append(
                    ConvDPUnit(chn, self.NK * 2, False))
            self.multi_level_obj.append(ConvDPUnit(chn, 1, False))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # bias_cls = -4.595
        # for m in self.cls_convs.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if m.bias is not None:
        #             m.bias.data.fill_(bias_cls)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        if self.shared_stack_convs > 0:
            feats = [
                convs(feat)
                for feat, convs in zip(feats, self.multi_level_share_convs)
            ]

        if self.stacked_convs > 0:
            feats_cls, feats_reg = [], []
            for i in range(self.strides_num):
                feats_cls.append(self.multi_level_cls_convs[i](feats[i]))
                feats_reg.append(self.multi_level_reg_convs[i](feats[i]))
            cls_preds = [
                convs(feat)
                for feat, convs in zip(feats_cls, self.multi_level_cls)
            ]
            bbox_preds = [
                convs(feat)
                for feat, convs in zip(feats_reg, self.multi_level_bbox)
            ]
            obj_preds = [
                convs(feat)
                for feat, convs in zip(feats_reg, self.multi_level_obj)
            ]
            kps_preds = [
                convs(feat)
                for feat, convs in zip(feats_reg, self.multi_level_kps)
            ]
        else:
            cls_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_cls)
            ]
            bbox_preds = [
                convs(feat)
                for feat, convs in zip(feats, self.multi_level_bbox)
            ]
            obj_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_obj)
            ]
            kps_preds = [
                convs(feat) for feat, convs in zip(feats, self.multi_level_kps)
            ]

        if torch.onnx.is_in_onnx_export():
            cls = [
                f.permute(0, 2, 3, 1).view(f.shape[0], -1,
                                           self.num_classes).sigmoid()
                for f in cls_preds
            ]
            obj = [
                f.permute(0, 2, 3, 1).view(f.shape[0], -1, 1).sigmoid()
                for f in obj_preds
            ]
            bbox = [
                f.permute(0, 2, 3, 1).view(f.shape[0], -1, 4)
                for f in bbox_preds
            ]
            kps = [
                f.permute(0, 2, 3, 1).view(f.shape[0], -1, self.NK * 2)
                for f in kps_preds
            ]
            return (cls, obj, bbox, kps)
        print("HEEEEEEEEEEERE")
        print(cls_preds, bbox_preds, obj_preds, kps_preds)
        print("-----------")
        return cls_preds, bbox_preds, obj_preds, kps_preds

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    # def get_bboxes(self,
    #                cls_scores,
    #                bbox_preds,
    #                objectnesses,
    #                kps_preds,
    #                img_metas=None,
    #                cfg=None,
    #                #rescale=False,
    #                with_nms=True):
    #     """Transform network outputs of a batch into bbox results.
    #     Args:
    #         cls_scores (list[Tensor]): Classification scores for all
    #             scale levels, each is a 4D-tensor, has shape
    #             (batch_size, num_priors * num_classes, H, W).
    #         bbox_preds (list[Tensor]): Box energies / deltas for all
    #             scale levels, each is a 4D-tensor, has shape
    #             (batch_size, num_priors * 4, H, W).
    #         objectnesses (list[Tensor], Optional): Score factor for
    #             all scale level, each is a 4D-tensor, has shape
    #             (batch_size, 1, H, W).
    #         img_metas (list[dict], Optional): Image meta info. Default None.
    #         cfg (mmcv.Config, Optional): Test / postprocessing configuration,
    #             if None, test_cfg would be used.  Default None.
    #         rescale (bool): If True, return boxes in original image space.
    #             Default False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Default True.
    #     Returns:
    #         list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
    #             The first item is an (n, 5) tensor, where the first 4 columns
    #             are bounding box positions (tl_x, tl_y, br_x, br_y) and the
    #             5-th column is a score between 0 and 1. The second item is a
    #             (n,) tensor where each item is the predicted class label of
    #             the corresponding box.
    #     """
    #     assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
    #     cfg = self.test_cfg if cfg is None else cfg

    #     num_imgs = 1 #TODO un-hard codecls_scores[0].shape[0]. This fixes the batch size for both testing and training. 

    #     featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    #     mlvl_priors = self.prior_generator.grid_priors(
    #         featmap_sizes,
    #         dtype=cls_scores[0].dtype,
    #         device=cls_scores[0].device,
    #         with_stride=True)
    #     flatten_priors = torch.cat(mlvl_priors).unsqueeze(0).repeat(
    #         num_imgs, 1, 1)

    #     # flatten cls_scores, bbox_preds and objectness
    #     flatten_cls_scores = [
    #         cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
    #                                               self.cls_out_channels)
    #         for cls_score in cls_scores
    #     ]
    #     flatten_bbox_preds = [
    #         bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
    #         for bbox_pred in bbox_preds
    #     ]
    #     flatten_objectness = [
    #         objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
    #         for objectness in objectnesses
    #     ]
    #     flatten_kps_preds = [
    #         bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
    #         for bbox_pred in bbox_preds
    #     ]
    #     flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    #     flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    #     flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
    #     flatten_priors = torch.cat(mlvl_priors)

    #     flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

    #     ##ADDED BY ME : FLATTEN KPS PREDS
    #     flatten_kps_preds = [
    #         kps_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.NK * 2)
    #         for kps_pred in kps_preds
    #     ]        
    #     flatten_kps_preds = torch.cat(flatten_kps_preds, dim=1)
    #     flatten_kps = self._kps_decode(flatten_priors, flatten_kps_preds)


    #     # if rescale:
    #     #     scale_factors = np.array(
    #     #         [img_meta['scale_factor'] for img_meta in img_metas])
    #     #     flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
    #     #         scale_factors).unsqueeze(1)


    #     result_list = []
    #     labels_list = []
    #     kps_list = []
    #     for img_id in range(num_imgs):

    #         cls_scores = flatten_cls_scores[img_id]
    #         score_factor = flatten_objectness[img_id]
    #         bboxes = flatten_bboxes[img_id]
    #         kps = flatten_kps[img_id]

    #         result = _bboxes_nms(cls_scores, bboxes, score_factor, cfg, kps)

    #         result_list.append(
    #             result['bbox'])
    #         labels_list.append(
    #             result['labels']
    #         )
    #         #result_list.append(
    #         #                    self._bboxes_nms(cls_scores, kps, score_factor, cfg))
    #         kps_list.append(result['kps'])
    #     return (result_list, labels_list,  kps_list)

    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[..., 2:]) + priors[..., :2]
        whs = bbox_preds[..., 2:].exp() * priors[..., 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)
        #NOTE : uninverted bboxes here, as it is the opposite of my code:
        decoded_bboxes = torch.stack([tl_y, tl_x, br_y, br_x], -1)
        return decoded_bboxes

    def _kps_decode(self, priors, kps_preds):
        num_points = 5 #TODO un-hardcodeint(kps_preds.shape[-1] / 2)
        decoded_kps = torch.cat(
            [(kps_preds[..., [2 * i, 2 * i + 1]] * priors[..., 2:]) +
             priors[..., :2] for i in range(num_points)], -1)
        return decoded_kps

    def _kps_encode(self, priors, kps):
        num_points = int(kps.shape[-1] / 2)
        encoded_kps = [
            (kps[..., [2 * i, 2 * i + 1]] - priors[..., :2]) / priors[..., 2:]
            for i in range(num_points)
        ]
        encoded_kps = torch.cat(encoded_kps, -1)
        return encoded_kps

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels, gt_kpss):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        gt_kpss = gt_kpss.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            kps_target = cls_preds.new_zeros((0, self.NK * 2))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target, 0)

        # Uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]

        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)

        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        # obj_target = pos_ious.unsqueeze(-1)

        bbox_target = sampling_result.pos_gt_bboxes

        kps_target = gt_kpss[pos_assigned_gt_inds, :, :2].reshape(
            (-1, self.NK * 2))
        kps_weight = torch.mean(
            gt_kpss[pos_assigned_gt_inds, :, 2], dim=1, keepdims=True)

        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                kps_target, kps_weight, num_pos_per_img)

#@torch.fx.wrap Uncomment this if untraceable parts of the code are to be reincorporated. This fixture makes the function to be traced as a functional call rather than as a module. 
def _bboxes_nms(cls_scores, bboxes, score_factor, cfg, kps):
    max_scores, labels = torch.max(cls_scores, 1)
    valid_mask = score_factor * max_scores >= cfg['score_thr']

    bboxes = bboxes[valid_mask]
    
    scores = max_scores[valid_mask] * score_factor[valid_mask]
    labels = labels[valid_mask]
    kps = kps[valid_mask]

    scores, inds = scores.sort(descending=True)
    boxes = bboxes[inds]
    kps = kps[inds]
    return  {'bbox': torch.cat([boxes, scores[:, None]], -1), 'labels' : labels,'kps' : kps}
    
    #cannot be traced, poubelle. 
    if labels.numel() == 0:
        return {'bbox': bboxes, 'labels' : labels, 'kps' : torch.empty(1)}
    else:



        dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
        if kps is not None:
            kps = kps[valid_mask]
            #  {'bbox': (dets, labels[keep]), 'kps' : kps[keep]}
            return {'bbox': dets, 'labels' : labels[keep],'kps' : kps[keep]}
        return {'bbox': dets, 'labels' : labels[keep], 'kps' : torch.empty(1)}