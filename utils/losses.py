import torch
from torch import nn
import numpy as np
from mmdet.core import multi_apply, reduce_mean

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
    


# ### YuNET loss
# from ..archs.builder import build_loss

# class YuNETLoss(nn.module):

#     def __init__(self,
#                     loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=True,
#                     reduction='sum',
#                     loss_weight=1.0),
#                 loss_bbox=dict(
#                     type='EIoULoss',
#                     mode='square',
#                     eps=1e-16,
#                     reduction='sum',
#                     loss_weight=5.0),
#                 use_kps=True,
#                 kps_num=5,
#                 train_cfg=None,
#                 test_cfg=None,
#                 loss_kps=dict(
#                 type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.1),
#                 loss_obj=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=True,
#                     reduction='sum',
#                     loss_weight=1.0),
#     ):
#             self.use_kps = use_kps
#             self.kps_num = kps_num
#             self.loss_cls = build_loss(loss_cls)
#             self.loss_bbox = build_loss(loss_bbox)
#             if self.use_kps:
#                 self.loss_kps = build_loss(loss_kps)
#             self.loss_obj = build_loss(loss_obj)
#             self.strides = self.prior_generator.strides
#             self.strides_num = len(self.strides)

#             self.test_cfg = test_cfg
#             self.train_cfg = train_cfg

#     def loss(self,
#                 cls_scores,
#                 bbox_preds,
#                 objectnesses,
#                 kps_preds,
#                 gt_bboxes,
#                 gt_labels,
#                 gt_kpss,
#                 img_metas,
#                 gt_bboxes_ignore=None):
#             """Compute loss of the head.
#             Args:
#                 cls_scores (list[Tensor]): Box scores for each scale level,
#                     each is a 4D-tensor, the channel number is
#                     num_priors * num_classes.
#                 bbox_preds (list[Tensor]): Box energies / deltas for each scale
#                     level, each is a 4D-tensor, the channel number is
#                     num_priors * 4.
#                 objectnesses (list[Tensor], Optional): Score factor for
#                     all scale level, each is a 4D-tensor, has shape
#                     (batch_size, 1, H, W).
#                 gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                     shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#                 gt_labels (list[Tensor]): class indices corresponding to each box
#                 img_metas (list[dict]): Meta information of each image, e.g.,
#                     image size, scaling factor, etc.
#                 gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                     boxes can be ignored when computing the loss.
#             """
#             num_imgs = len(img_metas)
#             featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
#             mlvl_priors = self.prior_generator.grid_priors(
#                 featmap_sizes,
#                 dtype=cls_scores[0].dtype,
#                 device=cls_scores[0].device,
#                 with_stride=True)

#             flatten_cls_preds = [
#                 cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
#                                                     self.num_classes)
#                 for cls_pred in cls_scores
#             ]
#             flatten_bbox_preds = [
#                 bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
#                 for bbox_pred in bbox_preds
#             ]
#             flatten_objectness = [
#                 objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
#                 for objectness in objectnesses
#             ]
#             flatten_kps_preds = [
#                 kps_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.NK * 2)
#                 for kps_pred in kps_preds
#             ]

#             flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
#             flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
#             flatten_objectness = torch.cat(flatten_objectness, dim=1)
#             flatten_kps_preds = torch.cat(flatten_kps_preds, dim=1)

#             flatten_priors = torch.cat(mlvl_priors).unsqueeze(0).repeat(
#                 num_imgs, 1, 1)

#             flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

#             (pos_masks, cls_targets, obj_targets, bbox_targets, kps_targets,
#             kps_weights, num_fg_imgs) = multi_apply(self._get_target_single,
#                                                     flatten_cls_preds.detach(),
#                                                     flatten_objectness.detach(),
#                                                     flatten_priors,
#                                                     flatten_bboxes.detach(),
#                                                     gt_bboxes, gt_labels, gt_kpss)

#             # The experimental results show that ‘reduce_mean’ can improve
#             # performance on the COCO dataset.
#             num_pos = torch.tensor(
#                 sum(num_fg_imgs),
#                 dtype=torch.float,
#                 device=flatten_cls_preds.device)
#             num_total_samples = max(reduce_mean(num_pos), 1.0)

#             pos_masks = torch.cat(pos_masks, 0)
#             cls_targets = torch.cat(cls_targets, 0)
#             obj_targets = torch.cat(obj_targets, 0)
#             bbox_targets = torch.cat(bbox_targets, 0)
#             kps_targets = torch.cat(kps_targets, 0)
#             kps_weights = torch.cat(kps_weights, 0)

#             loss_bbox = self.loss_bbox(
#                 flatten_bboxes.view(-1, 4)[pos_masks],
#                 bbox_targets) / num_total_samples
#             loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
#                                     obj_targets) / num_total_samples
#             loss_cls = self.loss_cls(
#                 flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
#                 cls_targets) / num_total_samples
#             # use focalloss
#             # loss_cls = self.loss_cls(
#             #     flatten_cls_preds.view(-1, self.num_classes),
#             #                             cls_targets)
#             if self.use_kps:
#                 encoded_kpss = self._kps_encode(
#                     flatten_priors.view(-1, 4)[pos_masks], kps_targets)

#                 loss_kps = self.loss_kps(
#                     flatten_kps_preds.view(-1, self.NK * 2)[pos_masks],
#                     encoded_kpss,
#                     weight=kps_weights.view(-1, 1),
#                     # reduction_override='sum',
#                     avg_factor=torch.sum(kps_weights))
#             loss_dict = dict(
#                 loss_cls=loss_cls,
#                 loss_bbox=loss_bbox,
#                 loss_obj=loss_obj,
#                 loss_kps=loss_kps)

#             return loss_dict