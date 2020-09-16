# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import boxlist_nms
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.modeling.box_coder import BoxCoder


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False,
        classification_activate='softmax',
        cfg=None,
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled
        self.nms_on_max_conf_agnostic = cfg.MODEL.ROI_HEADS.NMS_ON_MAX_CONF_AGNOSTIC
        if not self.cls_agnostic_bbox_reg:
            assert not self.nms_on_max_conf_agnostic
        self.classification_activate = classification_activate
        if classification_activate == 'softmax':
            self.logits_to_prob = lambda x: F.softmax(x, -1)
            self.cls_start_idx = 1
        elif classification_activate == 'sigmoid':
            self.logits_to_prob = torch.nn.Sigmoid()
            self.cls_start_idx = 0
        elif classification_activate == 'tree':
            from qd.layers import SoftMaxTreePrediction
            self.logits_to_prob = SoftMaxTreePrediction(
                    tree=cfg.MODEL.ROI_BOX_HEAD.TREE_0_BKG,
                    pred_thresh=self.score_thresh)
            self.cls_start_idx = 1
        else:
            raise NotImplementedError()

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = self.logits_to_prob(class_logits)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg and not self.nms_on_max_conf_agnostic:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if self.nms_on_max_conf_agnostic:
                boxlist = self.filter_results_nms_on_max(boxlist, num_classes)
            elif not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def prepare_empty_boxlist(self, boxlist):
        device = boxlist.bbox.device
        boxlist_empty = BoxList(torch.zeros((0,4)).to(device), boxlist.size,
                mode='xyxy')
        boxlist_empty.add_field("scores", torch.Tensor([]).to(device))
        boxlist_empty.add_field("labels", torch.full((0,), -1,
                dtype=torch.int64, device=device))
        return boxlist_empty

    def filter_results_nms_on_max(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist

        # cpu version is faster than gpu. revert it to gpu only by verifying
        boxlist = boxlist.to('cpu')

        boxes = boxlist.bbox.reshape(-1, 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        result = []
        max_scores, _ = scores[:, self.cls_start_idx:].max(dim=1, keepdim=False)
        keep = (max_scores > self.score_thresh).nonzero(as_tuple=False).squeeze(-1)
        if len(keep) == 0:
            return self.prepare_empty_boxlist(boxlist)
        boxes, scores, max_scores = boxes[keep], scores[keep], max_scores[keep]

        boxlist = BoxList(boxes, boxlist.size, mode=boxlist.mode)
        boxlist.add_field("scores", max_scores)
        boxlist.add_field('original_scores', scores)
        boxlist = boxlist_nms(boxlist, self.nms)

        scores = boxlist.get_field('original_scores')
        all_idxrow_idxcls = (scores[:, self.cls_start_idx:] > self.score_thresh).nonzero()
        all_idxrow_idxcls[:, 1] += self.cls_start_idx

        boxes = boxlist.bbox
        boxes = boxes[all_idxrow_idxcls[:, 0]]
        if boxes.dim() == 1:
            boxes = boxes[None, :]
        labels = all_idxrow_idxcls[:, 1]
        scores =  scores[all_idxrow_idxcls[:, 0], all_idxrow_idxcls[:, 1]]
        result = BoxList(boxes, boxlist.size, mode=boxlist.mode)
        result.add_field("labels", labels)
        result.add_field("scores", scores)

        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero(as_tuple=False).squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    classification_activate = cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_ACTIVATE

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled,
        classification_activate=classification_activate,
        cfg=cfg,
    )
    return postprocessor
