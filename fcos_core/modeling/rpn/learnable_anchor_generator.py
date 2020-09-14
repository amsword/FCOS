# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch import nn

from fcos_core.structures.bounding_box import BoxList


def reverse(x):
    return x.log()

def forward(x):
    return x.exp()

class LearnableAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,
    ):
        super().__init__()

        if len(anchor_strides) != len(sizes):
            raise RuntimeError("FPN should have #anchor_strides == #sizes")

        all_sizes = [s if isinstance(s, (tuple, list)) else (s,)  for s in sizes]
        all_sizes = [torch.tensor(ss).float() for ss in all_sizes]

        all_anchors = []
        for ss in all_sizes:
            anchors = []
            for s in ss:
                for r in aspect_ratios:
                    w = torch.sqrt(s * s / r)
                    h = w * r
                    anchors.append(torch.tensor([w, h]))
            all_anchors.append(torch.stack(anchors, dim=0))

        all_pre_anchors = [torch.nn.Parameter(reverse(anchors)) for anchors in all_anchors]
        self.all_pre_anchors = torch.nn.ParameterList(all_pre_anchors)
        self.strides = anchor_strides
        #self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh
        self._num_anchors_per_location = [len(pre_anchors) for pre_anchors in
                                          self.all_pre_anchors]

    def num_anchors_per_location(self):
        return self._num_anchors_per_location

    def grid_anchors(self, grid_sizes):
        cell_anchors = []
        for s, pre_anchors in zip(self.strides, self.all_pre_anchors):
            anchors = forward(pre_anchors)
            left = s / 2 - anchors[:, 0] / 2.
            right = s / 2 + anchors[:, 0] / 2.
            top = s / 2 - anchors[:, 1] / 2.
            bottom = s / 2 + anchors[:, 1] / 2.
            cell_anchors.append(torch.stack([left, top, right, bottom], dim=1))
        import ipdb;ipdb.set_trace(context=15)

        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors



# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


