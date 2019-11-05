from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
import cupy as cp
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox, bbox2loc, bbox_iou
from model.utils.nms import non_maximum_suppression
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

import torch
from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from torch.autograd import Variable
from utils.config import opt
from tqdm import tqdm
import pickle
import os
from torchvision import transforms as tvtsf



def nograd(f):
    def new_f(*args,**kwargs):
        with t.no_grad():
           return f(*args,**kwargs)
    return new_f

class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two functions :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :meth:`predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as
            :class:`model.region_proposal_network.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

    def __init__(self, extractor, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def forward(self, x, scale=1.):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        
        return roi_cls_locs, roi_scores, rois, roi_indices


    def forward_with_penultimate_features(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores, head_features = self.head(h, rois, roi_indices, return_penultimate=True)
        
        return roi_cls_locs, roi_scores, rois, roi_indices, head_features


    def forward_with_all_features(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores, head_features = self.head(h, rois, roi_indices, return_features=True)
        
        return roi_cls_locs, roi_scores, rois, roi_indices, head_features


    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        # if preset == 'visualize':
        #     self.nms_thresh = 0.3
        #     self.score_thresh = 0.7
        # elif preset == 'evaluate':
        #     self.nms_thresh = 0.3
        #     self.score_thresh = 0.05
        if preset == 'visualize':
            self.nms_thresh = 0.0
            self.score_thresh = 0.00
        elif preset == 'evaluate':
            self.nms_thresh = 0.0
            self.score_thresh = 0.00
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


    def _suppress_with_penultimate(self, raw_cls_bbox, raw_prob, raw_head_feats):
        import pdb; pdb.set_trace()
        bbox = list()
        label = list()
        score = list()
        features = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            feats_l = raw_head_feats
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            feats_l = feats_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
            features.append(feats_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        features = np.concatenate(features, axis=0).astype(np.float32)
        return bbox, label, score, features


    def _suppress_with_features(self, raw_cls_bbox, raw_prob, raw_head_feats):
        # import pdb; pdb.set_trace()
        bbox = list()
        label = list()
        score = list()
        features = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            feats_l = raw_head_feats
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            feats_l = [f[mask] for f in feats_l]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
            features.append([f[keep] for f in feats_l])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        features = [np.concatenate(list(f[i] for f in features), axis=0) for i in range(len(features[0]))]
        return bbox, label, score, features
        
        
    def _reform_raw_cls_bbox(self, raw_cls_bbox, raw_prob):
        """Reforms raw features to bboxes without non-maximum suppression."""
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            bbox.append(cls_bbox_l[:])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(cls_bbox_l),)))
            score.append(prob_l)
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs 
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None], cuda=True).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)            
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores


    def input_perturbation_mahalanobis(self, imgs, scale, epsilon=0.0005):
        # zero grad
        self.optimizer.zero_grad()
        
        _, _, H, W = imgs.shape
        img_size = (H, W)
        
        imgs = at.totensor(imgs)
        scale = at.scalar(scale)
        # Allow gradient on input imgs
        imgs.requires_grad_()

        roi_cls_loc, roi_scores, rois, _, head_feats = self.forward_with_penultimate_features(imgs, scale=scale)

        # Use the head features to predict classes and distances
        labels, distances = self.predict_label_mahalanobis(head_feats)
    
        # Backprop the distances
        distances.sum().backward()

        # Apply epsilon * signof input grad
        perturbation = epsilon * imgs.grad.sign()

        return imgs - perturbation


    def input_perturbation_odin(self, imgs, scale, epsilon, temper):
        # zero grad
        self.optimizer.zero_grad()
        
        _, _, H, W = imgs.shape
        img_size = (H, W)
        
        imgs = at.totensor(imgs)
        scale = at.scalar(scale)
        # Allow gradient on input imgs
        imgs.requires_grad_()

        # Get the class predictions
        roi_cls_loc, roi_scores, rois, _ = self.forward(imgs, scale=scale)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.squeeze(np.argmax(roi_scores.detach().cpu().numpy(), axis=1))
        labels = torch.tensor(maxIndexTemp).cuda()        

        # Using temperature scaling
        outputs = roi_scores / float(temper)
        
        # TODO: See if getting rid of the ignore index makes results better
        loss = nn.CrossEntropyLoss(ignore_index=-1)(outputs, labels)  # Should we ignore index=-1?
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  imgs.grad.sign()
        normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
        gradient = normalize(gradient[0]).unsqueeze_(0)

        # # Adding small perturbations to images
        perturbation = epsilon * gradient

        return imgs - perturbation


    # @nograd
    def predict_mahalanobis(self, imgs, sizes=None, visualize=False, perturbation=0):
        """Same as predict function but predicts class of objects using Mahalanobis distance."""
        # self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs 
        bboxes = list()
        labels = list()
        scores = list()
        dists = list()
        
        for img, size in zip(prepared_imgs, sizes):
            scale = img.shape[2] / size[1]

            img = at.totensor(img[None], cuda=True).float()

            if perturbation != 0:
                img = self.input_perturbation_mahalanobis(img, scale, epsilon=perturbation)

            roi_cls_loc, roi_scores, rois, _, head_feats = self.forward_with_penultimate_features(img, scale=scale)

            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)
            head_feats = at.tonumpy(head_feats)

            bbox, label, score, head_feats = self._suppress_with_penultimate(raw_cls_bbox, raw_prob, head_feats)

            if len(bbox) > 0:
                # import pdb; pdb.set_trace()
                label_dists = self.predict_label_mahalanobis(at.totensor(head_feats))
                bboxes.append(bbox)
                labels.append(at.tonumpy(label_dists[0]).astype(np.int8)) # Use the mahalanobis predicted labels rather than softmax
                scores.append(score)
                dists.append(at.tonumpy(label_dists[1]).astype(np.float32))
            else:
                bboxes.append(np.empty(shape=(0,4), dtype=np.float32))
                labels.append(np.empty(shape=(0), dtype=np.int32))
                scores.append(np.empty(shape=(0), dtype=np.float32))
                dists.append(np.empty(shape=(0), dtype=np.float32))

        self.use_preset('evaluate')
        # self.train()

        return bboxes, labels, dists



    # @nograd
    def predict_with_features(self, imgs, sizes=None, visualize=False, perturbation=0, temperature=1):
        """Same as predict function but predicts class of objects using Mahalanobis distance."""
        # self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs 
        bboxes = list()
        labels = list()
        scores = list()
        features = list()
        
        for img, size in zip(prepared_imgs, sizes):
            scale = img.shape[2] / size[1]

            img = at.totensor(img[None], cuda=True).float()

            h = self.extractor(img)
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img.shape[2:], scale)
            roi_cls_locs, roi_scores, head_feats = self.head(h, rois, roi_indices, return_features=True)

            # If ODIN parameters passed, perturb image and get new scores
            if perturbation != 0:
                img = self.input_perturbation_odin(img, scale, epsilon=perturbation, temper=temperature)
                with torch.no_grad():
                    h = self.extractor(img)
                    _, roi_scores = self.head(h, rois, roi_indices)
            
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_locs.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.totensor(roi_score) / float(temperature), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)
            head_feats = [at.tonumpy(f) for f in head_feats]

            bbox, label, score, head_feats = self._suppress_with_features(raw_cls_bbox, raw_prob, head_feats)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
            features.append(head_feats)

        # self.use_preset('evaluate')

        return bboxes, labels, scores, features


    def predict_label_mahalanobis(self, features):
        """Given a set of features, predict the class label. Requires training the mahal_means and inv_mahal_cov.
            Features is of shape (batch, head_feature_length)
            to_label: if int is specified, will only calculate distance to this label

            Return  label (int): The label of the class
                    distance (float): Mahalanobis distance from nearest class mean
        """
        # Tile the features and means to vectorize the operation
        num_means = self.mahal_means.shape[0]
        num_feats = features.shape[0]
        
        features = features.repeat(1, num_means).view(-1, features.shape[1])
        
        # subtract means in batches
        x = features - self.mahal_means.repeat(num_feats, 1)
        
        # matmul, take diagonal, then reshape
        dists = x.mm(self.inv_mahal_cov).mm(x.transpose(0,1)).diag()
        dists = dists.view(num_feats, num_means).transpose(0,1)
        
        min_dists, labels = dists.min(dim=0)
        return labels, min_dists


    @nograd
    def _predict_gt_features(self, imgs, gt_bboxes, gt_labels, scales):
        """Get the mahalanobis features needed to predict novelty score with predict_ood method.
            Use the gt_bbox explicitely to get the needed features.
        """
        self.eval()
        prepared_imgs = imgs 
        bboxes = list()
        labels = list()
        scores = list()
        features = list()
        
        for img, scale, gt_bbox, gt_label in zip(prepared_imgs, scales, gt_bboxes, gt_labels): 
            _, _, H, W = imgs.shape
            img_size = (H, W)
            img = at.totensor(img[None], cuda=True).float()
            scale = at.scalar(scale)

            h = self.extractor(img)
            _, _, rois, roi_indices, _ = self.rpn(h, img.shape[2:], scale)
            roi_cls_loc, roi_scores, head_feats = self.head(h, rois, roi_indices, return_features=True)
            # END SUB

            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale
            
            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=img_size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=img_size[1])

            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)
            raw_feats = at.tonumpy(head_feats)

            bbox, label, score, feats = self._suppress_with_penultimate(raw_cls_bbox, raw_prob, raw_feats)

            # For each gt bbox with associated label:
            #   If there is a predicted bbox and label with iou > 0.5 and label matching:
            #   then add the features for that prediction to the output
            
            for b, l in zip(gt_bbox, gt_label):
                gt_b = np.array([at.tonumpy(b)]) / scale
                gt_l = at.tonumpy(l)
                if len(bbox) > 0:
                    preds_bboxs_with_matching_label = bbox[gt_l == label]
                    preds_feats_with_matching_label = feats[gt_l == label]
                    preds_labels_with_matching_label = label[gt_l == label]
                    ious = bbox_iou(gt_b, preds_bboxs_with_matching_label)
                    features.append(preds_feats_with_matching_label[ious[0] > 0.5])
                    labels.append(preds_labels_with_matching_label[ious[0] > 0.5])

            bboxes.append(bbox)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return features, labels
        # return bboxes, labels, scores


    def _calc_mahal_means(self, features, labels):
        """Return the mahalanobis mean vector for each class.
            Shape will be [num_classes, feature_vector_length]
        """
        self.mahal_means = list()

        for c in range(self.n_class - 1):
            selection = labels == c
            if any(selection):
                mu_c = np.mean(features[selection], axis=0)
            else:
                mu_c = -1
                
            self.mahal_means.append(mu_c)


    def _calc_mahal_covariance_matrix(self, features, labels):
        """Return the mahalanobis covariance matrix for each class.
            Shape will be [num_classes, feature_vector_length, feature_vector_length]
        """
        sigma = np.zeros((4096, 4096))
        
        for c in range(self.n_class - 1):
            selection = labels == c
            if any(selection):
                feats = features[selection]
                mu_c = self.mahal_means[c]
                feats = feats - mu_c
                sigma += np.dot(feats.T, feats)
        
        self.mahal_cov = sigma / len(features)


    def _invert_mahal_covariance_matrix(self, epsilon=1e-18):
        self.inv_mahal_cov = np.linalg.inv(self.mahal_cov + np.eye(len(self.mahal_cov)) + epsilon)


    def train_ood(self, dataloader, num_train=opt.train_num):
        features = []
        gt_labels = []

        print("extracting features")
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader), total=num_train):
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # features_ = self._predict_gt_features(img, bbox, label, scale)
            features_, labels_ = self._predict_gt_features(img, bbox, label, scale)
            if len(features_) > 0:
                features.append(at.tonumpy(features_[0]))
                gt_labels.append(at.tonumpy(labels_[0]))
                
            if ii == num_train:
                break

        features = np.concatenate(features, axis=0)
        gt_labels = np.concatenate(gt_labels, axis=0)
        
        self.features = features # TODO: remove this after testing
        self.gt_labels = gt_labels # TODO: remove this after testing
        
        print("calculating feature means")
        self._calc_mahal_means(features, gt_labels)
        print("calculating feature covariance")
        self._calc_mahal_covariance_matrix(features, gt_labels)
        print("inverting feature covariance")
        self._invert_mahal_covariance_matrix()
        
        self.mahal_means = at.totensor(np.array(self.mahal_means))
        self.mahal_cov = at.totensor(self.mahal_cov).float()
        self.inv_mahal_cov = at.totensor(self.inv_mahal_cov).float()

        self.save_mahalanobis_features(save_dir='./')


    def save_mahalanobis_features(self, save_dir='./checkpoints'):
        """Save the mahalanobis means, covariance, and inv covariance."""
        with open(os.path.join(save_dir,'kitti_features.pickle'), 'wb') as f:
            pickle.dump(self.features, f)
        with open(os.path.join(save_dir, 'mahal_means.pickle'), 'wb') as f:
            pickle.dump(self.mahal_means, f)
        with open(os.path.join(save_dir, 'mahal_cov.pickle'), 'wb') as f:
            pickle.dump(self.mahal_cov, f)
        with open(os.path.join(save_dir, 'inv_mahal_cov.pickle'), 'wb') as f:
            pickle.dump(self.inv_mahal_cov, f)


    def load_mahalanobis_features(self, save_dir='./checkpoints'):
        """Load the mahalanobis means, covariance, and inv covariance."""
        with open(os.path.join(save_dir,'kitti_features.pickle'), 'rb') as f:
            self.features = pickle.load(f)
        with open(os.path.join(save_dir, 'mahal_means.pickle'), 'rb') as f:
            self.mahal_means = pickle.load(f)
        with open(os.path.join(save_dir, 'mahal_cov.pickle'), 'rb') as f:
            self.mahal_cov = pickle.load(f)
        with open(os.path.join(save_dir, 'inv_mahal_cov.pickle'), 'rb') as f:
            self.inv_mahal_cov = pickle.load(f)


    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer


    @nograd
    def get_rpn_output(self, imgs, scales):
        """Get the objectness scores for all anchor boxes."""
        self.eval()
        prepared_imgs = imgs
        # bboxes = list()
        # labels = list()
        # scores = list()
        
        for img, scale in zip(prepared_imgs, scales): 
            _, _, H, W = imgs.shape
            # img_size = (H, W)
            img = at.totensor(img[None], cuda=True).float()
            scale = at.scalar(scale)

            h = self.extractor(img)
            return self.rpn(h, img.shape[2:], scale)


    @nograd
    def predict_with_bg(self, imgs, sizes=None, visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs 
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None], cuda=True).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._dummy_suppress(raw_cls_bbox, raw_prob)            
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores


    def _dummy_suppress(self, raw_cls_bbox, raw_prob):
        """Same inputs and outputs as NMS function but includes the background class."""
        bbox = list()
        label = list()
        score = list()
        for l in range(0, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > 0.1  # self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score