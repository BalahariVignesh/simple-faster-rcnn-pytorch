from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
import cupy as cp
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox, bbox2loc, bbox_iou
from model.utils.nms import non_maximum_suppression
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
from data.dataset import preprocess
from torch.nn import functional as F
from utils.config import opt
from tqdm import tqdm


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


    def forward_with_head_features(self, x, scale=1.):
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
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
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

    def _suppress_with_feats(self, raw_cls_bbox, raw_prob, raw_head_feats):
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


    @nograd
    def predict_mahalanobis(self, imgs, sizes=None, visualize=False):
        """Same as predict function but predicts class of objects using Mahalanobis distance."""
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
        dists = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None], cuda=True).float()
            scale = img.shape[3] / size[1]

            roi_cls_loc, roi_scores, rois, _, head_feats = self.forward_with_head_features(img, scale=scale)

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

            bbox, label, score, head_feats = self._suppress_with_feats(raw_cls_bbox, raw_prob, head_feats)
            if len(bbox) > 0:
                # import pdb; pdb.set_trace()
                label_dists = self.predict_label_mahalanobis(head_feats)#, to_labels=label)
                bboxes.append(bbox)
                labels.append(label) # Use the softmax label
                # labels.append(label_dists[0].astype(np.int8)) # Use the mahalanobis predicted labels rather than softmax
                scores.append(score)
                dists.append(label_dists[1].astype(np.float32))
            else:
                bboxes.append(np.empty(shape=(0,4), dtype=np.float32))
                labels.append(np.empty(shape=(0), dtype=np.int32))
                scores.append(np.empty(shape=(0), dtype=np.float32))
                dists.append(np.empty(shape=(0), dtype=np.float32))

        self.use_preset('evaluate')
        self.train()

        return bboxes, labels, dists


    def predict_label_mahalanobis(self, features, to_labels=None):
        """Given a set of features, predict the class label. Requires training the mahal_means and inv_mahal_cov.
            Features is of shape (batch, head_feature_length)
            to_label: if int is specified, will only calculate distance to this label

            Return  label (int): The label of the class
                    distance (float): Mahalanobis distance from nearest class mean
        """
        def _dist_to_mean(mu_c, features):
            if type(mu_c) == type(-1) and to_labels is None:
                dist = np.array(float('inf'))
            elif type(mu_c) == type(-1):
                dist = np.ones(len(features)) * float('inf')
            x = (features - mu_c)
            dist = np.dot(np.dot(x, self.inv_mahal_cov), x.T)
            return dist

        dists = list()
        if to_labels is not None:
            for feature, label in zip(features, to_labels):
                dists.append(_dist_to_mean(self.mahal_means[label], feature))
            dists = np.array(dists)
            return to_labels, dists
        
        else:
            for mu_c in self.mahal_means:
                dists.append(np.diagonal(_dist_to_mean(mu_c, features)))
            dists = np.array(dists)
            return dists.argmin(axis=0), dists.min(axis=0)
        

    def _calc_mahal_means(self, features, labels):
        """Return the mahalanobis mean vector for each class.
            Shape will be [num_classes, feature_vector_length]
        """
        means = list()

        for c in range(self.n_class - 1):
            selection = labels == c
            if any(selection):
                mu_c = np.mean(features[selection], axis=0)
            else:
                mu_c = -1
                
            means.append(mu_c)
        
        return means


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
    
        return sigma / len(features)


    def _predict_novelty_score(self, features):
        """Given the feature layer, predict the novelty score."""
        distances = list()

        for mu_c in self.mahal_mean:
            x = features - mu_c
            M = np.dot(x.T, self.inv_mahal_cov)
            M = -np.dot(M, x)
            distances.append(M)

        return max(distances)

    
    @nograd
    def _predict_gt_features_roi_iou_method(self, imgs, gt_bboxes, gt_labels, scales):
        """Get the mahalanobis features needed to predict novelty score with predict_ood method."""
        self.eval()
        prepared_imgs = imgs 
        features = list()

        for img, scale, gt_bbox, gt_label in zip(prepared_imgs, scales, gt_bboxes, gt_labels):
            _, _, H, W = imgs.shape
            img_size = (H, W)
            img = at.totensor(img[None]).float()
            scale = at.scalar(scale)

            # roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            # SUB OUT FORWARD CALL
            h = self.extractor(img)
            _, _, rois, roi_indices, _ = self.rpn(h, img.shape[2:], scale)
            roi_cls_loc, roi_scores = self.head(h, rois, roi_indices)
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
            # raw_prob = at.tonumpy(prob)

            bbox = raw_cls_bbox.reshape((-1, self.n_class, 4))

            # Get the indexes of the best predicted bounding boxes for each gt_box                
            max_idxs = []
            for b, l in zip(gt_bbox, gt_label):
                gt_b = np.array([at.tonumpy(b)])
                gt_l = at.tonumpy(l)
                preds_bboxs_with_label = np.array(bbox)[:, gt_l]
                ious = bbox_iou(gt_b, preds_bboxs_with_label)
                max_idxs.append(ious.argmax())
            
            # Use the best predicted bbox predictions to select to corresponding roi
            closest_rois = rois[max_idxs]

            # Get the features using those rois
            _, _, feats = self.head(h, closest_rois, np.zeros(len(closest_rois)), return_features=True)

            features.append(at.tonumpy(feats))

        self.use_preset('evaluate')
        self.train()
        return features


    @nograd
    def _predict_gt_features_bbox_method(self, imgs, gt_bboxes, gt_labels, scales):
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
            img = at.totensor(img[None]).float()
            scale = at.scalar(scale)

            h = self.extractor(img)
            _, _, rois, roi_indices, _ = self.rpn(h, img.shape[2:], scale)
            rois = gt_bbox * scale
            roi_indices = np.zeros(len(rois))
            roi_cls_loc, roi_scores, feats = self.head(h, rois, roi_indices, return_features=True)

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

            raw_cls_bbox = at.tonumpy(cls_bbox).reshape((-1, self.n_class, 4))
            raw_prob = at.tonumpy(prob)
            raw_feats = at.tonumpy(feats)

            bboxes.append(at.totensor(np.array([b[l,:] for b, l in zip(raw_cls_bbox, gt_label)])))
            labels.append(at.totensor(raw_prob.argmax(axis=1).astype(np.int32)))
            scores.append(at.totensor(raw_prob.max(axis=1).astype(np.float32)))
            features.append(raw_feats)

        self.use_preset('evaluate')
        self.train()
        # import pdb; pdb.set_trace()
        
        return features
        # return bboxes, labels, scores


    @nograd
    def _predict_gt_features_naive_pred_iou_method(self, imgs, gt_bboxes, gt_labels, scales):
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

            bbox, label, score, feats = self._suppress_with_feats(raw_cls_bbox, raw_prob, raw_feats)

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
            

    @nograd
    def _predict_gt_features_training_funcs_method(self, imgs, gt_bboxes, gt_labels, scale):
        """Get the mahalanobis features needed to predict novelty score with predict_ood method."""
        self.eval()
        n = gt_bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = gt_bboxes[0]
        label = gt_labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        proposal_target_creator = ProposalTargetCreator()
        sample_roi, gt_roi_loc, gt_roi_label = proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score, feats = self.head(
            features,
            sample_roi,
            sample_roi_index, return_features=True)

        self.use_preset('evaluate')
        self.train()
        return feats


    def train_ood(self, dataloader, num_train=opt.train_num):
        features = []
        gt_labels = []

        print("extracting features")
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader), total=num_train):
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # features_ = self._predict_gt_features(img, bbox, label, scale)
            features_, labels_ = self._predict_gt_features_naive_pred_iou_method(img, bbox, label, scale)
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
        self.mahal_means = self._calc_mahal_means(features, gt_labels)
        
        print("calculating feature covariance")
        self.mahal_cov = self._calc_mahal_covariance_matrix(features, gt_labels)
        
        print("inverting feature covariance")
        self.inv_mahal_cov = np.linalg.inv(self.mahal_cov + np.eye(len(self.mahal_cov)) + 1e-18)

        return self.mahal_means, self.mahal_cov


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




