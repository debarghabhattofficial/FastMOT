from collections import defaultdict
from pathlib import Path
import configparser
import abc
import numpy as np
import numba as nb
import cupy as cp
import cupyx.scipy.ndimage
import cv2

from . import models
from .utils import TRTInference
from .utils.rect import as_tlbr, aspect_ratio, to_tlbr, get_size, area
from .utils.rect import enclosing, multi_crop, iom, diou_nms
from .utils.numba import find_split_indices

# Following code was written by DEB.
# ==================================================
import torch
from .yolov7.models.experimental import attempt_load
from .yolov7.utils.general import check_img_size
from .yolov7.utils.general import non_max_suppression
# ==================================================


DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


class Detector(abc.ABC):
    @abc.abstractmethod
    def __init__(self, size):
        self.size = size

    def __call__(self, frame):
        """Detect objects synchronously."""
        self.detect_async(frame)
        return self.postprocess()

    @abc.abstractmethod
    def detect_async(self, frame):
        raise NotImplementedError

    @abc.abstractmethod
    def postprocess(self):
        raise NotImplementedError


class SSDDetector(Detector):
    def __init__(self, size,
                 class_ids,
                 model='SSDInceptionV2',
                 tile_overlap=0.25,
                 tiling_grid=(4, 2),
                 conf_thresh=0.5,
                 merge_thresh=0.6,
                 max_area=120000):
        """An object detector for SSD models.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        class_ids : sequence
            Class IDs to detect. Note class ID starts at zero.
        model : str, optional
            SSD model to use.
            Must be the name of a class that inherits `models.SSD`.
        tile_overlap : float, optional
            Ratio of overlap to width and height of each tile.
        tiling_grid : tuple, optional
            Width and height of tile layout to split each frame for batch inference.
        conf_thresh : float, optional
            Detection confidence threshold.
        merge_thresh : float, optional
            Overlap threshold to merge bounding boxes across tiles.
        max_area : int, optional
            Max area of bounding boxes to detect.
        """
        super().__init__(size)
        self.model = models.SSD.get_model(model)
        assert 0 <= tile_overlap <= 1
        self.tile_overlap = tile_overlap
        assert tiling_grid[0] >= 1 and tiling_grid[1] >= 1
        self.tiling_grid = tiling_grid
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert 0 <= merge_thresh <= 1
        self.merge_thresh = merge_thresh
        assert max_area >= 0
        self.max_area = max_area

        self.label_mask = np.zeros(self.model.NUM_CLASSES, dtype=np.bool_)
        try:
            self.label_mask[tuple(class_ids),] = True
        except IndexError as err:
            raise ValueError('Unsupported class IDs') from err

        self.batch_size = int(np.prod(self.tiling_grid))
        self.tiles, self.tiling_region_sz = self._generate_tiles()
        self.scale_factor = tuple(np.array(self.size) / self.tiling_region_sz)
        self.backend = TRTInference(self.model, self.batch_size)
        self.inp_handle = self.backend.input.host.reshape(self.batch_size, *self.model.INPUT_SHAPE)

    def detect_async(self, frame):
        """Detects objects asynchronously."""
        self._preprocess(frame)
        self.backend.infer_async()

    def postprocess(self):
        """Synchronizes, applies postprocessing, and returns a record array
        of detections (DET_DTYPE).
        This API should be called after `detect_async`.
        Detections are sorted in ascending order by class ID.
        """
        det_out = self.backend.synchronize()[0]
        detections, tile_ids = self._filter_dets(det_out, self.tiles, self.model.TOPK,
                                                 self.label_mask, self.max_area,
                                                 self.conf_thresh, self.scale_factor)
        detections = self._merge_dets(detections, tile_ids)
        return detections

    def _preprocess(self, frame):
        frame = cv2.resize(frame, self.tiling_region_sz)
        self._normalize(frame, self.tiles, self.inp_handle)

    def _generate_tiles(self):
        tile_size = np.array(self.model.INPUT_SHAPE[:0:-1])
        tiling_grid = np.array(self.tiling_grid)
        step_size = (1 - self.tile_overlap) * tile_size
        total_size = (tiling_grid - 1) * step_size + tile_size
        total_size = np.rint(total_size).astype(int)
        tiles = np.array([to_tlbr((c * step_size[0], r * step_size[1], *tile_size))
                          for r in range(tiling_grid[1]) for c in range(tiling_grid[0])])
        return tiles, tuple(total_size)

    def _merge_dets(self, detections, tile_ids):
        detections = np.fromiter(detections, DET_DTYPE, len(detections)).view(np.recarray)
        tile_ids = np.fromiter(tile_ids, int, len(tile_ids))
        if len(detections) == 0:
            return detections
        detections = self._merge(detections, tile_ids, self.batch_size, self.merge_thresh)
        return detections.view(np.recarray)

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _normalize(frame, tiles, out):
        imgs = multi_crop(frame, tiles)
        for i in nb.prange(len(imgs)):
            bgr = imgs[i]
            # BGR to RGB
            rgb = bgr[..., ::-1]
            # HWC -> CHW
            chw = rgb.transpose(2, 0, 1)
            # Normalize to [-1.0, 1.0] interval
            out[i] = chw * (2 / 255.) - 1.

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _filter_dets(det_out, tiles, topk, label_mask, max_area, thresh, scale_factor):
        detections = []
        tile_ids = []
        for tile_idx in range(len(tiles)):
            tile = tiles[tile_idx]
            w, h = get_size(tile)
            tile_offset = tile_idx * topk
            for det_idx in range(topk):
                offset = (tile_offset + det_idx) * 7
                label = int(det_out[offset + 1])
                conf = det_out[offset + 2]
                if conf < thresh:
                    break
                if label_mask[label]:
                    xmin = (det_out[offset + 3] * w + tile[0]) * scale_factor[0]
                    ymin = (det_out[offset + 4] * h + tile[1]) * scale_factor[1]
                    xmax = (det_out[offset + 5] * w + tile[0]) * scale_factor[0]
                    ymax = (det_out[offset + 6] * h + tile[1]) * scale_factor[1]
                    tlbr = as_tlbr((xmin, ymin, xmax, ymax))
                    if 0 < area(tlbr) <= max_area:
                        detections.append((tlbr, label, conf))
                        tile_ids.append(tile_idx)
        return detections, tile_ids

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _merge(dets, tile_ids, num_tile, thresh):
        # find duplicate neighbors across tiles
        neighbors = [[0 for _ in range(0)] for _ in range(len(dets))]
        for i, det in enumerate(dets):
            max_ioms = np.zeros(num_tile)
            for j, other in enumerate(dets):
                if tile_ids[i] != tile_ids[j] and det.label == other.label:
                    overlap = iom(det.tlbr, other.tlbr)
                    # use the detection with the greatest IoM from each tile
                    if overlap >= thresh and overlap > max_ioms[tile_ids[j]]:
                        max_ioms[tile_ids[j]] = overlap
                        neighbors[i].append(j)

        # merge neighbors using depth-first search
        keep = set(range(len(dets)))
        stack = []
        for i in range(len(dets)):
            if len(neighbors[i]) > 0 and tile_ids[i] != -1:
                tile_ids[i] = -1
                stack.append(i)
                candidates = []
                while len(stack) > 0:
                    for j in neighbors[stack.pop()]:
                        if tile_ids[j] != -1:
                            candidates.append(j)
                            tile_ids[j] = -1
                            stack.append(j)
                for k in candidates:
                    dets[i].tlbr[:] = enclosing(dets[i].tlbr, dets[k].tlbr)
                    dets[i].conf = max(dets[i].conf, dets[k].conf)
                    keep.discard(k)
        dets = dets[np.array(list(keep))]

        # sort detections by class
        dets = dets[np.argsort(dets.label)]
        return dets


class YOLODetector(Detector):
    def __init__(self, 
                 size,
                 class_ids,
                 model='YOLOv4',
                 conf_thresh=0.25,
                 nms_thresh=0.5,
                 max_area=800000,
                 min_aspect_ratio=1.2):
        """An object detector for YOLO models.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        class_ids : sequence
            Class IDs to detect. Note class ID starts at zero.
        model : str, optional
            YOLO model to use.
            Must be the name of a class that inherits `models.YOLO`.
        conf_thresh : float, optional
            Detection confidence threshold.
        nms_thresh : float, optional
            Nonmaximum suppression overlap threshold.
            Set higher to detect crowded objects.
        max_area : int, optional
            Max area of bounding boxes to detect.
        min_aspect_ratio : float, optional
            Min aspect ratio (height over width) of bounding boxes to detect.
            Set to 0.1 for square shaped objects.
        """
        super().__init__(size)
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert 0 <= nms_thresh <= 1
        self.nms_thresh = nms_thresh
        assert max_area >= 0
        self.max_area = max_area
        assert min_aspect_ratio >= 0
        self.min_aspect_ratio = min_aspect_ratio

        self.label_mask = np.zeros(self.model.NUM_CLASSES, dtype=np.bool_)
        try:
            self.label_mask[tuple(class_ids),] = True
        except IndexError as err:
            raise ValueError('Unsupported class IDs') from err

        self.backend = TRTInference(self.model, 1)
        self.inp_handle, self.upscaled_sz, self.bbox_offset = self._create_letterbox()

    def detect_async(self, frame):
        """Detects objects asynchronously."""
        self._preprocess(frame)
        self.backend.infer_async(from_device=True)

    def postprocess(self):
        """Synchronizes, applies postprocessing, and returns a record array
        of detections (DET_DTYPE).
        This API should be called after `detect_async`.
        Detections are sorted in ascending order by class ID.
        """
        det_out = self.backend.synchronize()
        det_out = np.concatenate(det_out).reshape(-1, 7)
        detections = self._filter_dets(det_out, self.upscaled_sz, self.bbox_offset,
                                       self.label_mask, self.conf_thresh, self.nms_thresh,
                                       self.max_area, self.min_aspect_ratio)
        detections = np.fromiter(detections, DET_DTYPE, len(detections)).view(np.recarray)
        return detections

    def _preprocess(self, frame):
        zoom = np.roll(self.inp_handle.shape, -1) / frame.shape
        with self.backend.stream:
            frame_dev = cp.asarray(frame)
            # resize
            small_dev = cupyx.scipy.ndimage.zoom(frame_dev, zoom, order=1, mode='opencv', grid_mode=True)
            # BGR to RGB
            rgb_dev = small_dev[..., ::-1]
            # HWC -> CHW
            chw_dev = rgb_dev.transpose(2, 0, 1)
            # normalize to [0, 1] interval
            cp.multiply(chw_dev, 1 / 255., out=self.inp_handle)

    def _create_letterbox(self):
        src_size = np.array(self.size)
        dst_size = np.array(self.model.INPUT_SHAPE[:0:-1])
        if self.model.LETTERBOX:
            scale_factor = min(dst_size / src_size)
            scaled_size = np.rint(src_size * scale_factor).astype(int)
            img_offset = (dst_size - scaled_size) / 2
            roi = np.s_[:, img_offset[1]:img_offset[1] + scaled_size[1],
                        img_offset[0]:img_offset[0] + scaled_size[0]]
            upscaled_sz = np.rint(dst_size / scale_factor).astype(int)
            bbox_offset = (upscaled_sz - src_size) / 2
        else:
            roi = np.s_[:]
            upscaled_sz = src_size
            bbox_offset = np.zeros(2)
        inp_reshaped = self.backend.input.device.reshape(self.model.INPUT_SHAPE)
        inp_reshaped[:] = 0.5 # initial value for letterbox
        inp_handle = inp_reshaped[roi]
        return inp_handle, upscaled_sz, bbox_offset

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _filter_dets(det_out, size, offset, label_mask, conf_thresh, nms_thresh, max_area, min_ar):
        """
        det_out: a list of 3 tensors, where each tensor
                 contains a multiple of 7 float32 numbers in
                 the order of [x, y, w, h, box_confidence, class_id, class_prob]
        """
        # filter by class and score
        keep = []
        for i in range(len(det_out)):
            if label_mask[int(det_out[i, 5])]:
                score = det_out[i, 4] * det_out[i, 6]
                if score >= conf_thresh:
                    keep.append(i)
        det_out = det_out[np.array(keep)]

        # scale to pixel values
        det_out[:, :4] *= np.append(size, size)
        det_out[:, :2] -= offset

        # per-class NMS
        det_out = det_out[np.argsort(det_out[:, 5])]
        split_indices = find_split_indices(det_out[:, 5])
        all_indices = np.arange(len(det_out))

        keep = []
        for i in range(len(split_indices) + 1):
            begin = 0 if i == 0 else split_indices[i - 1]
            end = len(det_out) if i == len(split_indices) else split_indices[i]
            cls_dets = det_out[begin:end]
            cls_keep = diou_nms(cls_dets[:, :4], cls_dets[:, 4], nms_thresh)
            keep.extend(all_indices[begin:end][cls_keep])
        nms_dets = det_out[np.array(keep)]

        # create detections
        detections = []
        for i in range(len(nms_dets)):
            tlbr = to_tlbr(nms_dets[i, :4])
            label = int(nms_dets[i, 5])
            conf = nms_dets[i, 4] * nms_dets[i, 6]
            if 0 < area(tlbr) <= max_area and aspect_ratio(tlbr) >= min_ar:
                detections.append((tlbr, label, conf))
        return detections


class PublicDetector(Detector):
    def __init__(self, size,
                 class_ids,
                 frame_skip,
                 sequence_path=None,
                 conf_thresh=0.5,
                 max_area=800000):
        """Class to use MOT Challenge's public detections.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        class_ids : sequence
            Class IDs to detect. Only 1 (i.e. person) is supported.
        frame_skip : int
            Detector frame skip.
        sequence_path : str, optional
            Relative path to MOT Challenge's sequence directory.
        conf_thresh : float, optional
            Detection confidence threshold.
        max_area : int, optional
            Max area of bounding boxes to detect.
        """
        super().__init__(size)
        assert tuple(class_ids) == (1,)
        self.frame_skip = frame_skip
        assert sequence_path is not None
        self.seq_root = Path(__file__).parents[1] / sequence_path
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert max_area >= 0
        self.max_area = max_area

        assert self.seq_root.exists()
        seqinfo = configparser.ConfigParser()
        seqinfo.read(self.seq_root / 'seqinfo.ini')
        self.seq_size = (int(seqinfo['Sequence']['imWidth']), int(seqinfo['Sequence']['imHeight']))

        self.detections = defaultdict(list)
        self.frame_id = 0

        det_txt = self.seq_root / 'det' / 'det.txt'
        for mot_challenge_det in np.loadtxt(det_txt, delimiter=','):
            frame_id = int(mot_challenge_det[0]) - 1
            tlbr = to_tlbr(mot_challenge_det[2:6])
            # mot_challenge_det[6]
            conf = 1.0
            # mot_challenge_det[7]
            label = 1 # person
            # scale inside frame
            tlbr[:2] = tlbr[:2] / self.seq_size * self.size
            tlbr[2:] = tlbr[2:] / self.seq_size * self.size
            tlbr = np.rint(tlbr)
            if conf >= self.conf_thresh and area(tlbr) <= self.max_area:
                self.detections[frame_id].append((tlbr, label, conf))

    def detect_async(self, frame):
        pass

    def postprocess(self):
        detections = np.array(self.detections[self.frame_id], DET_DTYPE).view(np.recarray)
        self.frame_id += self.frame_skip
        return detections


class YOLOv7Detector(Detector):
    def __init__(self,
                 size,
                 class_ids=(0,),
                 ckpt_path=None,
                 conf_thresh=0.25,
                 nms_thresh=0.5,
                 max_area=800_000,
                 min_aspect_ratio=1.2):
        """
        An YOLOv7 object detector model.
        """
        super().__init__(size)
        assert 0 <= conf_thresh <= 1
        self.conf_thresh = conf_thresh
        assert 0 <= nms_thresh <= 1
        self.nms_thresh = nms_thresh
        assert max_area >= 0
        self.max_area = max_area
        assert min_aspect_ratio >= 0
        self.min_aspect_ratio = min_aspect_ratio
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = attempt_load(
            weights=ckpt_path, 
            map_location=self.device
        )
        self.model.eval()

    def _create_letterbox(self,
                          img,
                          new_shape=(640, 640),  # (960, 540)
                          color=(114, 114, 114),
                          auto=True,
                          scale_up=True,
                          stride=32):
        """
        This methods resizes and pads the image while
        meeting stride-multiple constraints. This is
        necessary to prevent error while running detection
        inference through the YOLOv7 model.
        """
        # Reisize the image.
        # Image original shape (heights, width)
        src_shape = img.shape[:2]
        # Image new height and width.
        dst_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
        
        # Calculate the scale ratio based on src_shape
        # and dest_shape
        # NOTE: We find the scaling ration ratio along
        # each dimension (height, width) by diving
        # dst_shape value by src_shape value, and select 
        # the minimum of the two.
        scale_ratio = min(
            dst_shape[0] / src_shape[0],
            dst_shape[1] / src_shape[1]
        )
        # If scale_up flag is set to False, we only scale
        # down the image and not scale up, which improves
        # the validation mAP.
        if not scale_up:
            scale_ratio = min(scale_ration, 1.0)

        # Compute padding along each dimesion
        # (height and width).
        new_unpad = (
            int(round(src_shape[1] * scale_ratio)),
            int(round(src_shape[0] * scale_ratio))
        )
        dh = dst_shape[0] - new_unpad[1]  # Height padding.
        dw = dst_shape[1] - new_unpad[0]  # Width padding.
        if auto:
            dw = np.mod(dw, stride)
            dh = np.mod(dh, stride)

        # Divide the padding into two sides.
        dw = dw / 2
        dh = dh / 2

        # Resize the image.
        resized_img = img.copy()
        if src_shape[::-1] != new_unpad:
            resized_img = cv2.resize(
                img, 
                new_unpad, 
                interpolation=cv2.INTER_AREA
            )

        # Add border.
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        resized_img = cv2.copyMakeBorder(
            resized_img, 
            top, bottom, 
            left, right, 
            cv2.BORDER_CONSTANT, 
            value=color
        )

        return resized_img, scale_ratio, (dw, dh)

    def _preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, scale_ratio, dwdh = self._create_letterbox(
            img=frame,
            auto=False
        )
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).float()
        frame = frame.permute(2, 0, 1)
        frame = frame.unsqueeze(0)
        frame = frame.to(self.device)
        frame = frame.contiguous()
        return frame, scale_ratio, dwdh

    def _filter_dets(self, 
                     detections, 
                     conf_thresh, 
                     nms_thresh,
                     scale_ratio,
                     dwdh,
                     class_id=0,
                     replacement_id=1):
        """
        This method applies non max suppression to filter 
        out the bounding boxes that have low confidence 
        score, or do not belong to the required object class.

        INPUT:
        detections: A list tensors, where each tensor 
            contains a multiple of 7 float32 numbers 
            in the order of [
                x, y, w, h, 
                box_confidence, 
                class_id, 
                class_prob
            ].

        OUTPUT:
        detections: A list of numpy record arrays where each 
            each record array has the ollowing attributes:
            (tlbr, label, conf).
        """
        # Apply non max suppression.
        detections = non_max_suppression(
            prediction=detections, 
            conf_thres=conf_thresh, 
            iou_thres=nms_thresh, 
            classes=[class_id], 
            agnostic=True
        )

        # Create list of numpy record arrays.
        if detections[0].shape[0] > 0:
            detections = [
                (
                    tuple(
                        self.postprocess(
                            boxes=detection[0, :4],
                            scale_ratio=scale_ratio,
                            dwdh=dwdh
                        ).cpu().numpy()
                    ),
                    replacement_id if int(detection[0, 5].item()) == class_id else int(detection[0, 5].item()),
                    detection[0, 4].item()
                )
                for detection in detections
            ]
            detections = np.array(detections, dtype=DET_DTYPE)
            detections = detections.view(np.recarray)
        else:
            detections = np.empty((0, ), dtype=DET_DTYPE)
            detections = detections.view(np.recarray)

        return detections

    def postprocess(self,
                    boxes,
                    scale_ratio,
                    dwdh):
        """
        This method converts the bounding box
        coordinates from the Yolov7 detector
        to the original image coordinates.
        """
        dwdh = torch.tensor(dwdh * 2).to(boxes.device)
        boxes = boxes - dwdh
        boxes = boxes / scale_ratio
        return boxes.clip_(0, 6400)

    def detect_async(self, frame):
        """Detects objects asynchronously."""
        # Preprocess the image frame before passing
        # it to the detector.
        pp_frame, scale_ratio, \
            dwdh =  self._preprocess(frame=frame)
        
        # Get the detections from the detector.
        detections = None
        with torch.no_grad():
            detections = self.model(pp_frame)[0]
        
        # Filter the detections.
        detections = self._filter_dets(
            detections=det_out,
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            scale_ratio=scale_ratio,
            dwdh=dwdh
        )

        return detections

    def __call__(self, frame):
        """Detect objects synchronously."""
        return self.detect_async(frame)