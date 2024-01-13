#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import argparse
import logging
import json
import cv2

import numpy as np

import fastmot
import fastmot.models
from fastmot.utils import ConfigDecoder, Profiler

# Import statements to use Yolov7 detector.
# [Added by DEB.]
# TODO: Currently, trying to execute the FastMOT code using
# pytorch files and not their TensorRT counterparts. Later,
# change the entire code to use TensorRT files.
# ============================================='
from pprint import pprint

import torch
from fastmot.yolov7.models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, 
    scale_coords, check_requirements,
    check_imshow, xyxy2xywh, increment_path, 
    strip_optimizer, colorstr, check_file
)


IMG_SIZE = (640, 640)  # (1920, 1080)  # (640, 640)  # (960, 540)


DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)  # Copied from fastmot/detector.py.


def letterbox(img,
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


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1, scale_ratio, dwdh = letterbox(
        img=image,
        auto=False
    )
    image1 = image1.astype(np.float32) / 255.0
    image1 = torch.from_numpy(image1).float()
    image1 = image1.permute(2, 0, 1)
    image1 = image1.unsqueeze(0)
    image1 = image1.cuda()
    image1 = image1.contiguous()
    return image1, scale_ratio, dwdh


def postprocess(boxes,
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
# =============================================


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    group = parser.add_mutually_exclusive_group()
    required.add_argument('-i', '--input-uri', metavar="URI", required=True, help=
                          'URI to input stream\n'
                          '1) image sequence (e.g. %%06d.jpg)\n'
                          '2) video file (e.g. file.mp4)\n'
                          '3) MIPI CSI camera (e.g. csi://0)\n'
                          '4) USB camera (e.g. /dev/video0)\n'
                          '5) RTSP stream (e.g. rtsp://<user>:<password>@<ip>:<port>/<path>)\n'
                          '6) HTTP stream (e.g. http://<user>:<password>@<ip>:<port>/<path>)\n')
    optional.add_argument('-c', '--config', metavar="FILE",
                          default=Path(__file__).parent / 'cfg' / 'mot.json',
                          help='path to JSON configuration file')
    optional.add_argument('-l', '--labels', metavar="FILE",
                          help='path to label names (e.g. coco.names)')
    optional.add_argument('-o', '--output-uri', metavar="URI",
                          help='URI to output video file')
    optional.add_argument('-t', '--txt', metavar="FILE",
                          help='path to output MOT Challenge format results (e.g. MOT20-01.txt)')
    optional.add_argument('-m', '--mot', action='store_true', help='run multiple object tracker')
    optional.add_argument('-s', '--show', action='store_true', help='show visualizations')
    group.add_argument('-q', '--quiet', action='store_true', help='reduce output verbosity')
    group.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    parser._action_groups.append(optional)
    args = parser.parse_args()
    if args.txt is not None and not args.mot:
        raise parser.error('argument -t/--txt: not allowed without argument -m/--mot')

    # Set up logging.
    logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(fastmot.__name__)
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Load config file.
    with open(args.config) as cfg_file:
        config = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))

    # Load labels if given.
    if args.labels is not None:
        with open(args.labels) as label_file:
            label_map = label_file.read().splitlines()
            print(f"label_map: {label_map}")  # DEB
            print("-" * 75)  # DEB
            fastmot.models.set_label_map(label_map)

    stream = fastmot.VideoIO(
        size=config.resize_to, 
        input_uri=args.input_uri, 
        output_uri=args.output_uri, 
        **vars(config.stream_cfg)
    )

    # Following code is written by DEB.
    # TODO: Read this from command line and do
    # not hardcode.
    # =============================================
    # Check if GPU is available and set the device accordingly
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    img_size = IMG_SIZE  # (1920, 1080)  # (640, 640)
    # =============================================

    mot = None
    txt = None
    if args.mot:
        draw = args.show or args.output_uri is not None

        # Following code written by DEB.
        # =============================================
        # Load the Yolov7 model.
        yolov7_ckpt_path = "fastmot/models/yolov7/yolov7.pt"
        yolov7_ckpt_path = "fastmot/models/yolov7/yolo_v7_3_1.pt"
        model = attempt_load(
            weights=yolov7_ckpt_path, 
            map_location=device
        )
        model.eval()
        names = model.names
        # Get number of classes for the model.
        num_classes =  model.nc
        # Get model stride.
        stride = model.stride.max().cpu().numpy()
        # Check img size.
        img_size = check_img_size(img_size[0], s=stride)
        # =============================================

        # Original code was not commented.
        # =============================================
        mot = fastmot.MOT(
            config.resize_to, 
            **vars(config.mot_cfg), 
            draw=draw
        )
        mot.reset(stream.cap_dt)
        # =============================================
    if args.txt is not None:
        Path(args.txt).parent.mkdir(
            parents=True, exist_ok=True
        )
        txt = open(args.txt, 'w')
    if args.show:
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)

    logger.info('Starting video capture...')
    stream.start_capture()
    try:
        with Profiler('app') as prof:
            while not args.show or cv2.getWindowProperty('Video', 0) >= 0:
                frame = stream.read()
                if frame is None:
                    break

                if args.mot:
                    # Following code written by DEB.
                    # =============================================
                    frame_img, scale_ratio, dwdh =  preprocess_image(image=frame)

                    preds = None
                    with torch.no_grad():
                        preds = model(frame_img)[0]

                    conf_thresh = config.mot_cfg.yolo_detector_cfg.conf_thresh
                    nms_thresh = config.mot_cfg.yolo_detector_cfg.nms_thresh

                    preds = non_max_suppression(
                        prediction=preds, 
                        conf_thres=conf_thresh, 
                        iou_thres=nms_thresh, 
                        classes=[0], 
                        agnostic=True
                    )

                    # Extract values from the predicted
                    # tensors and create a structured array.
                    if preds[0].shape[0] > 0:
                        preds = [
                            (
                                tuple(
                                    postprocess(
                                        boxes=pred_tensor[0, :4],
                                        scale_ratio=scale_ratio,
                                        dwdh=dwdh
                                    ).cpu().numpy()
                                ),
                                1 if int(pred_tensor[0, 5].item()) == 0 else 0, 
                                pred_tensor[0, 4].item()
                            )
                            for pred_tensor in preds
                        ]
                        preds = np.array(preds, dtype=DET_DTYPE)
                        preds = preds.view(np.recarray)

                        # Pass the structured detections to
                        # the next stage of the MOT pipeline.
                        mot.step(frame=frame, detections_v7=preds)
                    if txt is not None:
                        for track in mot.visible_tracks():
                            tl = track.tlbr[:2] / config.resize_to * stream.resolution
                            br = track.tlbr[2:] / config.resize_to * stream.resolution
                            w, h = br - tl + 1
                            txt.write(f"{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},"
                                      f"{w:.6f},{h:.6f},-1,-1,-1\n")
                if args.show:
                    cv2.imshow("Video", frame)
                    if cv2.waitKey(100) & 0xFF == 27:
                        break
                if args.output_uri is not None:
                    stream.write(frame)
    finally:
        # clean up resources
        if txt is not None:
            txt.close()
        stream.release()
        cv2.destroyAllWindows()

    # timing statistics
    if args.mot:
        avg_fps = round(mot.frame_count / prof.duration)
        logger.info('Average FPS: %d', avg_fps)
        mot.print_timing_info()


if __name__ == '__main__':
    main()
