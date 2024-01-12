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


def print_model_attributes(model, prefix=''):
    for name, param in model.named_parameters():
        print(f"{prefix}Parameter: {name}, Size: {param.size()}")
        
    for name, buffer in model.named_buffers():
        print(f"{prefix}Buffer: {name}, Size: {buffer.size()}")
        
    for name, module in model.named_children():
        print_model_attributes(module, prefix=f"{prefix}{name}.")


def preprocess_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (384,384))
    image = image.astype(np.float32) / 255.
    image = (image - mean) / std
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    image = image.cuda()
    image = image.contiguous()
    return image

DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)  # Copied from fastmot/detector.py.

# def convert_to_structured_array(preds):
#     """
#     This method converts the predictions
#     from the Yolov7 detector to a structured
#     array.
#     """
#     # Extract values from the predicted
#     # tensors and create a structured array.
#     joy = [
#         tuple(
#             pred_tensor[0, :4].cpu().numpy()
#         ) + (
#             int(pred_tensor[0, 4].item()), 
#             pred_tensor[0, 5].item()
#         )
#         for pred_tensor in preds
#     ]
    
#     return preds
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
    img_size = (640, 640)
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
        print(f"names: {names}")
        print("-" * 75)
        # Get number of classes for the model.
        num_classes =  model.nc 
        print(f"num_classes: {num_classes}")
        print("-" * 75)
        # Get model stride.
        stride = model.stride.max().cpu().numpy()
        print(f"stride: {stride}")
        print("-" * 75)
        # Check img size.
        img_size = check_img_size(img_size[0], s=stride)
        print(f"img_size: {img_size}")
        print("-" * 75)
        # =============================================

        # Original code was not commented.
        # =============================================
        mot = fastmot.MOT(
            config.resize_to, 
            **vars(config.mot_cfg), 
            draw=draw
        )
        mot.reset(stream.cap_dt)
        # quit()  # DEB
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
                    # print(f"frame.shape: {frame.shape}")  # DEB
                    # print("-" * 75)  # DEB
                    frame_img =  preprocess_image(image=frame)  # DEB
                    # print(f"frame_img shape: {frame_img.shape}")  # DEB
                    # print("-" * 75)  # DEB
                    preds = None
                    with torch.no_grad():
                        preds = model(frame_img)[0]  # DEB
                    # print(f"preds type: {type(preds)}")  # DEB
                    # print(f"preds shape: {preds.shape}")  # DEB
                    # print("-" * 75)  # DEB

                    conf_thresh = config.mot_cfg.yolo_detector_cfg.conf_thresh  # DEB
                    # print(f"conf_thresh: {conf_thresh}")  # DEB
                    # print("-" * 75)  # DEB
                    nms_thresh = config.mot_cfg.yolo_detector_cfg.nms_thresh  # DEB
                    # print(f"nms_thresh: {nms_thresh}")  # DEB
                    # print("-" * 75)  # DEB

                    preds = non_max_suppression(
                        prediction=preds, 
                        conf_thres=conf_thresh, 
                        iou_thres=nms_thresh, 
                        classes=[0], 
                        agnostic=True
                    ) # DEB
                    # print(f"preds type: {type(preds)}")  # DEB
                    # print(f"preds shape: {len(preds)}")  # DEB
                    # print(f"preds:")  # DEB
                    # pprint(preds)  # DEB
                    # print("-" * 75)  # DEB

                    # Extract values from the predicted
                    # tensors and create a structured array.
                    preds = [
                        (
                            tuple(pred_tensor[0, :4].cpu().numpy()),
                            int(pred_tensor[0, 4].item()), 
                            pred_tensor[0, 5].item()
                        )
                        for pred_tensor in preds
                    ]
                    preds = np.array(preds, dtype=DET_DTYPE)  # DEB
                    preds = preds.view(np.recarray)
                    # print(f"preds type: {type(preds)}")  # DEB
                    # print(f"preds shape: {preds.shape}")  # DEB
                    # print(f"preds:")  # DEB
                    # pprint(preds)  # DEB
                    # print("-" * 75)  # DEB

                    # Pass the structured detections to
                    # the next stage of the MOT pipeline.
                    mot.step(frame=frame, detections_v7=preds)  # DEB
                    if txt is not None:
                        for track in mot.visible_tracks():
                            tl = track.tlbr[:2] / config.resize_to * stream.resolution
                            br = track.tlbr[2:] / config.resize_to * stream.resolution
                            w, h = br - tl + 1
                            txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                                      f'{w:.6f},{h:.6f},-1,-1,-1\n')
                    # quit()  # DEB
                    # continue  # DEB
                    # =============================================
                    # Original code was not commented.
                    # =============================================
                    # mot.step(frame)
                    # if txt is not None:
                    #     for track in mot.visible_tracks():
                    #         tl = track.tlbr[:2] / config.resize_to * stream.resolution
                    #         br = track.tlbr[2:] / config.resize_to * stream.resolution
                    #         w, h = br - tl + 1
                    #         txt.write(f'{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
                    #                   f'{w:.6f},{h:.6f},-1,-1,-1\n')
                    # =============================================

                if args.show:
                    cv2.imshow('Video', frame)
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
