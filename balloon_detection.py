from detectron2.structures import BoxMode
import random
import os
import json
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import numpy as np
import tqdm
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        img_anns = json.load(f)
        # json format is kye:value
    dataset_dicts = []
    for idx, v in enumerate(img_anns.valuse()):
        record = {}

        filename = os.path.join(img_dir, v['filename'])
        height, width = cv2.imread(filename).shape[:2]

        record['file_name'] = filename
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x+0.5, y+0.5) for x, y in zip(px,py)]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox" :[np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode" : BoxMode.XYXY_ABS,
                "segmentation" : [poly],
                "category_id" :0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_"+d, lambda d= d:get_balloon_dicts("balloon/"+d))
    # add balloon to default classes
    MetadataCatalog.get("balloon_" +d).set(thing_classes=["balloon"])

balloon_metadata = MetadataCatalog.get("balloon_train")
dataset_dicts = get_balloon_dicts("balloon/train")
for d in random.sample(dataset_dicts,3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:,:,::-1], metadata=balloon_metadata, scale =0.5)
    out = visualizer.draw_instance_dict(d)
    cv2.imshow("input image",out.get_image())

# set up the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST=()
cfg.MODEL.DEVICE = 'gpu'
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH=2
cfg.SOLVER.BASE_LR=0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # only has one class to be detected
os.makdirs(cfg.OUTPUT_DIR, exist_ok = True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# eval the model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)
dataset_dicts = get_balloon_dicts('balloon/val')
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1], metadata=balloon_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("predict", out.get_image())





