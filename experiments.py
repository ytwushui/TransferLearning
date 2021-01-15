import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from predictor import VisualizationDemo
im = cv2.imread('./datasets/streetview.jpg')
img_rgb = cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)
Image.fromarray(im)
#cv2.imshow('stview', img_rgb)
#cv2.waitKey(0)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
#cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

test_image_dir = "./datasets/randomdetection/"
for image_name in glob.glob(test_image_dir+"*.jpg"):
    test_image = cv2.imread(image_name)
    outputs = predictor(test_image)
#print(outputs["instances"].pred_classes)
#print(outputs['instances'].pred_boxes)
# all the catagries
#MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    v = Visualizer(test_image[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    img_rgb = cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB)
    cv2.imwrite("name.png", img_rgb)
    cv2.imshow('out', img_rgb)
    cv2.waitKey(0)