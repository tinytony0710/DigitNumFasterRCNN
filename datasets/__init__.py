# @title # datasets
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from os import listdir, path
import torchvision.transforms.functional as F
import cv2
import numpy as np
import random
from utils.data_io import load_json, load_image


def reinforce_image(image, scale=1.0):
    
    # get gaussian blur, then minus it from original
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.addWeighted(image, 1.0+scale, blur, -scale, 0)
    
    return image

# --- 1. 訓練/驗證資料集類別 (DigitDataset) ---
class DigitDataset(Dataset):
    def __init__(self, json_path, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        print(f"正在載入標註文件: {json_path}")
        self.coco_data = load_json(json_path)

        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        self.image_id_to_info = {image['id']: image for image in self.images}
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            # 檢查 bbox 格式和有效性
            if not isinstance(ann.get('bbox'), list) or len(ann['bbox']) != 4:
                print(f"警告: 圖像 {image_id}, 標註 {ann.get('id', 'N/A')} bbox 格式錯誤: {ann.get('bbox')}. 跳過.")
                continue
            xmin, ymin, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                print(f"警告: 圖像 {image_id}, 標註 {ann.get('id', 'N/A')} bbox 尺寸無效 (w={w}, h={h}). 跳過.")
                continue
            self.image_id_to_annotations[image_id].append(ann)

        # category_id 1-10 -> label 0-9
        self.category_map = {i+1: i for i in range(10)}

        self.valid_image_indices = []
        valid_image_ids = set()
        for idx, image in enumerate(self.images):
            image_id = image['id']
            # 確保圖像有標註且標註列表不為空
            if image_id in self.image_id_to_annotations and self.image_id_to_annotations[image_id]:
                 self.valid_image_indices.append(idx)
                 valid_image_ids.add(image_id)

        original_image_count = len(self.images)
        loaded_image_count = len(self.valid_image_indices)
        print(f"載入了 {loaded_image_count} 張圖像 ({json_path}).")
        if original_image_count > loaded_image_count:
             print(f"注意: {original_image_count - loaded_image_count} 張圖像因無標註或標註無效而被過濾。")


    def __len__(self):
        return len(self.valid_image_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_image_indices[idx]
        image_info = self.images[actual_idx]
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = path.join(self.image_dir, file_name)

        image = load_image(image_path)

        # choose 1 out of 3
        # if random.random() < 0.33:
        #     image = reinforce_image(image, scale=3.0)


        annotations = self.image_id_to_annotations.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            # 再次檢查 bbox 有效性 (雖然初始化時已檢查)
            xmin, ymin, w, h = ann['bbox']
            if w <= 0 or h <= 0: continue

            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])

            category_id = ann['category_id']
            labels.append(category_id)

            areas.append(ann.get('area', w * h)) # 如果 JSON 中沒有 area，則計算
            iscrowd.append(ann.get('iscrowd', 0))

        # 轉換為 Tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.empty((0,), dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.empty((0,), dtype=torch.int64)

        # 再次檢查邊界框在 Tensor 層面是否有效
        if boxes.shape[0] > 0:
            valid_boxes_idx = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if not torch.all(valid_boxes_idx):
                 print(f"警告: 圖像 {image_id} 中 Tensor 轉換後發現無效邊界框，已過濾。")
                 boxes = boxes[valid_boxes_idx]
                 labels = labels[valid_boxes_idx]
                 areas = areas[valid_boxes_idx]
                 iscrowd = iscrowd[valid_boxes_idx]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms:
            image, target = self.transforms(image, target)

        # 再次檢查 target['boxes'] 是否為空，如果為空則此樣本無效
        if target["boxes"].shape[0] == 0:
             print(f"警告: 圖像 {image_id} 處理後沒有有效的標註框，將跳過此樣本。")
             # 返回 None 以便 collate_fn 過濾
             # return None, None # 如果 collate_fn 能處理，這是選項之一
             # 或者返回一個帶有空 box 的有效 target，模型內部需要處理
             # 這裡保持原樣，但訓練循環需要更健壯地處理潛在的空 target

        return image, target

    def coco_eval(self, coco_results):
        coco_gt = COCO()
        coco_gt.dataset = self.coco_data
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval

# --- 1b. 測試資料集類別 (TestDigitDataset) ---
class TestDigitDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = [f for f in listdir(image_dir)]
        print(f"找到 {len(self.image_files)} 張測試圖像在目錄: {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx >= len(self.image_files):
             # Handle potential index out of bounds, though DataLoader shouldn't cause this normally
             return None, None

        file_name = self.image_files[idx]
        image_path = path.join(self.image_dir, file_name)

        try:
            image_id = int(path.splitext(file_name)[0])
        except ValueError:
            print(f"警告: 無法從測試文件名 '{file_name}' 提取數字 ID。將使用索引 {idx} 作為 ID。")
            image_id = idx

        image = load_image(image_path)
        # image = reinforce_image(image, scale=3.0)

        if self.transforms:
            image = self.transforms(image, target=None) # 測試時 target 為 None

        return image, image_id