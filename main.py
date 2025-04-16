# @title # main
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import json
import csv
import time

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from PIL import Image
from torch import optim
from os import listdir, path
import torchvision.transforms.functional as F
import torch
from torch import optim
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from tqdm import tqdm
import torch.nn as nn

from datasets import DigitDataset, TestDigitDataset
from utils.data_io import load_image, save_csv, save_json
from argparse import ArgumentParser

class ToTensor:
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is None:
            return image
        else:
            return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target=None):
        for t in self.transforms:
            # 假設轉換函數能處理 (image) 或 (image, target)
            if hasattr(t, '__call__'):
                 try:
                     if target is None:
                         image = t(image)
                     else:
                         image, target = t(image, target)
                 except TypeError: # 如果轉換函數只接受圖像
                     if target is not None:
                         image = t(image)
                     else:
                          image = t(image) # TestDataset
            else:
                 # Handle cases if t is not callable, though unlikely for transform lists
                 pass
        if target is None:
            return image
        else:
            return image, target

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    # if train:
        # 在這裡添加訓練時的數據增強 (例如 RandomHorizontalFlip)
        # 注意：確保增強方法能正確處理 target 字典中的 bounding box
        # pass
    return Compose(transforms)

# --- 4. 工具函數 (Collate Fn) ---
def collate_fn(batch):
    # 過濾掉 Dataset 中返回 None 的無效樣本
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None # 如果整個批次都無效

    # 檢查返回的是 (image, target dict) 還是 (image, image_id)
    if isinstance(batch[0][1], dict): # Training/Validation
        # 過濾掉 target['boxes'] 為空的樣本 (如果有的話)
        # batch = list(filter(lambda x: x[1]["boxes"].shape[0] > 0, batch))
        # if not batch: return None, None # 如果過濾後批次為空
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets
    else: # Testing
        images = [item[0] for item in batch]
        image_ids = [item[1] for item in batch]
        return images, image_ids

# --- 5. Task 2 輔助函數 ---
def get_digit_sequence(boxes, digit):
    if len(boxes) == 0:
        return ""

    x_center = (boxes[:, 0] + boxes[:, 2]) / 2
    digit_indices = np.argsort(x_center)
    digit_sequence = [str(digit[i].item()) for i in digit_indices]
    return ''.join(digit_sequence)

def get_number_from_detections(predictions, score_threshold=0.5):
    """ 從模型預測中提取數字序列 """
    # if predictions is None: return ''
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    keep_indices = scores >= score_threshold
    boxes = boxes[keep_indices]
    labels = labels[keep_indices]
    scores = scores[keep_indices]

    return get_digit_sequence(boxes, labels)

def get_ground_truth_number(target):
    """ 從 target 字典中獲取真實的數字序列 """
    boxes = target['boxes'].cpu()
    labels = target['labels'].cpu()

    return get_digit_sequence(boxes, labels)

def find_digits(image_id, pred_boxes, pred_labels, pred_scores,
                score_threshold=0.5, secondary_threshold=1.0):
    keep_indices = pred_scores >= score_threshold
    peek_indices = pred_scores < score_threshold
    peek_indices &= pred_scores >= secondary_threshold

    # print('keep_indices',keep_indices)
    # print('peek_indices',peek_indices)
    # print('pred_boxes',pred_boxes)

    keep_boxes = pred_boxes[keep_indices]
    keep_labels = pred_labels[keep_indices]
    keep_scores = pred_scores[keep_indices]


    peek_boxes = pred_boxes[peek_indices]
    peek_labels = pred_labels[peek_indices]
    peek_scores = pred_scores[peek_indices]

    result = []
    for box, label, score in zip(keep_boxes, keep_labels, keep_scores):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: continue
        bbox_coco = [
            round(x1,2),
            round(y1,2),
            round(w,2),
            round(h,2)
        ]
        result.append({
            'image_id': image_id,
            'category_id': label.item(),
            'bbox': bbox_coco,
            'score': round(score.item(),4)
        })

    for box, label, score in zip(peek_boxes, peek_labels, peek_scores):
        print(f'潛在數字({score.item()})')
        print(f'  image_id: {image_id}')
        print(f'  category_id: {label.item()}')

    return result

# --- 6. 評估函數 (包含指標和損失) ---
@torch.no_grad()
def evaluate(model, data_loader, device, score_threshold=0.5):
    """ 在給定的數據加載器上評估模型，計算指標和損失。 """
    category_map = data_loader.dataset.category_map
    metrics = {}
    task2_correct = 0
    task2_total = 0
    coco_results = []

    # --- 第一步: 計算指標 (Accuracy, mAP) ---
    print("評估: 計算指標 (Accuracy/mAP)...")
    model.eval()
    batch_num = len(data_loader)
    # i = 0
    for images, targets in tqdm(data_loader, total=batch_num,
                                desc='Validating', unit='batch',
                                position=0, leave=True):
        images = list(image.to(device) for image in images)

        outputs = model(images) # 獲取預測

        # 移到 CPU 處理
        targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
        outputs_cpu = [{k: v.cpu() for k, v in t.items()} for t in outputs]

        for target_cpu, output_cpu in zip(targets_cpu, outputs_cpu):
            image_id = target_cpu['image_id'].item()

            # Task 1 準備 (mAP)
            pred_boxes = output_cpu['boxes']
            pred_labels = output_cpu['labels']
            pred_scores = output_cpu['scores']

            coco_results.extend(find_digits(
                image_id, pred_boxes, pred_labels, pred_scores,
                score_threshold=0.01
            ))

            # Task 2 評估
            output_cpu['labels'] = torch.tensor(
                [category_map[output_cpu['labels'][i].item()]
                    for i in range(len(output_cpu['labels']))]
            )
            target_cpu['labels'] = torch.tensor(
                [category_map[target_cpu['labels'][i].item()]
                    for i in range(len(target_cpu['labels']))]
            )
            pred_str = get_number_from_detections(output_cpu, score_threshold)
            gt_str = get_ground_truth_number(target_cpu)
            if pred_str == gt_str: task2_correct += 1
            task2_total += 1
        # i += 1
        # if i >= 10: break

    if task2_total > 0:
        metrics['task2_accuracy'] = round(task2_correct / task2_total, 4)
    else:
        print(f'警告: 無法計算 Task 2 準確率，因為沒有有效的樣本。')
        metrics['task2_accuracy'] = 0.0

    if coco_results:
        print('評估: 計算 COCO mAP...')
        stats = data_loader.dataset.coco_eval(coco_results)
        metrics['mAP'] = round(stats.stats[0], 4)
        metrics['mAP_50'] = round(stats.stats[1], 4)
    else:
        print(f'警告: 無法計算 COCO mAP，因為沒有有效的樣本。')
        metrics['mAP'] = -1.0
        metrics['mAP_50'] = -1.0

    print("評估完成!")
    return metrics


# --- 7. 主要執行流程 ---
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument(
        'directory',
        help='The directory where you save all the data. '
        + 'Should include train, val, test.'
    )
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    data_root = args.directory
    train_json_file = path.join(data_root, 'train.json')
    valid_json_file = path.join(data_root, 'valid.json')
    train_data_dir = path.join(data_root, 'train')
    valid_data_dir = path.join(data_root, 'valid')
    test_data_dir = path.join(data_root, 'test')

    class_num = 10 + 1 # 10 數字 + 1 背景
    epoch_num = args.epoch
    batch_size = 16
    lr = args.lr
    scheduler_step = 1
    scheduler_rate = 0.31622776601
    freeze = 4
    weight_decay = 0.0005
    score_threshold = 0.5

    # --- 配置參數 ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    VALIDATION_INTERVAL = 1 # 每隔多少週期驗證一次

    # --- 創建 Dataset & DataLoader ---
    train_dataset = DigitDataset(json_path=train_json_file,
                                 image_dir=train_data_dir,
                                 transforms=get_transform(train=True))
    dataset_val = DigitDataset(json_path=valid_json_file,
                               image_dir=valid_data_dir,
                               transforms=get_transform(train=False))

    data_loader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )


    # --- 創建模型 ---
    print("創建 Faster R-CNN 模型...")
    # model = Model(class_num, device, scheduler_rate=0.31622776601)

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights='COCO_V1',
        trainable_backbone_layers=6-freeze
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_num)

    # MobileNetV3 可能需要修改 RPN anchor generator (根據你的數字大小調整)
    # anchor_sizes = ((32,), (64,), (128,), (256,), (512,)) # 預設值，可能需要調整
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    # model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    model.to(device)

    # --- 優化器 & 學習率調度器 ---
    params = [p for p in model.parameters() if p.requires_grad]
    # 嘗試使用 AdamW 可能效果更好
    # optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    optimizer = optim.SGD(params,
                          lr=lr, momentum=0.9,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=scheduler_step,
                                          gamma=scheduler_rate)

    # --- 訓練循環 ---
    print(f"使用設備: {device}")
    print(f"開始訓練 {epoch_num} 個 Epochs...")
    best_val_metric = -float('inf') # 初始化為負無窮大，因為準確率/mAP越大越好
    best_val_loss = float('inf')   # 初始化為正無窮大，損失越小越好
    metric_to_monitor = 'task2_accuracy' # 或 'mAP' 或 'average_val_loss'
    best_model_path = f'faster_rcnn_digits_best_{metric_to_monitor}.pth'

    start_train_time = time.time()

    for epoch in range(epoch_num):

        model.train() # 確保每個 epoch 開始時是訓練模式
        epoch_loss = 0
        total = 0
        batch_num = len(data_loader_train)
        start_time = time.time()
        optimizer.zero_grad() # 在 epoch 開始時清零一次梯度可能更穩定

        # i = 0
        print(f"\n--- Epoch [{epoch+1}/{epoch_num}] ---")
        for images, targets in tqdm(data_loader_train, total=batch_num,
                                    desc='Training', unit='batch',
                                    position=0, leave=True):

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                                for t in targets]

            # 前向傳播獲取損失
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values()
                                if torch.isfinite(loss)) # 只加總有效的損失

            # 反向傳播和優化
            loss.backward()
            # 梯度裁剪 (可選，有助於穩定訓練)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad() # 在 step 後清零梯度

            epoch_loss += loss.item() * len(images)
            total += len(images)

            # if i >= 10: break
            # i += 1

        # 更新學習率
        scheduler.step()

        avg_epoch_loss = epoch_loss / total
        end_time = time.time()
        print(f"\nEpoch [{epoch+1}/{epoch_num}] 訓練完成"\
              f", 平均訓練 Loss: {avg_epoch_loss:.4f}"\
              f", 耗時: {(end_time - start_time):.2f} 秒"\
              f", 當前學習率: {optimizer.param_groups[0]['lr']:.6f}")

        # --- 執行驗證 ---
        val_metrics = evaluate(model, data_loader_val, device=device, score_threshold=score_threshold)
        print(f"--- Epoch [{epoch+1}/{epoch_num}] 驗證結果 ---")
        val_loss_str = "N/A"
        if 'average_val_loss' in val_metrics:
            val_loss = val_metrics['average_val_loss']
            val_loss_str = f"{val_loss:.4f}"
            print(f"  平均驗證損失: {val_loss_str}")
            for k, v in val_metrics.items():
                if k.startswith('avg_val_loss_'): print(f"    {k}: {v:.4f}")
        else:
            val_loss = float('inf') # 如果沒有計算出損失

        val_acc_str = "N/A"
        if 'task2_accuracy' in val_metrics:
            val_acc = val_metrics['task2_accuracy']
            val_acc_str = f"{val_acc:.4f}"
            print(f"  Task 2 序列準確率: {val_acc_str}")
        else:
            val_acc = 0.0

        val_map_str = "N/A (pycocotools 未安裝或計算失敗)"
        if 'mAP' in val_metrics and val_metrics['mAP'] >= 0:
                val_map = val_metrics['mAP']
                val_map_str = f"{val_map:.4f}"
                print(f"  COCO mAP @ IoU=0.50:0.95: {val_map_str}")
                if 'mAP_50' in val_metrics: print(f"  COCO mAP @ IoU=0.50: {val_metrics['mAP_50']:.4f}")
        else:
                val_map = 0.0 # 或者設為 -1 表示未計算/失敗

        print("-" * 36)

        # --- 模型選擇邏輯 ---
        save_model = False
        if metric_to_monitor == 'average_val_loss':
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metric = val_loss # 更新記錄值
                print(f"*** 發現新的最佳驗證損失: {best_val_metric:.4f} ***")
                save_model = True
        elif metric_to_monitor == 'mAP':
            if val_map > best_val_metric:
                best_val_metric = val_map
                print(f"*** 發現新的最佳 mAP: {best_val_metric:.4f} ***")
                save_model = True
        else: # 默認監控 task2_accuracy
            if val_acc > best_val_metric:
                best_val_metric = val_acc
                print(f"*** 發現新的最佳 Task 2 準確率: {best_val_metric:.4f} ***")
                save_model = True

        if save_model:
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"*** 模型已保存到 {best_model_path} ***")
            except Exception as e:
                print(f"!! 保存模型時出錯: {e} !!")

    end_train_time = time.time()
    print(f"\n訓練完成! 總耗時: {(end_train_time - start_train_time):.2f} 秒")
    print(f"最佳驗證指標 ({metric_to_monitor}) 值: {best_val_metric:.4f}")
    print(f"最佳模型保存在: {best_model_path}")

    # --- 使用最佳模型進行最終測試集推斷 ---
    print(f"\n使用最佳模型 '{best_model_path}' 在測試集上進行推斷...")
    # 載入最佳模型狀態
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)




    test_dataset = TestDigitDataset(image_dir=test_data_dir,
                                    transforms=get_transform(train=False))
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    model.eval()
    test_preditions = {}

    start_test_time = time.time()
    with torch.no_grad():
        # i = 0
        batch_num = len(test_data_loader)
        for images, image_ids in tqdm(test_data_loader, total=batch_num,
                                      desc='Testing', unit='batch',
                                      position=0, leave=True):

            images = list(image.to(device) for image in images)
            predictions = model(images)

            # 處理batch中的每個圖像 (雖然 batch_size=1)
            for pred, image_id in zip(predictions, image_ids):
                if isinstance(image_id, torch.Tensor):
                    image_id = image_id.item()
                    print(f'isinstance({image_id}, torch.Tensor)')

                pred = {k: v.cpu() for k, v in pred.items()}
                test_preditions[image_id] = pred

            # i += 1
            # if i >= 10: break



    task1_results, task2_results = [], {}
    category_map = train_dataset.category_map
    for image_id, pred in test_preditions.items():
        boxes = pred['boxes']
        labels = pred['labels']
        scores = pred['scores']

        if len(boxes) == 0:
            task2_results[image_id] = '-1'
            continue

        # task 1
        pred_detection_task1 = find_digits(
            image_id, boxes, labels, scores,
            score_threshold=score_threshold,
            secondary_threshold=0.3
        )
        if not pred_detection_task1:
            pred_detection_task1 = find_digits(
                image_id, boxes, labels, scores,
                score_threshold=0.3
            )
        task1_results.extend(pred_detection_task1)

        # Task 2
        pred['labels'] = torch.tensor(
            [category_map[pred['labels'][i].item()]
             for i in range(len(pred['labels']))]
        )
        pred_str = get_number_from_detections(pred, score_threshold)
        task2_results[image_id] = pred_str

    end_test_time = time.time()
    print(f"測試集推斷完成! 耗時: {(end_test_time - start_test_time):.2f} 秒")

    # --- 生成提交文件 ---
    output_json = 'pred.json'
    output_csv = 'pred.csv'

    print(f"\n生成 Task 1 文件: {output_json}...")
    save_json(output_json, task1_results)
    print(f"  成功生成 {output_json} ({len(task1_results)} detections)")

    print(f"\n生成 Task 2 文件: {output_csv}...")
    sorted_image_ids = sorted(task2_results.keys())
    task2_list_result = [[k, task2_results[k].item()] for k in sorted_image_ids]
    save_csv(output_csv, task2_list_result, ['image_id', 'pred_label'])
    print(f"  成功生成 {output_csv} ({len(sorted_image_ids)} images)")
