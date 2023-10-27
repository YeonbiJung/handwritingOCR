import copy
import yaml
import pickle
import json
from easydict import EasyDict

def load_setting(setting):
    with open(setting, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(cfg)

def load_tokenizer(path):
    try:
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)
    except ModuleNotFoundError:
        import sys, os
        sys.path.append(f'{os.getcwd()}/swin_transformer')
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)

    print("tokenizer loaded from {}".format(path))
    return tokenizer

def load_json_data(fn):
    with open(fn, 'r', encoding='utf8') as f:
        x = json.load(f)
    bboxes = []
    for d in x["bbox"]:
        txt = d["data"]
        lx, rx = d["x"][0], d["x"][2]
        ly, ry = d["y"][0], d["y"][1]
        bboxes.append(((lx, ly, rx, ry), txt))
    return bboxes

def is_in_box(point, box):
    x, y = point
    lx, ly, rx, ry = box
    if lx <= x <= rx and ly <= y <= ry:
        return True
    else:
        return False

def merge_two_box_with_text(lbox, rbox):
    """
    lbox [[lx, ly, rx, ry], text]
    rbox [[lx, ly, rx, ry], text]
    """
    lx, rx = min(lbox[0][0], rbox[0][0]), max(lbox[0][2], rbox[0][2])
    ly, ry = min(lbox[0][1], rbox[0][1]), max(lbox[0][3], rbox[0][3])
    text = lbox[1] + rbox[1]
    return ((lx, ly, rx, ry), text)

def match_text(pred_text, gt_text):
    if gt_text in pred_text: # or pred_text in gt_text:
        return True
    else:
        return False

def do_overlap(box1, box2):
    lx1, ly1, rx1, ry1 = box1
    lx2, ly2, rx2, ry2 = box2

    if lx1 < rx2 and lx2 < rx1 and ly1 < ry2 and ly2 < ry1:
        # if lx2 <= lx1 and rx2 >= rx1 and ly2 <= ly1 and ry2 >= ry1:
        return True
    else:
        return False

def calc_iou(box1, box2):
    lx1, ly1, rx1, ry1 = box1
    lx2, ly2, rx2, ry2 = box2

    inter = min(rx1-lx2, rx2-lx1) * min(ry1-ly2, ry2-ly1)
    comb = (rx1-lx1)*(ry1-ly1) + (rx2-lx2)*(ry2-ly2) - inter

    return inter / comb

def calc_f1(pred_boxes, gt_boxes, thres=0.5):
    """
    complexity : n_1 * n_2 * iou_complexity
    input:
        pred_boxes : [[lx, ly, rx, ry], ...]
        gt_boxes : [[lx, ly, rx, ry], ...]
    return:
        f1_score (0.0 ~ 1.0)
    """
    pred_boxes = copy.copy(pred_boxes)
    success = []

    for i, box1 in enumerate(gt_boxes):
        candidate = []
        for j, box2 in enumerate(pred_boxes):
            if do_overlap(box1, box2):
                iou = calc_iou(box1, box2)
                candidate.append((j, box2, iou))
        if candidate:
            best_iou = sorted(candidate, key=lambda x: x[2])[-1]
            if best_iou[2] > thres:
                success.append(pred_boxes.pop(best_iou[0]))
    failure = len(gt_boxes) - len(success)

    return len(success) / (len(success) + failure)

def calc_f1_with_text(pred_boxes, gt_boxes, thres=0.5):
    """
    complexity : n_1 * n_2 * iou_complexity
    input:
        pred_boxes : [[[lx, ly, rx, ry], text ], ...]
        gt_boxes : [[[lx, ly, rx, ry], text], ...]
    return:
        tp : matched
        tn :
        fp : when iou < threshold
        fn : when couldn't detect word box
        pred_boxes with word
    """
    pred_boxes = copy.copy(pred_boxes)
    tp, fp = 0, 0
    for i, (gt_box, gt_text) in enumerate(gt_boxes):
        candidate = []
        for j, (pred_box, pred_text) in enumerate(pred_boxes):
            if do_overlap(gt_box, pred_box):
                iou = calc_iou(gt_box, pred_box)
                candidate.append((j, pred_box, iou, pred_text))
        if candidate:
            best_iou = sorted(candidate, key=lambda x: x[2])[-1]
            if best_iou[2] > thres:
                if match_text(best_iou[3], gt_text):
                    tp += 1
                else:
                    fp += 1
            else:
                fp +=1

    fn = max(0, len(gt_boxes) - (tp + fp))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1, precision + recall)

    return f1, precision, recall

def calc_score(pred, gt, thres=0.5):
    pred_refined = []
    pred = sorted(pred, key=lambda x: x[0][0]) # lx 기준 정렬
    for gt_point, gt_text in gt:
        in_box = []
        for i, p in enumerate(pred):
            # pred_lx, pred_ly, pred_rx, pred_ry = p[0]
            # if is_in_box([(pred_lx+pred_rx)/2, (pred_ly+pred_ry)/2], gt_point):
            #     in_box.append(p)
            if do_overlap(p[0], gt_point):
                in_box.append(p)
        if len(in_box) >= 1:
            pred_refined.append(in_box)

    for i, boxes in enumerate(pred_refined):
        if len(boxes) == 1:
            pred_refined[i] = boxes[0]
        else:
            temp = boxes[0]
            for box in boxes[1:]:
                temp = merge_two_box_with_text(temp, box)
            pred_refined[i] = temp

    return calc_f1_with_text(pred_refined, gt), pred_refined
