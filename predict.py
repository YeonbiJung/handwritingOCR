import torch
import cv2
import argparse
import traceback
import random
import tqdm
import numpy as np
from PIL import Image
from time import time
from pathlib import Path

from torchvision.transforms.functional import crop

from craft.model import CRAFT
from swin_transformer.models import SwinTransformerOCR
from utils import load_setting, load_tokenizer, load_json_data, calc_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", "-t", type=str, default="test",
                        help="OCR target (image or directory)")
    args = parser.parse_args()

    cfg = load_setting("assets/config.yaml")
    cfg.update(vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.craft.device = device
    print("setting:", cfg)

    # craft
    craft = CRAFT(cfg.craft).to(device)
    saved = torch.load(cfg.craft.weight, map_location=device)
    craft.load_state_dict(saved)

    # swin-transformer
    tokenizer = load_tokenizer(cfg.swin_transformer.tokenizer)
    swin_transformer = SwinTransformerOCR(cfg.swin_transformer, tokenizer).to(device)
    saved = torch.load(cfg.swin_transformer.weight, map_location=device)
    swin_transformer.load_state_dict(saved)

    target = Path(args.target)
    if target.is_dir():
        target = list(target.glob("**/*.json"))
    elif target.suffix == '.txt':
        target = [target.parent.parent / line.strip() for line in target.open('r', encoding='utf8')]
    else:
        raise FileNotFoundError("target not found.")

    craft_time, swin_time = [], []
    scores = []
    logger = open("test_log.log", "a", encoding="utf8")
    logger.write("-"*20+"\n")
    logger.write('[image_fn], f1, precision, recall, pred_box_num, gt_box_num\n')
    gt_boxes = []
    try:
        for i, json_fn in tqdm.tqdm(enumerate(target), total=len(target)):
            start = time()
            # page = np.array(Image.open(image_fn).convert('BGR'))
            page = cv2.imread(str(json_fn.with_suffix('.png')))
            gt = load_json_data(json_fn)

            pred_boxes = craft.predict(page)
            craft_time.append(time()-start)

            # img = page.copy()
            # for box in pred_boxes:
            #     lx, ly, rx, ry = box
            #     img = cv2.rectangle(img, [lx, ly], [rx, ry], color=(255, 0, 0), thickness=1)
            # cv2.imwrite(f'test_result/{json_fn.stem}.png', img)

            start = time()
            cropped_images = []
            for box in pred_boxes:
                lx, ly, rx, ry = box
                cropped = page[ly:ry, lx:rx, :]
                cropped_images.append(cropped)
            result = swin_transformer.predict(cropped_images,
                                            batch_size=cfg.swin_transformer.bs)
            swin_time.append(time()-start)
            (f1, precision, recall), refined_pred = calc_score(list(zip(pred_boxes, result)), gt)

            # img = page.copy()
            # for box, text in refined_pred:
            #     lx, ly, rx, ry = box
            #     img = cv2.rectangle(img, [lx, ly], [rx, ry], color=(255, 0, 0), thickness=1)
            # for box, text in gt:
            #     lx, ly, rx, ry = box
            #     img = cv2.rectangle(img, [lx, ly], [rx, ry], color=(0, 255, 255), thickness=1)
            # cv2.imwrite(f'test_result/{json_fn.stem}_refined.png', img)

            logger.write(f"[{json_fn.stem}],{f1},{precision},{recall},{len(cropped_images),len(gt)}\n")
            scores.append(f1)
    except Exception as e:
        print(traceback.format_exc())
        logger.write(traceback.format_exc()+'\n')
    logger.close()
    print("total score:", sum(scores) / len(scores))
    print("craft avg time:", sum(craft_time) / len(craft_time))
    print("swin avg time:", sum(swin_time) / len(swin_time))
