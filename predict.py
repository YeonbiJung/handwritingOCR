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
        target = list(target.glob("**/*.png"))
    else:
        raise FileNotFoundError("target not found")
    try:
        for i, targ in tqdm.tqdm(enumerate(target), total=len(target)):
            start = time()

            page = cv2.imread(str(targ))

            pred_boxes = craft.predict(page)

            cropped_images = []
            for box in pred_boxes:
                lx, ly, rx, ry = box
                cropped = page[ly:ry, lx:rx, :]
                cropped_images.append(cropped)
            result = swin_transformer.predict(cropped_images,
                                            batch_size=cfg.swin_transformer.bs)
            print(result)
            
    except Exception as e:
        print(traceback.format_exc())
        logger.write(traceback.format_exc()+'\n')
