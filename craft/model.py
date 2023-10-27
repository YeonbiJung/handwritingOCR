import pytorch_lightning as pl
import torch
import cv2
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from craft.craft_modules import double_conv, vgg16_bn
from craft.misc import generate_word_bbox_batch, merge_adjacent_boxes


class CRAFT(pl.LightningModule):
    def __init__(self, cfg, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()
        self.cfg = cfg

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        self.transform = transforms.ToTensor()
        self.eval()

    def forward(self, x):
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1), feature

    def predict(self, image: np.ndarray):
        """
        input:
            img: [H, W, C]
        out:
            list of word boxes
        """
        img, (h_pad, w_pad), ratio = self.resize(image, self.cfg.img_size)
        x = self.transform(img).unsqueeze(0).to(self.device)
        out, _ = self.forward(x)

        half = self.cfg.img_size // 4
        character_map = out[:, :, :, 0].data.cpu().numpy()
        affinity_map = out[:, :, :, 1].data.cpu().numpy()

        characters = [character_map[:, :half, :half],
                      character_map[:, :half, half:],
                      character_map[:, half:, :half],
                      character_map[:, half:, half:]]

        affinities = [affinity_map[:, :half, :half],
                      affinity_map[:, :half, half:],
                      affinity_map[:, half:, :half],
                      affinity_map[:, half:, half:]]

        total_boxes = []
        merged_boxes = np.empty((0, 2, 2))
        for i, (character, affinity) in enumerate(zip(characters, affinities)):
            boxes = generate_word_bbox_batch(character,
                                            affinity,
                                            character_threshold=self.cfg.threshold_character,
                                            affinity_threshold=self.cfg.threshold_affinity,
                                            word_threshold=self.cfg.threshold_word)
            boxes = boxes[0][:,[0,2],:,:].squeeze(2)
            # boxes[:, 1, :] += max(image.shape[:2]) // 1000

            """
            lx : boxes[:, 0, 0]
            ly : boxes[:, 0, 1]
            rx : boxes[:, 1, 0]
            ry : boxes[:, 1, 1]
            """
            if i == 1:
                boxes[:, :, 0] += half
                # 0과 좌측 박스 비교
                merge_res = merge_adjacent_boxes(boxes, total_boxes[0], affinity_map[0,:,:], "left")
                boxes, total_boxes[0], nb = merge_res
                merged_boxes = np.concatenate((merged_boxes, nb))
            elif i == 2:
                boxes[:, :, 1] += half
                merge_res = merge_adjacent_boxes(boxes, total_boxes[0], affinity_map[0,:,:], "up")
                # 0과 상측 박스 비교
                boxes, total_boxes[0], nb = merge_res
                merged_boxes = np.concatenate((merged_boxes, nb))
            elif i == 3:
                boxes[:, :, 0] += half
                boxes[:, :, 1] += half

                # 1과 상측 박스 비교
                merge_res = merge_adjacent_boxes(boxes, total_boxes[1], affinity_map[0,:,:], "up")
                boxes, total_boxes[1], nb = merge_res
                merged_boxes = np.concatenate((merged_boxes, nb))

                # 2와 좌측 박스 비교
                merge_res = merge_adjacent_boxes(boxes, total_boxes[2], affinity_map[0,:,:], "left")
                boxes, total_boxes[2], nb = merge_res
                merged_boxes = np.concatenate((merged_boxes, nb))

            total_boxes.append(boxes)

        # 최종 결과 수정
        total_boxes = np.concatenate(total_boxes+[merged_boxes])*2
        total_boxes[:, :, 0] -= w_pad
        total_boxes[:, :, 1] -= h_pad
        total_boxes = total_boxes.reshape(-1,4) / ratio
        results = total_boxes.astype(int).tolist()

        return results


    def resize(self, image, big_side=1536):
        """
        Resizing the image while maintaining the aspect ratio and padding
        param:
            image: np.array, dtype=np.uint8, shape=[height, width, 3]
            big_side: new size to be reshaped to
        return:
            resized PIL image
            pad info (h_pad, w_pad)
            resize ratio
        """

        height, width, channel = image.shape
        ratio = big_side/max(height, width)
        big_resize = (int(width*ratio), int(height*ratio))
        image = cv2.resize(image, big_resize)

        big_image = np.ones([big_side, big_side, 3], dtype=np.float32)*255
        h_pad, w_pad = (big_side-image.shape[0])//2, (big_side-image.shape[1])//2
        big_image[h_pad: h_pad + image.shape[0], w_pad: w_pad + image.shape[1]] = image
        big_image = big_image.astype(np.uint8)

        return Image.fromarray(big_image), (h_pad, w_pad), ratio
