import torch
import pytorch_lightning as pl

from x_transformers import *
from x_transformers.autoregressive_wrapper import *

from timm.models.swin_transformer import SwinTransformer

import albumentations as alb
from albumentations.pytorch import ToTensorV2


class SwinTransformerOCR(pl.LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.encoder = CustomSwinTransformer( img_size=(cfg.height, cfg.width),
                                        patch_size=cfg.patch_size,
                                        in_chans=cfg.channels,
                                        num_classes=0,
                                        window_size=cfg.window_size,
                                        embed_dim=cfg.encoder_dim,
                                        depths=cfg.encoder_depth,
                                        num_heads=cfg.encoder_heads
                                        )
        self.decoder = CustomARWrapper(
                        TransformerWrapper(
                            num_tokens=len(tokenizer),
                            max_seq_len=cfg.max_seq_len,
                            attn_layers=Decoder(
                                dim=cfg.decoder_dim,
                                depth=cfg.decoder_depth,
                                heads=cfg.decoder_heads,
                                **cfg.decoder_cfg
                            )),
                        pad_value=cfg.pad_token
                    )
        self.bos_token = cfg.bos_token
        self.eos_token = cfg.eos_token
        self.max_seq_len = cfg.max_seq_len
        self.temperature = cfg.temperature

        self.transform = alb.Compose(
            [
                alb.Resize(cfg.height, cfg.width),
                alb.ToGray(always_apply=True),
                alb.Normalize(),
                ToTensorV2(),
            ]
        )

    def forward(self, x):
        '''
        x: (B, C, W, H)
        labels: (B, S)

        # B : batch size
        # W : image width
        # H : image height
        # S : source sequence length
        # E : hidden size
        # V : vocab size
        '''

        encoded = self.encoder(x)
        dec = self.decoder.generate(torch.LongTensor([self.bos_token]*len(x))[:, None].to(x.device), self.max_seq_len,
                                    eos_token=self.eos_token, context=encoded, temperature=self.temperature)
        return dec

    @torch.no_grad()
    def predict(self, images: list, batch_size=64):
        if not images: return ['']
        result = []
        for i in range((len(images)-1)//batch_size+1):
            batch = []
            for image in images[i:i+batch_size]:
                try:
                    image = self.transform(image=image)["image"]
                except Exception:
                    image = torch.ones((3, self.cfg.height, self.cfg.width)) * 255
                batch.append(image)
            batch = torch.stack(batch).to(self.device)
            dec = self(batch)
            result += self.tokenizer.decode(dec)
        return result


class CustomSwinTransformer(SwinTransformer):
    def __init__(self, img_size=224, *cfg, **kwcfg):
        super(CustomSwinTransformer, self).__init__(img_size=img_size, *cfg, **kwcfg)
        self.height, self.width = img_size

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C

        return x


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *cfg, **kwcfg):
        super(CustomARWrapper, self).__init__(*cfg, **kwcfg)

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwcfg):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwcfg.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]

            logits = self.net(x, mask=mask, **kwcfg)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out
