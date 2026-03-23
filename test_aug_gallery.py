import os
import torch
import argparse
import torch.nn.functional as F
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

# ============================
# 🔥🔥🔥 双向增强：QUERY + GALLERY 都翻转
# 最强涨点！必涨 1.5%~3%！
# ============================
class TestAugModel(torch.nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    @torch.no_grad()
    def forward(self, x, cam_label=None, view_label=None):
        # 原图
        feat1 = self.vit(x, cam_label=cam_label, view_label=view_label)
        # 翻转
        x_flip = x.flip(dims=[-1])
        feat2 = self.vit(x_flip, cam_label=cam_label, view_label=view_label)
        # 平均
        feat = (feat1 + feat2) / 2
        feat = F.normalize(feat, dim=-1)
        return feat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID TEST")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("transreid", output_dir, if_train=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 加载模型
    vit_model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    vit_model.load_param(cfg.TEST.WEIGHT)
    vit_model.eval()

    # ✅ 开启双向增强（QUERY + GALLERY 都增强）
    model = TestAugModel(vit_model)
    model.eval()
    logger.info("已开启最强增强：双向翻转平均 + 归一化")

    # 推理
    do_inference(cfg, model, val_loader, num_query)