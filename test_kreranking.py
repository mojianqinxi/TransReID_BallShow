import os
import torch
import argparse
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

# ============================
# 测试增强（翻转 + 归一化）
# ============================
class TestAugModel(torch.nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    @torch.no_grad()
    def forward(self, x, cam_label=None, view_label=None):
        # 原图
        feat1 = self.vit(x, cam_label=cam_label, view_label=view_label)
        # 水平翻转
        x_flip = x.flip(dims=[-1])
        feat2 = self.vit(x_flip, cam_label=cam_label, view_label=view_label)
        # 平均
        feat = (feat1 + feat2) / 2.0
        return feat

# ============================
# 主程序
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", nargs='*', default=[])
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("transreid", cfg.OUTPUT_DIR, False)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID.strip("'()")

    # 加载数据
    train_loader, _, val_loader, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)

    # 加载模型
    model = make_model(cfg, num_classes, cam_num, view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval()

    # 测试增强
    aug_model = TestAugModel(model)

    # 直接调用官方 inference，100% 不报错
    print("✅ 测试增强：原图 + 水平翻转 已开启")
    do_inference(cfg, aug_model, val_loader, num_query)