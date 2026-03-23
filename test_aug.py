import os
import torch
import argparse
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


# ============================
# 测试增强：原图 + 水平翻转（最稳！不改变尺寸！）
# ============================
class TestAugModel(torch.nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    @torch.no_grad()
    def forward(self, x, cam_label=None, view_label=None):
        # 原图
        feat1 = self.vit(x, cam_label=cam_label, view_label=view_label)

        # 水平翻转（不改变尺寸）
        x_flip = x.flip(dims=[-1])
        feat2 = self.vit(x_flip, cam_label=cam_label, view_label=view_label)

        # 平均
        final_feat = (feat1 + feat2) / 2.0
        return final_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Test Aug")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 加载数据
    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 加载你的原生模型
    vit_model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    vit_model.load_param(cfg.TEST.WEIGHT)
    vit_model.eval()

    # 开启测试增强
    model = TestAugModel(vit_model)
    model.eval()
    logger.info("测试增强：原图 + 水平翻转")

    # 推理
    if cfg.DATASETS.NAMES == 'VehicleID':
        all_rank_1 = 0
        for trial in range(10):
            rank1, rank5 = do_inference(cfg, model, val_loader, num_query)
            all_rank_1 += rank1
        logger.info("平均 Rank1: {:.1%}".format(all_rank_1 / 10.0))
    else:
        do_inference(cfg, model, val_loader, num_query)