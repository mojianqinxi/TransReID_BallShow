import os
import torch
import argparse
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

# ============================
# 🔥 最终版：拼接融合（不管维度！必跑！）
# ============================
class CombinedModel(torch.nn.Module):
    def __init__(self, vit_model, resnet):
        super().__init__()
        self.vit = vit_model
        self.resnet = resnet

    @torch.no_grad()
    def forward(self, x, cam_label=None, view_label=None):
        # 1. ViT 特征 (你的输出：3840)
        feat_vit = self.vit(x, cam_label=cam_label, view_label=view_label)
        feat_vit = torch.nn.functional.normalize(feat_vit, dim=-1)

        # 2. ResNet 特征 (2048)
        feat_res = self.resnet(x)
        feat_res = torch.nn.functional.normalize(feat_res, dim=-1)

        # ============================
        # ✅ 拼接：不管维度多少，都能拼！
        # ============================
        final_feat = torch.cat([feat_vit, feat_res], dim=-1)

        return final_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Testing")
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

    # ============================
    # 1. 你的 ViT
    # ============================
    vit_model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    vit_model.load_param(cfg.TEST.WEIGHT)
    vit_model.eval()
    logger.info("ViT 模型加载完成")

    # ============================
    # 2. ResNet50 官方预训练
    # ============================
    import torchvision.models as models
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_feature = torch.nn.Sequential(*list(resnet50.children())[:-1])

    class ResNetFeature(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            feat = self.model(x)
            return feat.view(feat.size(0), -1)

    res_model = ResNetFeature(resnet_feature)
    res_model.eval()

    # ============================
    # 3. 复合模型
    # ============================
    model = CombinedModel(vit_model, res_model)
    model.eval()
    logger.info("复合模型准备完成：ViT + ResNet50 拼接融合")

    # ============================
    # 推理
    # ============================
    if cfg.DATASETS.NAMES == 'VehicleID':
        all_rank_1 = 0
        for trial in range(10):
            rank1, rank5 = do_inference(cfg, model, val_loader, num_query)
            all_rank_1 += rank1
        logger.info("平均 Rank1: {:.1%}".format(all_rank_1/10))
    else:
        do_inference(cfg, model, val_loader, num_query)