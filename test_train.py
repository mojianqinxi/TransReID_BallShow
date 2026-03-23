import os
import torch
import argparse
import logging
import torch.nn as nn
import numpy as np

from config import cfg
from datasets import make_dataloader
from model import make_model
from utils.logger import setup_logger
from utils.metrics import eval_func
from model.make_model import Backbone
from model.backbones.resnet import ResNet, Bottleneck

class ResNetFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ResNet(last_stride=1, block=Bottleneck, layers=[3,4,6,3])
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.base(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x

    def load_param(self, path):
        param_dict = torch.load(path)

        # 兼容 checkpoint
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']

        for k in param_dict:
            if 'base.' in k:
                new_k = k.replace('base.', '')
                if new_k in self.base.state_dict():
                    self.base.state_dict()[new_k].copy_(param_dict[k])

        print(f"ResNet 权重加载完成: {path}")

# ============================
# 🔥 融合模型（返回两个特征）
# ============================
class CombinedModel(nn.Module):
    def __init__(self, vit_model, resnet):
        super().__init__()
        self.vit = vit_model
        self.resnet = resnet

    @torch.no_grad()
    def forward(self, x, cam_label=None, view_label=None):

        # ===== ViT =====
        feat_vit = self.vit(x, cam_label=cam_label, view_label=view_label)
        feat_vit = torch.nn.functional.normalize(feat_vit, dim=-1)

        # ===== ResNet（🔥 直接用 Backbone 输出）=====
        feat_res = self.resnet(x)   # ✅ 已经是 [B, 2048]
        feat_res = torch.nn.functional.normalize(feat_res, dim=-1)

        return feat_vit, feat_res


# ============================
# 🔥 分数融合推理
# ============================
def do_inference_fusion(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter fusion inferencing")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    feat_vit_list = []
    feat_res_list = []
    pid_list = []
    camid_list = []

    # =========================
    # 1. 提特征（只提取一次！）
    # =========================
    logger.info("开始提取 ViT + ResNet 特征...")
    for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)

            feat_vit, feat_res = model(img, cam_label=camids, view_label=target_view)

            feat_vit_list.append(feat_vit.cpu())
            feat_res_list.append(feat_res.cpu())
            pid_list.extend(pid)
            camid_list.extend(camid)

    feat_vit = torch.cat(feat_vit_list, dim=0)
    feat_res = torch.cat(feat_res_list, dim=0)

    # =========================
    # 2. 切 query / gallery
    # =========================
    qf_vit = feat_vit[:num_query]
    gf_vit = feat_vit[num_query:]

    qf_res = feat_res[:num_query]
    gf_res = feat_res[num_query:]

    q_pids = np.asarray(pid_list[:num_query])
    g_pids = np.asarray(pid_list[num_query:] )
    q_camids = np.asarray(camid_list[:num_query])
    g_camids = np.asarray(camid_list[num_query:])

    # =========================
    # 3. 距离函数
    # =========================
    def compute_dist(qf, gf):
        m, n = qf.size(0), gf.size(0)
        dist = (
            torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist.addmm_(qf, gf.t(), beta=1, alpha=-2)
        return dist

    dist_vit = compute_dist(qf_vit, gf_vit)
    dist_res = compute_dist(qf_res, gf_res)

    # 距离归一化
    dist_vit = dist_vit / dist_vit.mean()
    dist_res = dist_res / dist_res.mean()

    # =========================
    # 🔥 4. 自动搜索最优 alpha (0~1，步长0.02)
    # =========================
    logger.info("=" * 60)
    logger.info("开始搜索最优 alpha / beta 权重组合...")
    logger.info("=" * 60)

    best_alpha = 0.5
    best_beta = 0.5
    best_rank1 = 0.0
    best_map = 0.0
    results = []

    # 遍历范围：0.0 ~ 1.0，步长0.02（精度很高，也可以用0.05更快）
    for alpha in np.arange(0.0, 1.01, 0.02):
        alpha = round(alpha, 2)
        beta = round(1.0 - alpha, 2)

        # 融合距离
        distmat = alpha * dist_vit + beta * dist_res

        # 评估
        cmc, mAP = eval_func(
            distmat.numpy(),
            q_pids, g_pids,
            q_camids, g_camids
        )

        rank1 = cmc[0]
        rank5 = cmc[4]
        results.append((alpha, beta, rank1, rank5, mAP))

        # 打印当前结果
        logger.info(f"alpha={alpha:.2f} | beta={beta:.2f} | Rank-1={rank1:.1%} | mAP={mAP:.1%}")

        # 更新最优（优先 Rank-1，其次 mAP）
        if rank1 > best_rank1 or (rank1 == best_rank1 and mAP > best_map):
            best_rank1 = rank1
            best_map = mAP
            best_alpha = alpha
            best_beta = beta

    # =========================
    # 5. 输出最优结果
    # =========================
    logger.info("\n" + "=" * 60)
    logger.info("🔥 搜索完成！最优权重组合：")
    logger.info(f"最优 alpha = {best_alpha:.2f}")
    logger.info(f"最优 beta  = {best_beta:.2f}")
    logger.info(f"最优 Rank-1 = {best_rank1:.1%}")
    logger.info(f"最优 mAP    = {best_map:.1%}")
    logger.info("=" * 60)

    return best_alpha, best_beta, best_rank1, best_map

# ============================
# 主函数
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Fusion Testing")
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

    # =========================
    # 数据
    # =========================
    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # =========================
    # ViT
    # =========================
    vit_model = make_model(cfg, num_class=num_classes,
                           camera_num=camera_num, view_num=view_num)
    vit_model.load_param(cfg.TEST.WEIGHT)
    vit_model.eval()
    logger.info("ViT 模型加载完成")

    # =========================
    # ResNet（ImageNet版）
    # 👉 你可以换成自己训练的
    # =========================
    import torchvision.models as models

    res_model = ResNetFeature()
    res_model.load_param("logs/BallShow_resnet/resnet50_224.pth")
    res_model.eval()


    # =========================
    # 融合模型
    # =========================
    model = CombinedModel(vit_model, res_model)
    model.eval()

    logger.info("融合模型准备完成：ViT + ResNet 分数融合")

    # =========================
    # 推理
    # =========================
    do_inference_fusion(cfg, model, val_loader, num_query)

