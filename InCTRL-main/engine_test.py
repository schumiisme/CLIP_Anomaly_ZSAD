# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""
import os
import random
import json
import open_clip
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
import open_clip.utils.checkpoint as cu
import open_clip.utils.distributed as du
import open_clip.utils.logging as logging
import open_clip.utils.misc as misc
import numpy as np
import torch
from datasets import loader
from torchvision import transforms
from open_clip.utils.meters import EpochTimer, TrainMeter, ValMeter
from sklearn.metrics import average_precision_score, roc_auc_score
from binary_focal_loss import BinaryFocalLoss
import torch.distributed as dist
import matplotlib.pyplot as plt
from open_clip.model import get_cast_dtype
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
from PIL import Image

logger = logging.get_logger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')

import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ✅ 更新後的 save_anomaly_map()
def save_anomaly_map(anomaly_map, orig_img, save_path):
    import cv2
    import numpy as np

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if anomaly_map.dim() == 3:
        anomaly_map = anomaly_map.squeeze(0)
    anomaly_map = anomaly_map.cpu().numpy()

    orig_img = orig_img.detach().cpu().permute(1, 2, 0).numpy()  # CHW → HWC
    orig_img = (orig_img * 255).astype(np.uint8)

    # 將 anomaly map 插值回原圖大小
    anomaly_map_resized = cv2.resize(anomaly_map, (orig_img.shape[1], orig_img.shape[0]))
    anomaly_map_resized = (anomaly_map_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)

    # 疊圖
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)


@torch.no_grad()
def eval_epoch(val_loader, model, cfg, tokenizer, normal_list=None, mode="val"):
    import os
    import torchvision.transforms.functional as TF
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

    def compute_image_f1(scores, labels, threshold=0.5):
        preds = (scores >= threshold).astype(int)
        return f1_score(labels, preds)

    def compute_precision(scores, labels, threshold=0.5):
        preds = (scores >= threshold).astype(int)
        return precision_score(labels, preds)

    def compute_recall(scores, labels, threshold=0.5):
        preds = (scores >= threshold).astype(int)
        return recall_score(labels, preds)

    model.eval()
    scores = []
    gts = []

    os.makedirs(os.path.join("vis_outputs", cfg.category), exist_ok=True)

    normal_saved = 0
    max_normal_save = 10

    for cur_iter, (inputs, types, labels) in enumerate(val_loader):
        labels = labels.cuda()
        inputs = [inp.cuda() for inp in inputs]

        with torch.no_grad():
            preds, vis_dict = model(tokenizer, inputs, types, normal_list)

        scores.append(preds.detach().cpu())
        gts.append(labels.detach().cpu())

        anomaly_maps = vis_dict["anomaly_map"]
        orig_imgs = vis_dict["orig_img"]

        for i in range(len(labels)):
            is_anomaly = labels[i].item() == 1
            is_normal = labels[i].item() == 0

            # 儲存條件：anomaly 或最多儲存 N 張 normal
            if is_anomaly or (is_normal and normal_saved < max_normal_save):
                anomaly_map = anomaly_maps[i].cpu().squeeze(0)  # (15,15)
                orig_img = orig_imgs[i].cpu()  # (3,240,240)

                label_str = "anomaly" if is_anomaly else "normal"
                img_name = f"{cfg.category}_{cur_iter:03d}_{i:02d}_{label_str}"
                base_path = os.path.join("vis_outputs", cfg.category, img_name)

                # 原圖（修正顏色）
                if torch.is_floating_point(orig_img):
                    ori_img = TF.to_pil_image(orig_img.clamp(0.0, 1.0))
                else:
                    ori_img = TF.to_pil_image(orig_img.clamp(0, 255).to(torch.uint8))
                ori_img.save(base_path + "_ori.jpg")

                # Heatmap
                plt.imsave(base_path + "_heatmap.jpg", anomaly_map.numpy(), cmap='jet')

                # Overlay
                orig_np = np.array(ori_img)
                anomaly_resized = TF.resize(
                    TF.to_pil_image(anomaly_map.unsqueeze(0)), [240, 240]
                )
                anomaly_resized = np.array(anomaly_resized)

                plt.figure(figsize=(4, 4))
                plt.imshow(orig_np)
                plt.imshow(anomaly_resized, cmap='jet', alpha=0.5)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(base_path + "_gt.jpg", bbox_inches='tight')
                plt.close()

                if is_normal:
                    normal_saved += 1

    # 評估 AUROC / AUPR / F1 / Precision / Recall
    scores = torch.cat(scores).numpy()
    gts = torch.cat(gts).numpy()

    auroc = roc_auc_score(gts, scores)
    aupr = average_precision_score(gts, scores)
    f1 = compute_image_f1(scores, gts)
    prec = compute_precision(scores, gts)
    recall = compute_recall(scores, gts)

    print(f"[InCTRL] AUC-ROC: {auroc:.4f}, AUC-PR: {aupr:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}")

    return {
        "category": cfg.category,
        "i_roc": auroc,
        "p_pro": aupr,
        "i_f1": f1,
        "p_roc": prec,
        "r_f1": recall
    }


def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;


def drawing(cfg, data, xlabel, ylabel, dir):
    plt.switch_backend('Agg')
    plt.figure()
    plt.plot(data, 'b', label='loss')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, dir))


def test(cfg, load=None, mode = None):
    """
    Perform testing on the pretrained model.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    device = torch.cuda.current_device()

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = open_clip.model.InCTRL(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
    model = model.cuda(device=device)

    cu.load_test_checkpoint(cfg, model)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    if load == None:
        load = loader.construct_loader(cfg, "test", transform)
        mode = "test"

    few_shot_path = os.path.join(cfg.few_shot_dir, cfg.category+".pt")
    normal_list = torch.load(few_shot_path)

    # Create meters.
    total_roc = eval_epoch(load, model, cfg, tokenizer, normal_list, mode)

    return total_roc
