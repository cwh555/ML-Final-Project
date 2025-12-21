import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm
from typing import Dict, List, Any
import numpy as np
from dataset import get_subset_dataloader

# ==========================================
# Local Imports
# ==========================================
# 假設你的檔案結構如下:
# src/
#   dataset.py
#   models.py
#   stn_model.py
#   losses.py
#   metrics.py
from dataset import (
    GeometricDataset, Compose, BaseTransform, 
    LoadGeometricData, ToTensord, RandomGeoAugment,
    ImageNoiseColorAugment, ResizeImaged, MorphologyAugment,
    NonLinearAugment,
    MNISTGeometricDataset,
)

from cage_wo_res_model import FullContrastiveModel
# from old_model import FullContrastiveModel
from new_STNmodel import DeformationExtractor
from STNmodel import ShapeTransformationNetwork
from loss import InfoNCELoss, ContrastiveAccuracy
from utils import set_random_seed
from cage_wo_res_eval_utils import evaluate_prototypical_k_shot, evaluate_class_analysis

# ==========================================
# Training & Validation Functions
# ==========================================

def save_checkpoint(model, optimizer, epoch, config, filename="checkpoint.pth"):
    """
    只儲存 Encoder 的權重，不存 Wrapper 和 STN。
    """
    save_path = os.path.join(config.train.save_dir, filename)
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': model.encoder.state_dict(), # 只存 Encoder
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Saved checkpoint to {save_path}")

def log_wandb_images(batch, batch_idx, prefix="train"):
    """
    從 Batch 中提取第一張圖片的兩個 View 及其 Mask 並上傳到 WandB。
    
    Batch Structure:
      image: (B, 2, 5, 4, H, W) -> 需要解開
      mask:  (B, 2, 1, H, W)
    """
    # 只在每個 Epoch 的第一個 Batch 紀錄，節省資源
    if not wandb.run:# or batch_idx != 0:
        return

    # 取 Batch 中的第一筆資料 (Sample 0)
    # images shape: (B, 2, 5, 4, H, W)
    # 我們只看 Template Stack 中的第 0 個 (通常最能代表原始圖，因為 dataset concat 邏輯)
    # 且只看 Channel 0 (Image 本體，Channel 1~3 是 template)
    
    # View 1
    # [Sample 0, View 0, Template 0, Channel 0-3 (Take :3 for RGB)]
    # Dataset 邏輯: cat([img, template], dim=0)
    # 所以 img 在前，template 在後。如果 img 是 RGB (3ch)，則 :3 是原圖。
    # 如果 img 是 Gray (1ch)，則 0 是原圖。
    
    # 假設 Image 是 3 Channel (RGB)
    img_v1 = batch['image'][0, 0, 0, :3, :, :].detach().cpu()
    mask_v1 = batch['mask'][0, 0, :, :, :].detach().cpu()
    
    # View 2
    img_v2 = batch['image'][0, 1, 0, :3, :, :].detach().cpu()
    mask_v2 = batch['mask'][0, 1, :, :, :].detach().cpu()
    # Log to WandB
    wandb.log({
        f"{prefix}/visual_samples": [
            wandb.Image(img_v1, caption=f"{prefix}_view1_img"),
            wandb.Image(mask_v1, caption=f"{prefix}_view1_mask"),
            wandb.Image(img_v2, caption=f"{prefix}_view2_img"),
            wandb.Image(mask_v2, caption=f"{prefix}_view2_mask"),
        ]
    })

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train() # Wrapper 會強制 STN 保持 Eval
    
    total_loss = 0
    total_acc = 0
    num_batches = 0
    metric_calc = ContrastiveAccuracy()
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Input shape: (B, 2, 5, 4, H, W) (因為 Dataset two_views=True)
        log_wandb_images(batch, batch_idx, prefix="train")
        images = batch['image'].to(device)
        # names = np.array(batch['template_names']).transpose(2, 0, 1)
        # names = names.reshape(-1, names.shape[-1])
        weights_boundary = batch['weights_boundary'].to(device)
        wb_view1 = weights_boundary[:, 0]
        wb_view2 = weights_boundary[:, 1]
        concat_wb = torch.cat([wb_view1, wb_view2], dim=0)
        
        # 拆分 Views 並重組 -> (2B, 5, 4, H, W)
        view1 = images[:, 0]
        view2 = images[:, 1]
        concat_input = torch.cat([view1, view2], dim=0)
        
        optimizer.zero_grad()
        
        # Forward (STN forward 會自動用 no_grad 和 test=True)
        # embeddings = model(concat_input)
        embeddings = model(concat_input, weight_boundary=concat_wb, template_names=None)    
        
        # Loss & Backward
        loss = criterion(embeddings)
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = metric_calc(embeddings)
        
        # Stats
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})
        
        if wandb.run:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_acc": acc,
                "train/lr": optimizer.param_groups[0]['lr']
            })
            
    return total_loss / num_batches, total_acc / num_batches

def validate(model, dataloader, device):
    model.eval() # Encoder Eval
    
    total_acc = 0
    num_batches = 0
    metric_calc = ContrastiveAccuracy()
    
    # 使用 inference_mode 加速並節省記憶體
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            log_wandb_images(batch, batch_idx, prefix="val")
            images = batch['image'].to(device)
            # names = np.array(batch['template_names']).transpose(2, 0, 1)
            # names = names.reshape(-1, names.shape[-1])
            weights_boundary = batch['weights_boundary'].to(device)
            wb_view1 = weights_boundary[:, 0]
            wb_view2 = weights_boundary[:, 1]
            concat_wb = torch.cat([wb_view1, wb_view2], dim=0)
            
            # Validation 也需要 two views 才能算 Accuracy
            view1 = images[:, 0]
            view2 = images[:, 1]
            concat_input = torch.cat([view1, view2], dim=0)
            
            # embeddings = model(concat_input)
            # embeddings = model(concat_input, template_names=names)
            embeddings = model(concat_input, weight_boundary=concat_wb, template_names=None)
            
            acc = metric_calc(embeddings)
            total_acc += acc
            num_batches += 1
            
    return total_acc / num_batches

# ==========================================
# Main Execution
# ==========================================

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Train Geometric Contrastive Model")
    parser.add_argument("--config", type=str, default="default", help="Config file name in configs/ (without .yaml)")
    args = parser.parse_args()
    
    config_path = f"configs/{args.config}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    
    # 2. Setup
    device = torch.device(cfg.project.device)
    set_random_seed(cfg.project.seed)
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    # 3. Transforms
    # 定義訓練與驗證共用的前處理 (Augmentation 另外在 Dataset 內部呼叫)
    # 注意順序: Load -> ToTensor -> Resize -> Augment
    common_transforms = [
        LoadGeometricData(load_key="data_path", output_keys=["image", "mask", "label"]),
        ToTensord(keys=["image", "mask", "label"]),
        ResizeImaged(keys=["image", "mask"], size=[cfg.data.img_size, cfg.data.img_size])
    ]
    
    # 訓練用 Transform: 加入幾何增強
    train_transform = Compose(common_transforms + [
        RandomGeoAugment(keys=["image", "mask"], translate_range=0.1, degrees=20),
        MorphologyAugment(keys=["image", "mask"], prob=0.5, kernel_size=3, iteration_range=(1, 8)),
        NonLinearAugment(
            keys=["image", "mask"],
            perspective_prob=0.5, distortion_scale=0.3,
            elastic_prob=0.5, alpha_range=(50.0, 100.0), sigma_range=(8.0, 12.0)
        ),
        ImageNoiseColorAugment(keys=["image"]),
    ])
    
    # 驗證用 Transform: 加入同樣的幾何增強 
    # (因為算 Contrastive Accuracy 需要同一張圖的兩個不同 View，所以 Val 也要 Augment)
    val_transform = Compose(common_transforms + [
        RandomGeoAugment(keys=["image", "mask"], translate_range=0.1, degrees=15),
        MorphologyAugment(keys=["image", "mask"], prob=0.5, kernel_size=3, iteration_range=(1, 8)),
        NonLinearAugment(
            keys=["image", "mask"],
            perspective_prob=0.5, distortion_scale=0.3,
            elastic_prob=0.5, alpha_range=(50.0, 100.0), sigma_range=(8.0, 12.0)
        ),
        ImageNoiseColorAugment(keys=["image"]),
    ])

    # 4. Datasets & DataLoaders
    # Train
    train_ds = GeometricDataset(
        base_dir=cfg.data.train.root_dir,
        template_dir=cfg.data.template_dir,
        transform=train_transform,
        two_views=True # 訓練必須開啟
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.train.batch_size,
        shuffle=cfg.data.train.shuffle,
        num_workers=cfg.data.train.num_workers,
        drop_last=cfg.data.train.drop_last,
        # pin_memory=cfg.data.train.pin_memory,
        prefetch_factor=3,
    )
    
    # Val
    val_ds = GeometricDataset(
        base_dir=cfg.data.val.root_dir,
        template_dir=cfg.data.template_dir,
        transform=val_transform,
        two_views=True # 驗證為了算 Metric 也必須開啟
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.val.batch_size,
        shuffle=cfg.data.val.shuffle,
        num_workers=cfg.data.val.num_workers,
        drop_last=cfg.data.val.drop_last,
        # pin_memory=cfg.data.val.pin_memory,
        prefetch_factor=3,
    )

    mnist_support_ds = MNISTGeometricDataset(
        root=cfg.data.mnist_root, # 需要在 yaml 定義
        template_dir=cfg.data.template_dir,
        train=False, # Test Set
        download=True,
        thicken=cfg.train.mnist_thicken,
        thicken_kernel_size=cfg.train.mnist_thicken_kernel_size,
        thicken_iterations=cfg.train.mnist_thicken_iterations,
    )
    # mnist_support_loader = DataLoader(
    #     mnist_support_ds, 
    #     batch_size=cfg.data.val.batch_size, 
    #     shuffle=False, 
    #     num_workers=4, 
    #     prefetch_factor=3,
    # )
    
    # Query Set: MNIST Train Set (60k)
    mnist_query_ds = MNISTGeometricDataset(
        root=cfg.data.mnist_root,
        template_dir=cfg.data.template_dir,
        train=True, # Train Set
        download=True,
        thicken=cfg.train.mnist_thicken,
        thicken_kernel_size=cfg.train.mnist_thicken_kernel_size,
        thicken_iterations=cfg.train.mnist_thicken_iterations,
    )
    # mnist_query_loader = DataLoader(
    #     mnist_query_ds, 
    #     batch_size=cfg.data.val.batch_size, 
    #     shuffle=False, 
    #     num_workers=4, 
    #     prefetch_factor=3,
    # )

    # 5. Model Initialization
    print("Initializing Model...")
    model = FullContrastiveModel(
        # stn_class=ShapeTransformationNetwork,
        stn_class=DeformationExtractor,
        stn_ckpt_path=cfg.model.stn.ckpt_path,
        encoder_config=cfg.model.encoder.encoder_config,
        encoder_ckpt_path=cfg.model.encoder.ckpt_path,
        cfg=cfg,
        device=device
    ).to(device)
    
    # 6. Optimizer & Scheduler
    # 只訓練 Encoder，STN 已凍結
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs,
        eta_min=1e-6
    )
    
    # 7. Loss
    criterion = InfoNCELoss(temperature=cfg.train.temperature)

    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity="gaga999-national-taiwan-university",
        )
        wandb.watch(model, log="all")
    
    # 8. Main Loop
    best_acc = 0.0
    print("Start Training...")
    
    for epoch in range(1, cfg.train.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Scheduler Step
        scheduler.step()
        
        log_dict = {
            "train/epoch_loss": train_loss, 
            "train/epoch_acc": train_acc, 
            "epoch": epoch
        }

        val_acc = validate(model, val_loader, device)
        log_dict["val/acc"] = val_acc
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Validation
        if epoch % cfg.train.check_val_every_n_epoch == 0:
            
            # Save Best Model
            # if val_acc > best_acc:
            #     best_acc = val_acc
            #     save_checkpoint(model, optimizer, epoch, cfg, filename="best_encoder.pth")
            #     print(f"New best model saved! ({best_acc:.4f})")

            mnist_query_loader = get_subset_dataloader(
                dataset=mnist_query_ds,
                ratio=cfg.train.mnist_query_ratio, # e.g., 0.1
                batch_size=cfg.data.val.batch_size,
                num_workers=cfg.data.val.num_workers
            )
            
            # 2. 動態建立 Support Loader
            # 建議 ratio 設為 1.0，但保留這個彈性
            mnist_support_loader = get_subset_dataloader(
                dataset=mnist_support_ds,
                ratio=cfg.train.mnist_support_ratio, # e.g., 0.1
                batch_size=cfg.data.val.batch_size,
                num_workers=cfg.data.val.num_workers
            )
            
            print(f"[Validation] MNIST Query size: {len(mnist_query_loader.dataset)}")
            
            # k_shot_results = evaluate_prototypical_k_shot(
            #     model,
            #     support_loader=mnist_support_loader, # Test Set (10k)
            #     query_loader=mnist_query_loader,     # Train Set (60k)
            #     device=device,
            #     k_list=[1, 2, 5, 10],
            #     rounds=250 # 每個 K 跑 5 次取平均
            # )
            
            # # Log to WandB
            # if wandb.run:
            #     for k, acc in k_shot_results.items():
            #         wandb.log({f"val/mnist_{k}shot_acc": acc, "epoch": epoch})
            
            # # 假設我們最關心 5-shot 的表現來決定 Best Model
            # current_metric = k_shot_results[5]
            # if current_metric > best_acc:
            #     best_acc = current_metric
            #     save_checkpoint(model, optimizer, epoch, cfg, filename="best_encoder.pth")

            k_shot_metrics = evaluate_class_analysis(
                model,
                support_loader=mnist_support_loader,
                query_loader=mnist_query_loader,
                device=device,
                k_list=[1, 2, 5, 10], # 指定要跑的 K
                rounds=250,             # 指定平均幾次
                max_classes_per_batch=20
            )
            
            # Log metrics
            if wandb.run:
                for k, acc in k_shot_metrics.items():
                    wandb.log({f"val/mnist_{k}shot_acc": acc, "epoch": epoch})
            
            # Save Best (e.g., based on 5-shot)
            current_acc = k_shot_metrics.get(2, 0.0)
            if current_acc > best_acc:
                best_acc = current_acc
                save_checkpoint(model, optimizer, epoch, cfg, filename="best_encoder.pth")

        # Regular Save
        if epoch % cfg.train.save_every_n_epoch == 0:
            save_checkpoint(model, optimizer, epoch, cfg, filename=f"epoch_{epoch}_encoder.pth")
            
        if wandb.run:
            wandb.log(log_dict)

if __name__ == "__main__":
    main()