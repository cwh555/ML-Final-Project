import os
import argparse
import time
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
from dataset import get_subset_dataloader, get_omniglot_loaders, get_quickdraw_loaders

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
from grid_model import FullContrastiveModel
# from old_model import FullContrastiveModel
# from noGaussian_model import FullContrastiveModel
from new_STNmodel import DeformationExtractor
from STNmodel import ShapeTransformationNetwork
from loss import InfoNCELoss, ContrastiveAccuracy
from utils import set_random_seed
from grid_eval_utils import evaluate_prototypical_k_shot, evaluate_class_analysis

# ==========================================
# Inference Time Measurement
# ==========================================

def measure_inference_time(model, dataloader, device, warmup_iters=50, test_iters=300):
    """
    使用 torch.cuda.Event 精確測量模型推論時間。
    
    測量內容：
    - Input GPU transfer
    - Backbone Feature Extraction  
    - Deformation Network
    - Grid Sample / Warping
    - Feature normalization
    
    不測量：
    - DataLoader 硬碟讀取時間
    - 資料前處理時間
    - 視覺化/存檔/Print log 時間
    
    Args:
        model: 要測量的模型
        dataloader: 資料載入器 (用來取得真實輸入格式)
        device: GPU device
        warmup_iters: GPU warm-up 次數
        test_iters: 正式測量次數
        
    Returns:
        dict: 包含 mean_time_ms, std_time_ms, fps, batch_size, total_samples
    """
    model.eval()
    
    # 取得一個 batch 來確定輸入格式
    sample_batch = next(iter(dataloader))
    sample_input = sample_batch['image'].to(device)
    batch_size = sample_input.shape[0]
    
    print(f"\n{'='*60}")
    print("[Inference Time Measurement]")
    print(f"{'='*60}")
    print(f"  Input shape: {sample_input.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Warmup iterations: {warmup_iters}")
    print(f"  Test iterations: {test_iters}")
    
    # 1. GPU Warm-up (非常重要！前幾次執行會比較慢)
    print(f"\n-> GPU Warm-up ({warmup_iters} iterations)...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(sample_input)
    torch.cuda.synchronize()
    print("   Warm-up complete.")
    
    # 2. 正式測量
    print(f"-> Measuring inference time ({test_iters} iterations)...")
    
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = np.zeros(test_iters)
    
    with torch.no_grad():
        for rep in range(test_iters):
            # 確保輸入在 GPU 上 (模擬真實推論情境，資料已在 GPU)
            # 如果要測量包含 CPU->GPU transfer，可以改成每次重新 .to(device)
            
            starter.record()
            
            # --- 核心推論部分 ---
            output = model(sample_input)
            # -------------------
            
            ender.record()
            
            # 等待 GPU 完成
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 單位: 毫秒 (ms)
            timings[rep] = curr_time
    
    # 3. 統計結果
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    median_time = np.median(timings)
    
    # Per-sample time (除以 batch_size)
    per_sample_mean = mean_time / batch_size
    per_sample_std = std_time / batch_size
    
    # FPS (throughput)
    fps = 1000.0 / per_sample_mean  # 1000ms / per_sample_ms
    
    # 4. 輸出結果
    print(f"\n{'='*60}")
    print("[Inference Time Results]")
    print(f"{'='*60}")
    print(f"  Hardware: {torch.cuda.get_device_name(device)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Test iterations: {test_iters}")
    print(f"")
    print(f"  [Per-Batch Statistics]")
    print(f"    Mean: {mean_time:.3f} ms")
    print(f"    Std:  {std_time:.3f} ms")
    print(f"    Min:  {min_time:.3f} ms")
    print(f"    Max:  {max_time:.3f} ms")
    print(f"    Median: {median_time:.3f} ms")
    print(f"")
    print(f"  [Per-Sample Statistics]")
    print(f"    Mean: {per_sample_mean:.3f} ms/sample")
    print(f"    Std:  {per_sample_std:.3f} ms/sample")
    print(f"")
    print(f"  [Throughput]")
    print(f"    FPS: {fps:.2f} samples/sec")
    print(f"    Batch throughput: {1000.0/mean_time:.2f} batches/sec")
    print(f"{'='*60}")
    
    return {
        'batch_size': batch_size,
        'test_iters': test_iters,
        'per_batch_mean_ms': mean_time,
        'per_batch_std_ms': std_time,
        'per_batch_min_ms': min_time,
        'per_batch_max_ms': max_time,
        'per_sample_mean_ms': per_sample_mean,
        'per_sample_std_ms': per_sample_std,
        'fps': fps,
        'device_name': torch.cuda.get_device_name(device),
    }


# ==========================================
# Training & Validation Functions
# ==========================================


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

def validate(model, dataloader, device):
    model.eval() # Encoder Eval
    
    total_acc = 0
    num_batches = 0
    total_loss = 0.0
    metric_calc = ContrastiveAccuracy()
    criterion = InfoNCELoss()
    
    # 使用 inference_mode 加速並節省記憶體
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            log_wandb_images(batch, batch_idx, prefix="val")
            images = batch['image'].to(device)
            # names = np.array(batch['template_names']).transpose(2, 0, 1)
            # names = names.reshape(-1, names.shape[-1])
            # weights_boundary = batch['weights_boundary'].to(device)
            # wb_view1 = weights_boundary[:, 0]
            # wb_view2 = weights_boundary[:, 1]
            # concat_wb = torch.cat([wb_view1, wb_view2], dim=0)
            
            # Validation 也需要 two views 才能算 Accuracy
            view1 = images[:, 0]
            view2 = images[:, 1]
            concat_input = torch.cat([view1, view2], dim=0)
            
            embeddings = model(concat_input)
            # embeddings = model(concat_input, template_names=names)
            # embeddings = model(concat_input, weight_boundary=concat_wb, template_names=None)
            
            acc = metric_calc(embeddings)
            loss = criterion(embeddings)
            total_acc += acc
            total_loss += loss.item()
            num_batches += 1
            
    return total_acc / num_batches, total_loss / num_batches
# ==========================================
# Main Execution
# ==========================================

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Train Geometric Contrastive Model")
    parser.add_argument("--config", type=str, default="default", help="Config file name in configs/ (without .yaml)")
    parser.add_argument("--eval_rounds", type=int, default=5, help="Number of evaluation rounds (each round randomly samples classes)")
    parser.add_argument("--time", action="store_true", help="Output timing information for each dataset evaluation")
    args = parser.parse_args()
    
    config_path = f"configs/{args.config}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    
    # 2. Setup
    device = torch.device(cfg.project.device)
    set_random_seed(cfg.project.seed)

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

    # # 4. Datasets & DataLoaders
    
    # # Val
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
    mnist_support_loader = DataLoader(
        mnist_support_ds, 
        batch_size=cfg.data.val.batch_size, 
        shuffle=False, 
        num_workers=4, 
        prefetch_factor=3,
    )
    
    # # Query Set: MNIST Train Set (60k)
    mnist_query_ds = MNISTGeometricDataset(
        root=cfg.data.mnist_root,
        template_dir=cfg.data.template_dir,
        train=True, # Train Set
        download=True,
        thicken=cfg.train.mnist_thicken,
        thicken_kernel_size=cfg.train.mnist_thicken_kernel_size,
        thicken_iterations=cfg.train.mnist_thicken_iterations,
    )
    mnist_query_loader = DataLoader(
        mnist_query_ds, 
        batch_size=cfg.data.val.batch_size, 
        shuffle=False, 
        num_workers=4, 
        prefetch_factor=3,
    )

    # 5. Model Initialization
    print("Initializing Model...")
    model = FullContrastiveModel(
        stn_class=ShapeTransformationNetwork,
        # stn_class=DeformationExtractor,
        stn_ckpt_path=cfg.model.stn.ckpt_path,
        encoder_config=cfg.model.encoder.encoder_config,
        encoder_ckpt_path=cfg.model.encoder.eval_path,
        cfg=cfg,
        device=device
    ).to(device)
    
    # 6. Optimizer & Scheduler
    # 只訓練 Encoder，STN 已凍結

    # 7. Loss

    if cfg.wandb.enable:
        wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity="gaga999-national-taiwan-university",
        )
        wandb.watch(model, log="all")
    
    # 8. Main Loop

    val_acc, val_loss = validate(model, val_loader, device)
    log_dict = {
        "val/loss": val_loss,
        "val/acc": val_acc,
    }
    print(f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        
            
    mnist_support_size = len(mnist_support_loader.dataset)
    mnist_query_size = len(mnist_query_loader.dataset)
    mnist_total_samples = mnist_support_size + mnist_query_size
    print(f"[Validation] MNIST Support size: {mnist_support_size}")
    print(f"[Validation] MNIST Query size: {mnist_query_size}")
    
    # # mnist_start_time = time.time()
    # # k_shot_results = evaluate_prototypical_k_shot(
    # #     model,
    # #     support_loader=mnist_support_loader, # Test Set (10k)
    # #     query_loader=mnist_query_loader,     # Train Set (60k)
    # #     device=device,
    # #     k_list=[1, 2, 5, 10],
    # #     rounds=250 # 每個 K 跑 5 次取平均
    # # )
    # # mnist_elapsed_time = time.time() - mnist_start_time
    
    # # if args.time:
    # #     print("\n" + "="*60)
    # #     print("[Timing] MNIST Evaluation (Prototypical K-Shot)")
    # #     print("="*60)
    # #     print(f"  Dataset: Support={mnist_support_size}, Query={mnist_query_size}")
    # #     print(f"  Model inference samples: {mnist_total_samples} (each sample runs through model once)")
    # #     print(f"  K-shot settings: K={[1, 2, 5, 10]}, rounds={250}")
    # #     print(f"  Total time (feature extraction + k-shot eval): {mnist_elapsed_time:.2f} sec ({mnist_elapsed_time/60:.2f} min)")
    # #     print(f"  Avg throughput: {mnist_total_samples/mnist_elapsed_time:.2f} samples/sec")
    # #     print("="*60)
    
    # # # Log to WandB
    # # if wandb.run:
    # #     for k, acc in k_shot_results.items():
    # #         wandb.log({f"val/mnist_{k}shot_acc": acc, "epoch": epoch})
    
    # # 假設我們最關心 5-shot 的表現來決定 Best Model
    # current_metric = k_shot_results[5]
    # if current_metric > best_acc:
    #     best_acc = current_metric
    #     save_checkpoint(model, optimizer, epoch, cfg, filename="best_encoder.pth")

    # mnist_ca_start_time = time.time()
    k_shot_metrics = evaluate_class_analysis(
        model,
        support_loader=mnist_support_loader,
        query_loader=mnist_query_loader,
        device=device,
        k_list=[1, 2, 5, 10], # 指定要跑的 K
        rounds=250,             # 指定平均幾次
        max_classes_per_batch=20,
        # max_classes_per_batch=5
        eval_rounds=args.eval_rounds  # 評估回合數
    )
    # mnist_ca_elapsed_time = time.time() - mnist_ca_start_time
    
    # if args.time:
    #     print("\n" + "="*60)
    #     print("[Timing] MNIST Evaluation (Class Analysis)")
    #     print("="*60)
    #     print(f"  Dataset: Support={mnist_support_size}, Query={mnist_query_size}")
    #     print(f"  Model inference samples: {mnist_total_samples} (each sample runs through model once)")
    #     print(f"  K-shot settings: K={[1, 2, 5, 10]}, rounds={250}, eval_rounds={args.eval_rounds}")
    #     print(f"  Total time (feature extraction + k-shot eval): {mnist_ca_elapsed_time:.2f} sec ({mnist_ca_elapsed_time/60:.2f} min)")
    #     print(f"  Avg throughput: {mnist_total_samples/mnist_ca_elapsed_time:.2f} samples/sec")
    #     print("="*60)
    
    # # Log metrics
    # if wandb.run:
    #     for k, acc in k_shot_metrics.items():
    #         wandb.log({f"val/mnist_{k}shot_acc": acc})

    omni_support_loader, omni_query_loader = get_omniglot_loaders(cfg)
    omni_support_size = len(omni_support_loader.dataset)
    omni_query_size = len(omni_query_loader.dataset)
    omni_total_samples = omni_support_size + omni_query_size
    print(f"[Validation] Omniglot Support size: {omni_support_size}")
    print(f"[Validation] Omniglot Query size: {omni_query_size}")
    
    # ==========================================
    # 精確推論時間測量 (使用 torch.cuda.Event)
    # ==========================================
    if args.time:
        print("\n" + "="*60)
        print("PRECISE INFERENCE TIME MEASUREMENT")
        print("="*60)
        inference_stats = measure_inference_time(
            model=model,
            dataloader=omni_support_loader,  # 使用 Omniglot support loader 取得真實輸入
            device=device,
            warmup_iters=50,
            test_iters=300
        )
        
        # Log to WandB
        if wandb.run:
            wandb.log({
                "timing/per_sample_mean_ms": inference_stats['per_sample_mean_ms'],
                "timing/per_sample_std_ms": inference_stats['per_sample_std_ms'],
                "timing/fps": inference_stats['fps'],
                "timing/batch_size": inference_stats['batch_size'],
                "timing/device": inference_stats['device_name'],
            })
    
    omni_start_time = time.time()
    k_shot_metrics_omni = evaluate_class_analysis(
        model,
        support_loader=omni_support_loader,
        query_loader=omni_query_loader,
        device=device,
        k_list=[1, 2, 5, 10], # 指定要跑的 K
        rounds=50,             # 指定平均幾次
        max_classes_per_batch=20,
        # max_classes_per_batch=5,
        eval_rounds=args.eval_rounds  # 評估回合數
    )
    omni_elapsed_time = time.time() - omni_start_time
    
    if args.time:
        print("\n" + "="*60)
        print("[Wall-Clock Time] Omniglot Full Evaluation Pipeline")
        print("="*60)
        print(f"  (Note: This includes DataLoader I/O, not pure inference)")
        print(f"  Dataset: Support={omni_support_size}, Query={omni_query_size}")
        print(f"  Total samples processed: {omni_total_samples}")
        print(f"  K-shot settings: K={[1, 2, 5, 10]}, rounds={50}, eval_rounds={args.eval_rounds}")
        print(f"  Total wall-clock time: {omni_elapsed_time:.2f} sec ({omni_elapsed_time/60:.2f} min)")
        print(f"  Effective throughput (incl. I/O): {omni_total_samples/omni_elapsed_time:.2f} samples/sec")
        print("="*60)
    
    if wandb.run:
        for k, acc in k_shot_metrics_omni.items():
            wandb.log({f"val/omniglot_{k}shot_acc": acc})
    
    # ==========================================
    # # QuickDraw Evaluation
    # # ==========================================
    if hasattr(cfg.data, 'quickdraw_root') and cfg.data.quickdraw_root:
        print("\n" + "="*50)
        print("QuickDraw Evaluation")
        print("="*50)
        qd_support_loader, qd_query_loader = get_quickdraw_loaders(cfg)
        qd_support_size = len(qd_support_loader.dataset)
        qd_query_size = len(qd_query_loader.dataset)
        qd_total_samples = qd_support_size + qd_query_size
        print(f"[Validation] QuickDraw Support size: {qd_support_size}")
        print(f"[Validation] QuickDraw Query size: {qd_query_size}")

        if args.time:
            print("\n" + "="*60)
            print("PRECISE INFERENCE TIME MEASUREMENT")
            print("="*60)
            inference_stats = measure_inference_time(
                model=model,
                dataloader=qd_support_loader,  # 使用 Omniglot support loader 取得真實輸入
                device=device,
                warmup_iters=50,
                test_iters=300
            )
            
            # Log to WandB
            if wandb.run:
                wandb.log({
                    "timing/per_sample_mean_ms": inference_stats['per_sample_mean_ms'],
                    "timing/per_sample_std_ms": inference_stats['per_sample_std_ms'],
                    "timing/fps": inference_stats['fps'],
                    "timing/batch_size": inference_stats['batch_size'],
                    "timing/device": inference_stats['device_name'],
                })

        
        qd_start_time = time.time()
        k_shot_metrics_qd = evaluate_class_analysis(
            model,
            support_loader=qd_support_loader,
            query_loader=qd_query_loader,
            device=device,
            k_list=[1, 2, 5, 10],
            rounds=50,
            max_classes_per_batch=20,
            eval_rounds=args.eval_rounds  # 評估回合數
        )
        qd_elapsed_time = time.time() - qd_start_time
        
        if args.time:
            print("\n" + "="*60)
            print("[Wall-Clock Time] QuickDraw Full Evaluation Pipeline")
            print("="*60)
            print(f"  (Note: This includes DataLoader I/O, not pure inference)")
            print(f"  Dataset: Support={qd_support_size}, Query={qd_query_size}")
            print(f"  Total samples processed: {qd_total_samples}")
            print(f"  K-shot settings: K={[1, 2, 5, 10]}, rounds={50}, eval_rounds={args.eval_rounds}")
            print(f"  Total wall-clock time: {qd_elapsed_time:.2f} sec ({qd_elapsed_time/60:.2f} min)")
            print(f"  Effective throughput (incl. I/O): {qd_total_samples/qd_elapsed_time:.2f} samples/sec")
            print("="*60)
        
        if wandb.run:
            for k, acc in k_shot_metrics_qd.items():
                wandb.log({f"val/quickdraw_{k}shot_acc": acc})
        
        print("QuickDraw k-shot results:")
        for k, acc in k_shot_metrics_qd.items():
            print(f"  {k}-shot: {acc:.4f}")
    
    if wandb.run:
        wandb.log(log_dict)


if __name__ == "__main__":
    main()