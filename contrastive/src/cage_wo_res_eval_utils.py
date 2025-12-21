# src/eval_utils.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import wandb

@torch.inference_mode()
def extract_all_features(model, dataloader, device, log_prefix="extract"):
    """
    提取特徵並記錄圖片到 WandB
    log_prefix: 用來區分是 support 還是 query 的圖片
    """
    model.eval()
    all_feats = []
    all_labels = []
    
    # 使用 enumerate 以便控制 logging 頻率
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting ({log_prefix})")):
        imgs = batch['image'].to(device) # (B, 5, 4, H, W)
        labels = batch['label'].to(device)
        weights_boundary = batch['weights_boundary'].to(device)
        
        # --- WandB Logging (Log 1 image per batch) ---
        if wandb.run and batch_idx % 1 == 0: # 每個 batch 都 log
            # 取出 Batch 中第一張圖 (Index 0)
            # 取出 Stack 中第一個 Template (Index 0) (這通常是最原本的變形前視角)
            # 取出 RGB Channels (:3)
            # img_vis shape: (3, H, W)
            img_vis = imgs[0, 0, :3].detach().cpu()
            
            # 也可以把 mask 取出來看 (channel 3 是 mask 嗎？不，dataset 是 cat([img, template])
            # 所以 channel 3 是 template mask。
            # 如果你想看原始圖片的 mask，dataset 回傳字`典裡有 'mask' key，但這裡 dataloader 有沒有回傳看你的 collate
            # 不過光看 img_vis 應該就能看出有沒有變粗了
            
            wandb.log({
                f"debug/{log_prefix}_batch_sample": wandb.Image(
                    img_vis, 
                    caption=f"Batch {batch_idx} Label: {labels[0].item()}"
                )
            })
        # ---------------------------------------------

        # Forward
        feats = model(imgs, weight_boundary=weights_boundary, template_names=None)
        # feats = model(imgs)
        feats = F.normalize(feats, p=2, dim=1)
        
        all_feats.append(feats)
        all_labels.append(labels)
        
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

def _calc_4_stats(data_list):
    """
    Helper: 計算 [Mean, Max, Median, 75%]
    """
    arr = np.array(data_list)
    return [
        np.mean(arr),
        np.max(arr),
        np.median(arr),
        np.percentile(arr, 75)
    ]

def evaluate_prototypical_k_shot(
    model, 
    support_loader, # 來源: MNIST Test Set (10k)
    query_loader,   # 來源: MNIST Train Set (60k)
    device,
    k_list=[1, 2, 5, 10], 
    rounds=5
):
    """
    Args:
        support_loader: 提供 K-shot 樣本的池子 (Test Set)
        query_loader: 用來評估準確率的查詢集 (Train Set)
        rounds: 每個 K 重複抽樣幾次取平均
    """
    print(f"\n[Validation] Starting Prototypical K-Shot Evaluation...")
    
    # 1. 預先提取所有特徵 (Feature Caching)
    # 這一步最花時間，但只需做一次
    print("-> Extracting Support Pool (MNIST Test Set)...")
    support_feats, support_labels = extract_all_features(model, support_loader, device)
    
    print("-> Extracting Query Set (MNIST Train Set)...")
    query_feats, query_labels = extract_all_features(model, query_loader, device)
    
    # 2. 整理 Support Indices (按類別分組)
    # class_indices[label] = [idx1, idx2, ...]
    class_indices = defaultdict(list)
    for idx, label in enumerate(support_labels.cpu().numpy()):
        class_indices[label].append(idx)
        
    results = {}
    
    # 3. 開始迴圈評估
    for k in k_list:
        round_accs = []
        
        for r in range(rounds):
            # --- Build Prototypes (Support Set) ---
            prototypes = []
            valid_classes = []
            
            for cls in range(10): # MNIST 0-9
                indices = class_indices[cls]
                if len(indices) < k:
                    # 理論上 MNIST Test 每類都有 ~1000 張，不會不夠
                    continue
                
                # 隨機選 K 張
                selected_indices = random.sample(indices, k)
                
                # 取出特徵 (K, D)
                selected_feats = support_feats[selected_indices]
                
                # [核心邏輯] 取平均變 Prototype (1, D)
                # 這裡再次 Normalize 是 Prototypical Networks 的常見技巧，
                # 讓 Prototype 也在單位球面上
                proto = selected_feats.mean(dim=0)
                proto = F.normalize(proto, p=2, dim=0)
                
                prototypes.append(proto)
                valid_classes.append(cls)
                
            # Stack -> (10, D)
            prototypes = torch.stack(prototypes).to(device)
            
            # --- Evaluate (Query Set) ---
            # Cosine Similarity: Query(N, D) @ Prototypes(D, 10) -> (N, 10)
            sim_matrix = torch.matmul(query_feats, prototypes.T)
            
            # Prediction
            # index 0 對應 valid_classes[0] (通常就是 label 0)
            pred_indices = torch.argmax(sim_matrix, dim=1)
            
            # 因為 valid_classes 剛好是 0-9 排序，pred_indices 就是預測的數字
            # 如果不是 0-9 排序，需要 map 回去
            
            correct = (pred_indices == query_labels).float().sum()
            acc = correct / len(query_labels)
            round_accs.append(acc.item())
            
        # 計算 Mean & Std
        mean_acc = np.mean(round_accs)
        std_acc = np.std(round_accs)
        max_acc = np.max(round_accs)
        min_acc = np.min(round_accs)
        results[k] = mean_acc
        
        print(f"  K={k:2d} | Mean Acc: {mean_acc:.4f} (+/- {std_acc:.4f})  | Max Acc: {max_acc:.4f} | Min Acc: {min_acc:.4f} | Rounds: {rounds}")
        
    return results

def _generate_analysis_table(classes, true_labels, pred_labels):
    """
    Helper: 生成分析表格數據
    輸入的 true_labels 和 pred_labels 可以是多個 round 累積下來的結果
    """
    y_true = true_labels.cpu().numpy()
    y_pred = pred_labels.cpu().numpy()
    classes = np.array(classes)
    n_classes = len(classes)
    
    label_to_idx = {label: i for i, label in enumerate(classes)}
    
    # 初始化混淆矩陣
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    total_samples = len(y_true)
    
    # 計算各項指標 (避免除以 0)
    row_sums = cm.sum(axis=1)
    class_acc = np.divide(cm.diagonal(), row_sums, out=np.zeros_like(cm.diagonal(), dtype=float), where=row_sums!=0)
    
    input_dist = row_sums / total_samples
    col_sums = cm.sum(axis=0)
    output_dist = col_sums / total_samples
    
    # Row Normalized CM (Input Class Output Distribution)
    cm_norm = np.divide(cm, row_sums[:, None], out=np.zeros_like(cm, dtype=float), where=row_sums[:, None]!=0)

    # 組裝表格
    columns = ["Metric"] + [str(c) for c in classes]
    table_data = []
    
    table_data.append(["Accuracy"] + [f"{v:.2%}" for v in class_acc])
    table_data.append(["Input Prob"] + [f"{v:.2%}" for v in input_dist])
    table_data.append(["Output Prob"] + [f"{v:.2%}" for v in output_dist])
    
    for i, cls in enumerate(classes):
        row_name = f"True {cls} Dist"
        row_vals = [f"{v:.2%}" for v in cm_norm[i]]
        table_data.append([row_name] + row_vals)
        
    return columns, table_data

def evaluate_class_analysis(
    model, 
    support_loader, 
    query_loader,   
    device,
    k_list=[1, 2, 5, 10], 
    rounds=5,
    max_classes_per_batch=10,
    eval_rounds=1  # 新增：評估回合數，每個回合隨機抽取 class
):
    """
    K-Shot 分析函式。
    針對每個 K，重複 Rounds 次，並針對第一個 Chunk 生成分析報表。
    eval_rounds: 整體評估回合數，每個回合會隨機 shuffle classes
    """
    print(f"\n[Analysis] Starting Multi-K Analysis: {k_list} (Rounds={rounds}, EvalRounds={eval_rounds})...")
    
    # 1. 提取特徵
    print("-> Extracting Support Features...")
    support_feats, support_labels = extract_all_features(model, support_loader, device)
    print("-> Extracting Query Features...")
    query_feats, query_labels = extract_all_features(model, query_loader, device)
    
    # 2. 檢查類別一致性
    unique_support = torch.unique(support_labels).cpu().numpy()
    unique_query = torch.unique(query_labels).cpu().numpy()
    set_support = set(unique_support)
    set_query = set(unique_query)
    
    if set_support != set_query:
        raise ValueError(f"Class Mismatch! Support: {set_support}, Query: {set_query}")
        
    all_classes = sorted(list(set_support))
    num_classes = len(all_classes)
    
    # 建立索引映射
    support_indices_map = defaultdict(list)
    for idx, label in enumerate(support_labels.cpu().numpy()):
        support_indices_map[label].append(idx)
        
    query_indices_map = defaultdict(list)
    for idx, label in enumerate(query_labels.cpu().numpy()):
        query_indices_map[label].append(idx)

    # 3. 分批處理 (Chunking) - 加入 eval_rounds 迴圈
    global_stats_collector = defaultdict(lambda: defaultdict(list))
    
    total_chunks = (num_classes + max_classes_per_batch - 1) // max_classes_per_batch
    
    for eval_round in range(eval_rounds):
        # 每個 eval_round 重新隨機 shuffle classes
        shuffled_classes = all_classes.copy()
        random.shuffle(shuffled_classes)
        
        print(f"\n{'='*60}")
        print(f"[Eval Round {eval_round+1}/{eval_rounds}]")
        print(f"{'='*60}")
        
        for i in range(0, num_classes, max_classes_per_batch):
            chunk_classes = shuffled_classes[i : i + max_classes_per_batch]
            chunk_idx = i // max_classes_per_batch
            print(f"\n--- Chunk {chunk_idx+1}/{total_chunks}: Classes {chunk_classes[:5]}{'...' if len(chunk_classes) > 5 else ''} ---")
        
            # 準備 Query Data (對於這個 Chunk 是固定的)
            chunk_query_feats_list = []
            chunk_query_labels_list = []
            for cls in chunk_classes:
                indices = query_indices_map[cls]
                chunk_query_feats_list.append(query_feats[indices])
                chunk_query_labels_list.append(query_labels[indices])
                
            if not chunk_query_feats_list: continue
            
            # (Total_Q_Chunk, D)
            chunk_query_feats = torch.cat(chunk_query_feats_list, dim=0)
            chunk_query_labels = torch.cat(chunk_query_labels_list, dim=0)
            
            # === Loop over K ===
            for k in k_list:
                round_accs = []
                
                # 用來累積 rounds 次的所有預測，以便生成穩定的表格
                all_rounds_true = []
                all_rounds_pred = []
                
                # === Loop over Rounds ===
                for r in range(rounds):
                    # A. Build Prototypes
                    prototypes = []
                    valid_chunk_classes = []
                    
                    for cls in chunk_classes:
                        indices = support_indices_map[cls]
                        if len(indices) < k: continue # Skip if not enough samples
                        
                        selected_indices = random.sample(indices, k)
                        selected_feats = support_feats[selected_indices]
                        
                        proto = selected_feats.mean(dim=0)
                        proto = F.normalize(proto, p=2, dim=0)
                        prototypes.append(proto)
                        valid_chunk_classes.append(cls)
                    
                    if not prototypes: break
                    
                    prototypes = torch.stack(prototypes).to(device)
                    
                    # B. Predict
                    sim_matrix = torch.matmul(chunk_query_feats, prototypes.T)
                    pred_indices = torch.argmax(sim_matrix, dim=1)
                    
                    # Map back to original labels
                    pred_labels = torch.tensor(
                        [valid_chunk_classes[idx] for idx in pred_indices.cpu().numpy()]
                    ).to(device)
                    
                    # C. Metrics
                    correct = (pred_labels == chunk_query_labels).float().sum()
                    acc = correct / len(chunk_query_labels)
                    
                    round_accs.append(acc.item())
                    
                    # 收集用於表格的數據
                    all_rounds_true.append(chunk_query_labels)
                    all_rounds_pred.append(pred_labels)
                
                # === End of Rounds (Aggregation for this K) ===
                if not round_accs: continue
                
                chunk_stats = _calc_4_stats(round_accs) # [Mean, Max, Median, 75%]
                
                # 存入收集器
                global_stats_collector[k]['mean'].append(chunk_stats[0])
                global_stats_collector[k]['max'].append(chunk_stats[1])
                global_stats_collector[k]['median'].append(chunk_stats[2])
                global_stats_collector[k]['p75'].append(chunk_stats[3])
                
                # === Table Generation (Only First Chunk of First Eval Round) ===
                if i == 0 and eval_round == 0:
                    # 把所有 rounds 的資料串接起來，樣本數 = Q * rounds
                    flat_true = torch.cat(all_rounds_true, dim=0)
                    flat_pred = torch.cat(all_rounds_pred, dim=0)
                    
                    columns, table_data = _generate_analysis_table(
                        valid_chunk_classes, 
                        flat_true, 
                        flat_pred
                    )
                    
                    # 1. Terminal Print (Only for k_list[1] if exists, else k_list[0])
                    target_k_idx = 1 if len(k_list) > 1 else 0
                    if k == k_list[target_k_idx]:
                        print(f"\n[Analysis Table (Chunk 0, K={k}, Avg of {rounds} rounds)]")
                        df_display = pd.DataFrame(table_data, columns=columns)
                        print(df_display.to_string(index=False))
                        print("-" * 50)

                    # 2. WandB Log (All K)
                    if wandb.run:
                        wandb_table = wandb.Table(columns=columns, data=table_data)
                        wandb.log({f"analysis/table_k{k}": wandb_table})

    # 計算並回傳最終結果
    # final_metrics = {}
    # print("\n[Final Results]")
    # for k in k_list:
    #     if k in global_results:
    #         mean_acc = np.mean(global_results[k])
    #         final_metrics[k] = mean_acc
    #         print(f"  K={k:<2}: {mean_acc:.4f}")
    #     else:
    #         final_metrics[k] = 0.0

    # return final_metrics
    final_metrics = {} # 用來回傳給 main 做 checkpoint 判斷 (通常用 Mean-Mean)
    
    print("\n" + "="*60)
    print(f"[Final Analysis] Aggregated over {total_chunks} Chunks x {eval_rounds} EvalRounds")
    print("With 95% Confidence Interval (Mean ± 1.96 * SE)")
    print("="*60)

    stat_names = ['Mean', 'Max', 'Median', 'P75']
    
    for k in k_list:
        if k not in global_stats_collector:
            final_metrics[k] = 0.0
            continue
            
        print(f"\n>>> K = {k}")
        
        # 準備 4x4 矩陣數據
        # Rows: Chunk-Mean, Chunk-Max, Chunk-Median, Chunk-P75
        # Cols: Global-Mean, Global-Max, Global-Median, Global-P75
        matrix_data = []
        
        # 針對每一種 Chunk 統計指標 (例如: 所有 Chunks 的 Mean 列表)
        for row_stat in stat_names: # iterate keys: 'mean', 'max', 'median', 'p75'
            key = row_stat.lower()
            if key == 'p75': key = 'p75' # 保持 key 一致
            
            values_from_chunks = global_stats_collector[k][key]
            
            if not values_from_chunks:
                matrix_data.append([0.0]*4)
                continue
                
            # 再算一次 4 種統計 (Global Aggregation)
            global_aggs = _calc_4_stats(values_from_chunks)
            matrix_data.append(global_aggs)
            
        # 建立 DataFrame
        df_4x4 = pd.DataFrame(
            matrix_data, 
            index=[f"Chunk {s}" for s in stat_names],
            columns=[f"Global {s}" for s in stat_names]
        )
        
        # 格式化顯示 (百分比)
        print(df_4x4.applymap(lambda x: f"{x:.2%}"))
        
        # 計算並輸出 Mean ± 95% CI
        mean_values = global_stats_collector[k]['mean']
        if len(mean_values) > 1:
            arr = np.array(mean_values)
            mean_acc = np.mean(arr)
            std_acc = np.std(arr, ddof=1)  # 使用 sample std
            n = len(arr)
            # 95% CI: mean ± 1.96 * (std / sqrt(n))
            ci_95 = 1.96 * std_acc / np.sqrt(n)
            print(f"\n  [Summary] Mean: {mean_acc:.2%} ± {ci_95:.2%} (95% CI, N={n})")
        else:
            mean_acc = mean_values[0] if mean_values else 0.0
            print(f"\n  [Summary] Mean: {mean_acc:.2%} (N=1, no CI)")
        
        # 記錄主要指標 (通常取 Mean of Means)
        final_metrics[k] = df_4x4.loc["Chunk Mean", "Global Mean"]
        
        # Log to WandB (4x4 Table)
        if wandb.run:
            # 轉成 wandb table
            # 需要把 index 變成一個 column 才能完美顯示
            df_log = df_4x4.reset_index().rename(columns={'index': 'Chunk Stat'})
            wandb.log({f"analysis/stats_matrix_k{k}": wandb.Table(dataframe=df_log)})

    return final_metrics