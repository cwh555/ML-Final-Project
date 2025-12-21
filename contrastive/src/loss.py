import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features: (2*N, embedding_dim)
                Input shape should be the concatenation of two views:
                [view1_0, view1_1, ..., view2_0, view2_1, ...]
                So features[0] and features[N] are positive pairs.
        Returns:
            loss: Scalar tensor
        """
        device = features.device
        
        # 1. 確保特徵已經歸一化 (L2 Normalized)
        # 雖然 Encoder 最後一層有做，但這裡再做一次保險，且不影響梯度方向
        features = F.normalize(features, dim=-1)
        
        # 計算 Batch Size N
        batch_size = features.shape[0] // 2
        
        # 2. 計算相似度矩陣 (Cosine Similarity)
        # (2N, D) @ (D, 2N) -> (2N, 2N)
        similarity_matrix = torch.matmul(features, features.T)
        
        # 縮放 Temperature
        logits = similarity_matrix / self.temperature
        
        # 3. 建立 Label 和 Mask
        # 目標：對於第 i 個樣本 (View 1)，它的正樣本是 i + batch_size (View 2)
        # 對於第 i + batch_size 個樣本 (View 2)，它的正樣本是 i
        
        # 建立正樣本對的索引
        # labels: [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
        labels = torch.cat([
            torch.arange(batch_size, dtype=torch.long, device=device) + batch_size,
            torch.arange(batch_size, dtype=torch.long, device=device)
        ], dim=0)
        
        # 4. 移除對角線 (Self-similarity)
        # InfoNCE 的分母不包含自己與自己的相似度 (那是 1，會影響 LogSumExp)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        
        # 為了使用 CrossEntropyLoss，我們需要處理 mask
        # 標準做法是把對角線設為極小值 (-inf)，但 CrossEntropyLoss 需要正確的 target index
        # 這裡我們使用一個小技巧：直接用 CrossEntropyLoss，因為它會自動做 Softmax
        # 我們只需要讓對角線的值非常小，使其在 Softmax 後趨近於 0
        
        logits.masked_fill_(mask, -1e9)
        
        # 5. 計算 Loss
        # PyTorch 的 CrossEntropyLoss 包含了 LogSoftmax
        loss = F.cross_entropy(logits, labels)
        
        return loss
    

class ContrastiveAccuracy:
    """
    計算 Contrastive Learning 的 Instance Discrimination Accuracy (Top-1)。
    不繼承 nn.Module，作為普通的 Callable Object 使用。
    """
    def __init__(self):
        pass

    def __call__(self, features: torch.Tensor) -> float:
        """
        Args:
            features: (2N, D) Tensor
                Input shape should be the concatenation of two views:
                [view1_0, ... view1_N, view2_0, ... view2_N]
                Correct pairs are (i, i+N) and (i+N, i).
        
        Returns:
            accuracy: float (0.0 ~ 1.0)
        """
        # 確保不計算梯度，節省記憶體
        with torch.inference_mode():
            device = features.device
            batch_size = features.shape[0] // 2
            
            # 1. Normalization
            features = F.normalize(features, p=2, dim=1)
            
            # 2. Similarity Matrix (2N, 2N)
            logits = torch.matmul(features, features.T)
            
            # 3. Mask Diagonal (Self)
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
            logits.masked_fill_(mask, float('-inf'))
            
            # 4. Ground Truth Labels
            # i -> i + batch_size
            # i + batch_size -> i
            labels = torch.cat([
                torch.arange(batch_size, device=device) + batch_size,
                torch.arange(batch_size, device=device)
            ], dim=0)
            
            # 5. Predictions (Top-1)
            preds = torch.argmax(logits, dim=1)
            
            # 6. Accuracy
            correct = (preds == labels).float()
            accuracy = correct.mean().item() # 轉回 Python float
            
        return accuracy