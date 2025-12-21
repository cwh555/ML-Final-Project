import os
import glob
import numpy as np
import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torchvision.transforms.functional import to_tensor, resize
from torchvision.transforms import Compose
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Dict, List, Optional, Union, Callable
from PIL import Image
from utils import set_random_seed
from torchvision.datasets import MNIST
from torchvision.datasets import Omniglot
import cv2
import grid_utils as utils


# ==========================================
# 1. Transform Infrastructure (基礎建設)
# ==========================================

class BaseTransform:
    def __init__(self, keys: List[str], *args, **kwargs):
        """
        keys: 需要被處理的字典鍵值列表
        """
        self.keys = keys
        self._parse_var(*args, **kwargs)

    def _parse_var(self, *args, **kwargs):
        pass

    def _process(self, data, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        # 針對這類 Transform，通常是對字典內的每一個 key 獨立處理
        # 但有些操作 (如 Load) 可能需要特殊處理，由子類別決定
        return self.apply_transform(data, *args, **kwargs)
    
    def apply_transform(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        # 預設行為：遍歷 keys 並逐一處理
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], *args, **kwargs)
            else:
                # 某些情況允許 key 不存在 (例如只有 train 有 label)
                pass 
        return data

# ==========================================
# 2. Specific Transforms (具體功能)
# ==========================================

class LoadGeometricData(BaseTransform):
    """
    專門讀取 [image, mask, label] 格式的 .npy 檔案
    輸入: 包含檔案路徑的字典 (例如 key='data_path')
    輸出: 將 image, mask, label 展開到字典中
    """
    def __init__(self, load_key: str, output_keys: List[str]=['image', 'mask', 'label']):
        super().__init__(keys=[load_key])
        self.load_key = load_key
        self.output_keys = output_keys # [image_key, mask_key, label_key]

    def apply_transform(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        path = data[self.load_key]
        
        # Load npy file: [image, mask, label]
        try:
            # content[0]: image (3, 256, 256)
            # content[1]: mask (256, 256)
            # content[2]: label (int)
            content = np.load(path, allow_pickle=True)
            
            data[self.output_keys[0]] = content[0]
            data[self.output_keys[1]] = content[1]
            data[self.output_keys[2]] = content[2]
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")
            
        return data

class ToTensord(BaseTransform):
    """將 numpy array 或數值轉為 Tensor"""
    def _process(self, item, *args, **kwargs):
        if isinstance(item, (np.ndarray, list)):
            item = torch.from_numpy(np.array(item))
            # 確保 Image 是 Float, Label 是 Long
            if item.dtype == torch.float64:
                item = item.float()
            elif item.dtype == torch.int32 or item.dtype == torch.int64:
                item = item.long()
        elif isinstance(item, (int, float)):
            item = torch.tensor(item)
        return item

class ResizeImaged(BaseTransform):
    def __init__(self, keys, size, **kwargs):
        super(ResizeImaged, self).__init__(keys, **kwargs)
        self.size = size

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        return resize(data, self.size)

class RandomGeoAugment(BaseTransform):
    """
    隨機平移與旋轉 (+- 15度)
    必須確保 image 和 mask 做完全一樣的變換
    """
    def __init__(self, keys: List[str], translate_range=0.1, degrees=15):
        # keys 通常是 ['image', 'mask']，不包含 label
        super().__init__(keys=keys)
        self.translate_range = translate_range
        self.degrees = degrees

    def apply_transform(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        # 1. 決定隨機參數 (所有 keys 共用同一組參數)
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Translate 需要 (dx, dy) 像素單位或是比例
        # 這裡簡化為固定比例的隨機位移
        # 假設 keys[0] 是 image，用它來抓尺寸
        img_h, img_w = data[self.keys[0]].shape[-2:] 
        max_dx = self.translate_range * img_w
        max_dy = self.translate_range * img_h
        translate = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))

        # 2. 對每個 key 應用相同的變換
        for key in self.keys:
            if key not in data: continue
            
            tensor = data[key]
            # 確保維度正確: Mask 如果是 (H, W) 要擴充成 (1, H, W) 才能轉，轉完再壓回來
            needs_squeeze = False
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
                needs_squeeze = True
            
            # 使用 Torchvision functional 進行 affine transform
            # interpolation: Image 用 Bilinear, Mask 用 Nearest (避免 mask 出現小數點)
            interp = TF.InterpolationMode.BILINEAR if key == 'image' else TF.InterpolationMode.NEAREST
            
            transformed = TF.affine(
                tensor, 
                angle=angle, 
                translate=translate, 
                scale=1.0, 
                shear=0.0,
                interpolation=interp
            )
            
            if needs_squeeze:
                transformed = transformed.squeeze(0)
                
            data[key] = transformed
            
        return data

class ImageNoiseColorAugment(BaseTransform):
    """
    只對 'image' 進行光度增強：
    1. ColorJitter (亮度、對比、飽和度、色相)
    2. Gaussian Noise (高斯雜訊)
    
    注意：預期輸入 Image 為 Tensor [0, 1] 範圍
    """
    def __init__(self, keys: List[str], sigma=0.03, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, **kwargs):
    # def __init__(self, keys: List[str], sigma=0.02, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, **kwargs):
        super().__init__(keys=keys, **kwargs)
        self.sigma = sigma
        # ColorJitter 只接受 (C, H, W) 或 (B, C, H, W)
        self.jitter = T.ColorJitter(
            brightness=brightness, 
            contrast=contrast, 
            saturation=saturation, 
            hue=hue
        )

    def _process(self, data, *args, **kwargs):
        # 這裡不實作，使用 apply_transform 統一處理
        raise NotImplementedError

    def apply_transform(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        for key in self.keys:
            # 只處理 image，跳過 mask/label
            if key == 'image' and key in data:
                img = data[key]
                
                # 1. Color Jitter
                # 需要確保 img 是 (C, H, W)
                if img.ndim == 2: img = img.unsqueeze(0)
                img = self.jitter(img)
                
                # 2. Gaussian Noise
                if self.sigma > 0:
                    noise = torch.randn_like(img) * self.sigma
                    img = img + noise
                
                # 3. Clamp 確保數值在合理範圍 [0, 1]
                img = torch.clamp(img, 0.0, 1.0)
                
                data[key] = img
                
        return data
    
class MorphologyAugment(BaseTransform):
    """
    形態學增強：使用 OpenCV 進行膨脹 (Dilation) 或腐蝕 (Erosion)。
    [Update] 改用隨機 Iterations + 固定 Kernel，以獲得更細緻的粗細控制。
    """
    def __init__(self, 
                 keys: List[str], 
                 prob=0.5, 
                 kernel_size: int = 3,          # 固定 Kernel 大小 (建議 3)
                 iteration_range: tuple = (1, 8), # 隨機 Iterations (建議 1~8)
                 **kwargs):
        super().__init__(keys=keys, **kwargs)
        self.prob = prob
        self.kernel_size = kernel_size
        self.iteration_range = iteration_range

    def _apply_cv2_morph(self, tensor, mode, iterations):
        # Tensor (C, H, W) -> Numpy (H, W, C)
        device = tensor.device
        ndim = tensor.ndim
        
        # 1. 轉 Numpy (保持 float32 0~1)
        arr = tensor.detach().cpu().numpy()
        
        if ndim == 3:
            # (C, H, W) -> (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
        
        # 2. 建立 Kernel (固定使用橢圓形，最自然)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        
        # 3. 執行 Morph (傳入 iterations)
        if mode == 'dilate':
            res = cv2.dilate(arr, kernel, iterations=iterations)
        elif mode == 'erode':
            res = cv2.erode(arr, kernel, iterations=iterations)
        else:
            res = arr
            
        # 4. 維度修正 (cv2 有時會吃掉最後一個維度如果 C=1)
        if ndim == 3 and res.ndim == 2:
            res = res[..., np.newaxis]
            
        # 5. 轉回 Tensor
        if ndim == 3:
            res = np.transpose(res, (2, 0, 1))
            
        return torch.from_numpy(res).to(device)

    def apply_transform(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        if random.random() >= self.prob:
            return data
            
        # 隨機決定模式
        mode = random.choice(['dilate', 'erode'])
        
        # 隨機決定 Iterations
        min_iter, max_iter = self.iteration_range
        iterations = random.randint(min_iter, max_iter)
        
        for key in self.keys:
            if key not in data: continue
            
            # 對 Image 和 Mask 做一樣的操作
            if key in ['image', 'mask']:
                data[key] = self._apply_cv2_morph(data[key], mode, iterations)
                
        return data


class NonLinearAugment(BaseTransform):
    """
    非線性幾何增強：透視 (Perspective) + 彈性 (Elastic)。
    [Fix] 
    1. Mask 改用 Bilinear + Threshold，解決與 Image 邊緣不一致的問題。
    2. Elastic Displacement 使用 Normalized Coordinates，解決破碎問題。
    3. 強制 4D Input，解決 grid_sample 報錯。
    """
    def __init__(self, 
                 keys: List[str], 
                 perspective_prob=0.5,
                 distortion_scale=0.5,
                 elastic_prob=0.5,
                 alpha_range=(50.0, 100.0), 
                 sigma_range=(9.0, 11.0),   
                 **kwargs):
        super().__init__(keys=keys, **kwargs)
        self.perspective_prob = perspective_prob
        self.distortion_scale = distortion_scale
        self.elastic_prob = elastic_prob
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    def _get_elastic_displacement(self, h, w, alpha, sigma):
        # 1. 生成噪聲
        displacement = torch.randn(2, h, w) * alpha
        
        # 2. 高斯模糊
        kernel_size = int(math.ceil(sigma * 3) * 2 + 1)
        displacement = TF.gaussian_blur(displacement, [kernel_size, kernel_size], [sigma, sigma])
        
        # 3. 轉為 (1, H, W, 2)
        displacement = displacement.permute(1, 2, 0).unsqueeze(0)
        
        # 4. [關鍵] Pixel -> Normalized [-1, 1]
        # 這是修復"圖片破碎"的關鍵
        displacement[..., 0] *= (2.0 / (w - 1))
        displacement[..., 1] *= (2.0 / (h - 1))
        
        return displacement

    def apply_transform(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        if self.keys[0] in data:
            shape = data[self.keys[0]].shape
            h, w = shape[-2:]
        else:
            return data

        # --- 1. 準備參數 (一次生成) ---
        
        # Perspective Params
        do_perspective = random.random() < self.perspective_prob
        perspective_params = None
        if do_perspective:
            perspective_params = T.RandomPerspective.get_params(
                width=w, height=h, distortion_scale=self.distortion_scale
            )

        # Elastic Params
        do_elastic = random.random() < self.elastic_prob
        displacement = None
        if do_elastic:
            alpha = random.uniform(*self.alpha_range)
            sigma = random.uniform(*self.sigma_range)
            displacement = self._get_elastic_displacement(h, w, alpha, sigma)

        # --- 2. 執行變換 ---
        for key in self.keys:
            if key not in data: continue
            tensor = data[key]
            
            # [關鍵修改] Mask 也用 Bilinear 插值，避免 Nearest 造成的 0.5 pixel 錯位
            interp = TF.InterpolationMode.BILINEAR
            
            # 維度處理: 升維到 4D (1, C, H, W)
            original_ndim = tensor.ndim
            if original_ndim == 2: x = tensor.unsqueeze(0).unsqueeze(0)
            elif original_ndim == 3: x = tensor.unsqueeze(0)
            else: x = tensor

            # A. Perspective
            if do_perspective and perspective_params:
                startpoints, endpoints = perspective_params
                try:
                    x = TF.perspective(x, startpoints, endpoints, interpolation=interp)
                except: pass

            # B. Elastic
            if do_elastic and displacement is not None:
                try:
                    x = TF.elastic_transform(x, displacement=displacement, interpolation=interp)
                except: pass

            # [還原維度]
            if original_ndim == 2: tensor = x.squeeze(0).squeeze(0)
            elif original_ndim == 3: tensor = x.squeeze(0)
            else: tensor = x
            
            # [關鍵修改] 如果是 Mask，用 Threshold 轉回 0/1
            if key == 'mask':
                tensor = (tensor > 0.5).float()
            
            data[key] = tensor
            
        return data

# ==========================================
# 3. Dataset Implementation
# ==========================================

class GeometricDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        template_dir: str,
        transform: Optional[Compose] = None,
        two_views: bool = True
    ):
        self.base_dir = base_dir
        self.transform = transform
        self.two_views = two_views
        self.templates = []
        self.template_names = []
        self.weights_boundaries = []
        self.num_boundary = 2048
        # self.num_boundary = 256

        rest_cage = utils.generate_circular_cage(
            # cage_num_vertices, radius=cage_radius, device='cpu'
            128, radius=1.2, device='cpu'
        )
        cage_batch = rest_cage.unsqueeze(0)

        # --- 1. Load Data List ---
        all_files = glob.glob(os.path.join(base_dir, "*.npy"))
        self.data_list = []
        for f in all_files:
            filename = os.path.basename(f)
            parts = filename.split('_')
            if len(parts) >= 2 and parts[-1] == "0.npy":
                self.data_list.append(f)
        self.data_list.sort()
        print(f"[Dataset] Found {len(self.data_list)} files. Two_views: {self.two_views}")

        # --- 2. Load Templates ---
        template_files = sorted(glob.glob(os.path.join(template_dir, "*.npz")))
        if len(template_files) == 0:
            raise RuntimeError(f"No .npz template files found in {template_dir}")
            
        print(f"[Dataset] Loading {len(template_files)} templates...")
        for t_path in template_files:
            try:
                t_name = os.path.splitext(os.path.basename(t_path))[0]
                self.template_names.append(t_name)
                with np.load(t_path) as data:
                    if 'mask' not in data:
                        raise KeyError(f"'mask' key not found in {t_path}")
                    mask = data['mask']
                
                t_tensor = torch.from_numpy(mask).float()
                
                # Ensure (1, H, W)
                if t_tensor.ndim == 2:
                    t_tensor = t_tensor.unsqueeze(0)
                elif t_tensor.ndim == 3 and t_tensor.shape[0] != 1:
                     if t_tensor.shape[-1] == 1:
                         t_tensor = t_tensor.permute(2, 0, 1)
                
                self.templates.append(t_tensor)
            except Exception as e:
                print(f"Error loading template {t_path}: {e}")
                raise e
        
        for t_path in template_files:
            try:
                t_name = os.path.splitext(os.path.basename(t_path))[0]
                with np.load(t_path) as data:
                    if 'mask' not in data: raise KeyError(f"'mask' key not found in {t_path}")
                    mask = data['mask']
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
                
                h, w = mask.shape
                
                # === A. Sample Boundary Points ===
                # CRITICAL: Use CHAIN_APPROX_NONE to get ALL boundary points!
                # CHAIN_APPROX_SIMPLE compresses contours and loses information
                # (e.g., rectangle would only have 4 corner points)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                # Get all boundary points
                b_pts_raw = cnts[0].squeeze().reshape(-1, 2).astype(np.float32)
                
                # Normalize to [-1, 1]
                b_pts_norm = (b_pts_raw / np.array([w-1, h-1])) * 2 - 1
                b_pts_tensor = torch.tensor(b_pts_norm, dtype=torch.float32)
                total_b = b_pts_tensor.shape[0]
                
                # Compute MVC weights for boundary (in chunks to save memory)
                w_b_list = []
                chunk_size = 5000
                for i in range(0, len(b_pts_tensor), chunk_size):
                    chunk = b_pts_tensor[i:i+chunk_size].unsqueeze(0)
                    w_b_list.append(utils.compute_mvc_weights(chunk, cage_batch).squeeze(0))
                weights_boundary = torch.cat(w_b_list, dim=0)
                
                start_idx = np.random.randint(0, total_b)
                
                if total_b >= self.num_boundary:
                    # Uniform stride to cover entire contour
                    stride = total_b / self.num_boundary
                    idx_b = np.array([int((start_idx + i * stride) % total_b) for i in range(self.num_boundary)])
                else:
                    # Not enough points: repeat with interpolation
                    idx_b = np.array([int((start_idx + i) % total_b) for i in range(self.num_boundary)])
                idx_b = torch.tensor(idx_b, dtype=torch.long)

                self.weights_boundaries.append(weights_boundary[idx_b])
            except Exception as e:
                print(f"Error loading template {t_path}: {e}")

        self.weights_boundaries = torch.stack(self.weights_boundaries, dim=0)

    def _process_single_view(self, data_dict: Dict) -> Dict:
        """
        處理單一視角的邏輯: Transform -> Masking -> Template Stacking
        """
        # 1. Transform (Augmentation)
        if self.transform:
            data_dict = self.transform(data_dict)
        
        # 確保 mask 是 (1, H, W) 用於廣播
        if 'mask' in data_dict and data_dict['mask'].ndim == 2:
            data_dict['mask'] = data_dict['mask'].unsqueeze(0)

        # 2. Apply Masking (Image * Mask)
        # 濾掉 Augmentation 產生的邊界雜訊，只保留物件本身
        if 'image' in data_dict and 'mask' in data_dict:
            # Image: (3, H, W), Mask: (1, H, W) -> Broadcast OK
            data_dict['image'] = data_dict['image'] * data_dict['mask']

        # 3. Stack Templates (Image only)
        # 將 5 個 Templates 接在 Image Channel 後面
        if 'image' in data_dict:
            img_tensor = data_dict['image'] # (3, H, W)
            stacked_inputs = []
            for t_tensor in self.templates:
                # combined: (4, H, W)
                combined = torch.cat([img_tensor, t_tensor], dim=0)
                stacked_inputs.append(combined)
            
            # Result: (5, 4, H, W)
            data_dict['image'] = torch.stack(stacked_inputs, dim=0)
            data_dict['weights_boundary'] = self.weights_boundaries.clone()
            data_dict['template_names'] = self.template_names[:] # list [wb] = [:]
            # data_dict['weights_boundary'] = compute(template_names) -> list [:] of wb
            
        return data_dict

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        # 初始字典
        raw_data = {
            "filename": os.path.basename(self.data_list[idx]),
            "data_path": self.data_list[idx] 
        }

        if self.two_views:
            # === Two Views Mode ===
            # 分別獨立做兩次 Transform + Processing
            view1 = self._process_single_view(raw_data.copy())
            view2 = self._process_single_view(raw_data.copy())
            
            # 將所有 Tensor 類型的 key 進行 Stack
            final_data = {}
            # 遍歷 view1 的 key，將 view2 對應的 key 接起來
            for k, v in view1.items():
                if isinstance(v, torch.Tensor):
                    v2 = view2[k]
                    # Stack dim=0: (2, ...)
                    final_data[k] = torch.stack([v, v2], dim=0)
                elif k == 'template_names':
                    # [核心修改] List 堆疊: [5] -> [2, 5] (List of Lists)
                    # view1['template_names'] 是 ['a', 'b', ...]
                    # 結果變成 [['a', 'b', ...], ['a', 'b', ...]]
                    final_data[k] = [view1[k], view2[k]]
                else:
                    # 字串檔名等 metadata，通常保留一份即可，或用 list
                    final_data[k] = v 
            
            return final_data

        else:
            # === Single View Mode ===
            return self._process_single_view(raw_data.copy())
        
class MNISTGeometricDataset(MNIST):
    def __init__(self, 
                 root, 
                 template_dir, 
                 train=False, 
                 img_size=256, 
                 download=True,
                 thicken=False,          # [新增] 是否加粗
                 thicken_kernel_size=3,   # [新增] 加粗程度 (建議 3 或 5)
                 thicken_iterations=8,
                 ):
        super().__init__(root, train=train, download=download)
        self.img_size = img_size
        self.templates = []
        self.template_names = []
        self.weights_boundaries = []
        self.num_boundary = 2048
        # self.num_boundary = 256
        rest_cage = utils.generate_circular_cage(
            # cage_num_vertices, radius=cage_radius, device='cpu'
            128, radius=1.2, device='cpu'
        )
        cage_batch = rest_cage.unsqueeze(0)
        
        # [新增] 加粗設定
        self.thicken = thicken
        self.thicken_kernel_size = thicken_kernel_size
        self.thicken_iterations = thicken_iterations
        
        # Load Templates
        template_files = sorted(glob.glob(os.path.join(template_dir, "*.npz")))
        # print(f"[MNIST] Loading {len(template_files)} templates...") # 避免洗版可以註解掉
        
        for t_path in template_files:
            try:
                t_name = os.path.splitext(os.path.basename(t_path))[0]
                self.template_names.append(t_name)
                with np.load(t_path) as data:
                    if 'mask' not in data: continue
                    mask = data['mask']
                t_tensor = torch.from_numpy(mask).float()
                if t_tensor.ndim == 2: t_tensor = t_tensor.unsqueeze(0)
                elif t_tensor.ndim == 3 and t_tensor.shape[0] != 1:
                     if t_tensor.shape[-1] == 1: t_tensor = t_tensor.permute(2, 0, 1)
                self.templates.append(t_tensor)
            except Exception as e:
                print(f"Error loading template {t_path}: {e}")

        for t_path in template_files:
            try:
                t_name = os.path.splitext(os.path.basename(t_path))[0]
                with np.load(t_path) as data:
                    if 'mask' not in data: raise KeyError(f"'mask' key not found in {t_path}")
                    mask = data['mask']
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
                
                h, w = mask.shape
                
                # === A. Sample Boundary Points ===
                # CRITICAL: Use CHAIN_APPROX_NONE to get ALL boundary points!
                # CHAIN_APPROX_SIMPLE compresses contours and loses information
                # (e.g., rectangle would only have 4 corner points)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                # Get all boundary points
                b_pts_raw = cnts[0].squeeze().reshape(-1, 2).astype(np.float32)
                
                # Normalize to [-1, 1]
                b_pts_norm = (b_pts_raw / np.array([w-1, h-1])) * 2 - 1
                b_pts_tensor = torch.tensor(b_pts_norm, dtype=torch.float32)
                total_b = b_pts_tensor.shape[0]
                
                # Compute MVC weights for boundary (in chunks to save memory)
                w_b_list = []
                chunk_size = 5000
                for i in range(0, len(b_pts_tensor), chunk_size):
                    chunk = b_pts_tensor[i:i+chunk_size].unsqueeze(0)
                    w_b_list.append(utils.compute_mvc_weights(chunk, cage_batch).squeeze(0))
                weights_boundary = torch.cat(w_b_list, dim=0)

                start_idx = np.random.randint(0, total_b)
                
                if total_b >= self.num_boundary:
                    # Uniform stride to cover entire contour
                    stride = total_b / self.num_boundary
                    idx_b = np.array([int((start_idx + i * stride) % total_b) for i in range(self.num_boundary)])
                else:
                    # Not enough points: repeat with interpolation
                    idx_b = np.array([int((start_idx + i) % total_b) for i in range(self.num_boundary)])
                idx_b = torch.tensor(idx_b, dtype=torch.long)

                self.weights_boundaries.append(weights_boundary[idx_b])
            except Exception as e:
                print(f"Error loading template {t_path}: {e}")

        self.weights_boundaries = torch.stack(self.weights_boundaries, dim=0)

    def _stack_templates(self, img_tensor):
        stacked_inputs = []
        for t_tensor in self.templates:
            combined = torch.cat([img_tensor, t_tensor], dim=0)
            stacked_inputs.append(combined)
        return torch.stack(stacked_inputs, dim=0)

    def _apply_thickening(self, img_tensor):
        """
        對 Tensor 圖像進行膨脹操作
        img_tensor: (1, H, W) range [0, 1]
        """
        # 1. Tensor -> Numpy (H, W)
        # 注意：cv2 需要 uint8 或 float32，這裡保持 float
        arr = img_tensor.squeeze(0).cpu().numpy()
        
        # 2. 建立橢圓 Kernel (比較滑順)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.thicken_kernel_size, self.thicken_kernel_size)
        )
        
        # 3. Dilate
        # iterations=1 通常就夠了，靠 kernel size 控制粗細
        arr_dilated = cv2.dilate(arr, kernel, iterations=self.thicken_iterations)
        
        # 4. Numpy -> Tensor (1, H, W)
        return torch.from_numpy(arr_dilated).unsqueeze(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        
        # 1. PIL/Tensor -> Resize
        img = TF.to_pil_image(img)
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.to_tensor(img) # (1, 256, 256)
        
        # 2. [新增] 加粗處理 (Thickening)
        # 在產生 Mask 之前做，這樣 Mask 也會跟著變大
        if self.thicken:
            img = self._apply_thickening(img)

        # 3. 產生 Mask
        mask = (img > 0.1).float()
        
        # 4. 擴充 Channel (1 -> 3)
        img = img.repeat(3, 1, 1) 
        
        # 5. Masking
        img = img * mask
        
        # 6. Stack Templates
        final_img = self._stack_templates(img)
        
        return {
            "image": final_img,
            "label": target,
            "mask": mask,
            "template_names": self.template_names[:],
            "weights_boundary": self.weights_boundaries.clone(),
            # "weights_boundary" = compute(template_names) -> list [:] of wb
        }
    
class OmniglotGeometricDataset(Omniglot):
    def __init__(self, 
                 root, 
                 template_dir, 
                 partition='support',    # [控制] 'support' (前10張), 'query' (後10張), 'all'
                 img_size=256, 
                 download=True,
                 thicken=True,           # Omniglot 線條很細，預設開啟加粗
                 thicken_kernel_size=3,
                 thicken_iterations=8,   # Omniglot 通常需要較多次迭代來讓線條明顯
                 ):
        # Omniglot 的 background=True 代表 Training set (964 classes)
        # 我們用這 964 類來做 split
        super().__init__(root, background=True, download=download)
        
        self.img_size = img_size
        self.partition = partition
        
        # --- 1. Data Splitting (Partitioning) ---
        # Omniglot 每個 Character 有 20 張圖 (rotations=0 情況下)
        # self._flat_character_images 是一個 list of (image_path, character_class_index)
        # Torchvision 載入時已經按順序排好了，所以每 20 個就是一組
        
        full_data = self._flat_character_images
        filtered_data = []
        
        for idx, item in enumerate(full_data):
            # 取餘數決定是該類別的第幾張 (0~19)
            sample_idx = idx % 20 
            
            if partition == 'support':
                # 取前 10 張 (0-9)
                if sample_idx < 10:
                    filtered_data.append(item)
            elif partition == 'query':
                # 取後 10 張 (10-19)
                if sample_idx >= 10:
                    filtered_data.append(item)
            else:
                # 'all'
                filtered_data.append(item)
        
        # 覆蓋原本的資料列表
        self._flat_character_images = filtered_data
        print(f"[Omniglot] Partition: {partition} | Samples: {len(self._flat_character_images)}")

        # --- 2. Template & Boundary Weights Logic (複製自你的 MNIST) ---
        self.templates = []
        self.template_names = []
        self.weights_boundaries = []
        # self.num_boundary = 256 # 與 MNIST 保持一致
        self.num_boundary = 2048
        
        # 生成 Cage
        rest_cage = utils.generate_circular_cage(
            128, radius=1.2, device='cpu'
        )
        cage_batch = rest_cage.unsqueeze(0)
        
        # 加粗設定
        self.thicken = thicken
        self.thicken_kernel_size = thicken_kernel_size
        self.thicken_iterations = thicken_iterations
        
        # Load Templates
        template_files = sorted(glob.glob(os.path.join(template_dir, "*.npz")))
        
        # First Pass: Load Masks for Tensor Stacking
        for t_path in template_files:
            try:
                t_name = os.path.splitext(os.path.basename(t_path))[0]
                self.template_names.append(t_name)
                with np.load(t_path) as data:
                    if 'mask' not in data: continue
                    mask = data['mask']
                t_tensor = torch.from_numpy(mask).float()
                if t_tensor.ndim == 2: t_tensor = t_tensor.unsqueeze(0)
                elif t_tensor.ndim == 3 and t_tensor.shape[0] != 1:
                     if t_tensor.shape[-1] == 1: t_tensor = t_tensor.permute(2, 0, 1)
                self.templates.append(t_tensor)
            except Exception as e:
                print(f"Error loading template {t_path}: {e}")

        # Second Pass: Compute MVC Weights
        print("[Omniglot] Computing template boundary weights...")
        for t_path in template_files:
            try:
                with np.load(t_path) as data:
                    if 'mask' not in data: raise KeyError(f"'mask' key not found in {t_path}")
                    mask = data['mask']
                
                # 確保 mask 是 uint8 0-255
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
                
                h, w = mask.shape
                
                # Contour extraction
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                b_pts_raw = cnts[0].squeeze().reshape(-1, 2).astype(np.float32)
                
                # Normalize [-1, 1]
                b_pts_norm = (b_pts_raw / np.array([w-1, h-1])) * 2 - 1
                b_pts_tensor = torch.tensor(b_pts_norm, dtype=torch.float32)
                total_b = b_pts_tensor.shape[0]
                
                # Compute MVC
                w_b_list = []
                chunk_size = 5000
                for i in range(0, len(b_pts_tensor), chunk_size):
                    chunk = b_pts_tensor[i:i+chunk_size].unsqueeze(0)
                    w_b_list.append(utils.compute_mvc_weights(chunk, cage_batch).squeeze(0))
                weights_boundary = torch.cat(w_b_list, dim=0)

                # Sampling
                start_idx = np.random.randint(0, total_b)
                if total_b >= self.num_boundary:
                    stride = total_b / self.num_boundary
                    idx_b = np.array([int((start_idx + i * stride) % total_b) for i in range(self.num_boundary)])
                else:
                    idx_b = np.array([int((start_idx + i) % total_b) for i in range(self.num_boundary)])
                idx_b = torch.tensor(idx_b, dtype=torch.long)

                self.weights_boundaries.append(weights_boundary[idx_b])
            except Exception as e:
                print(f"Error computing weights for {t_path}: {e}")

        self.weights_boundaries = torch.stack(self.weights_boundaries, dim=0)

    def _stack_templates(self, img_tensor):
        stacked_inputs = []
        for t_tensor in self.templates:
            combined = torch.cat([img_tensor, t_tensor], dim=0)
            stacked_inputs.append(combined)
        return torch.stack(stacked_inputs, dim=0)

    def _apply_thickening(self, img_tensor):
        # Tensor (1, H, W) -> Numpy
        arr = img_tensor.squeeze(0).cpu().numpy()
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.thicken_kernel_size, self.thicken_kernel_size)
        )
        arr_dilated = cv2.dilate(arr, kernel, iterations=self.thicken_iterations)
        return torch.from_numpy(arr_dilated).unsqueeze(0)

    def __getitem__(self, index):
        # Omniglot 回傳 (img, target)，img 是 PIL Image
        img, target = super().__getitem__(index)
        
        # 1. To Tensor & Invert Color (關鍵!)
        # Omniglot 是白底(1)黑字(0)，我們要轉成黑底(0)白字(1)
        img = TF.to_tensor(img) # (1, 105, 105)
        img = 1.0 - img 
        
        # 2. Resize
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        
        # 3. Thicken
        if self.thicken:
            img = self._apply_thickening(img)

        # 4. Mask
        mask = (img > 0.1).float()
        
        # 5. Expand Channel
        img = img.repeat(3, 1, 1)
        
        # 6. Masking
        img = img * mask
        
        # 7. Stack Templates
        final_img = self._stack_templates(img)
        
        return {
            "image": final_img,
            "label": target,
            "mask": mask,
            "template_names": self.template_names[:],
            "weights_boundary": self.weights_boundaries.clone()
        }
    
def get_omniglot_loaders(cfg):
    # 1. Support Set (前 10 張)
    omni_support_ds = OmniglotGeometricDataset(
        root=cfg.data.omniglot_root, 
        template_dir=cfg.data.template_dir,
        partition='support',  # <--- 關鍵
        img_size=cfg.data.img_size,
        download=True,
        thicken=cfg.train.omniglot_thicken,        # e.g. True
        thicken_kernel_size=cfg.train.omniglot_thicken_kernel_size, # e.g. 3
        thicken_iterations=cfg.train.omniglot_thicken_iterations    # e.g. 8
    )
    
    omni_support_loader = DataLoader(
        omni_support_ds,
        batch_size=cfg.data.val.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 2. Query Set (後 10 張)
    omni_query_ds = OmniglotGeometricDataset(
        root=cfg.data.omniglot_root, 
        template_dir=cfg.data.template_dir,
        partition='query',    # <--- 關鍵
        img_size=cfg.data.img_size,
        download=True,
        thicken=cfg.train.omniglot_thicken,  # e.g. True
        thicken_kernel_size=cfg.train.omniglot_thicken_kernel_size,
        thicken_iterations=cfg.train.omniglot_thicken_iterations
    )
    
    omni_query_loader = DataLoader(
        omni_query_ds,
        batch_size=cfg.data.val.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return omni_support_loader, omni_query_loader


class QuickDrawGeometricDataset(Dataset):
    """
    QuickDraw 幾何資料集
    
    資料結構:
        base_dir/
            category_1/
                category_1_0.npy
                category_1_1.npy
                ...
            category_2/
                ...
            label_mapping.json
    
    每個 .npy 檔案包含: [image (3, 256, 256), mask (256, 256), label (int)]
    
    支援 partition='support' (前 N 張) 或 'query' (後 M 張) 來做 few-shot evaluation
    """
    def __init__(self, 
                 base_dir: str, 
                 template_dir: str, 
                 partition: str = 'all',  # 'support', 'query', 'all'
                 support_samples: int = 10,  # 每類 support set 的樣本數
                 img_size: int = 256, 
                 ):
        super().__init__()
        self.base_dir = base_dir
        self.img_size = img_size
        self.partition = partition
        self.support_samples = support_samples
        
        self.templates = []
        self.template_names = []
        self.weights_boundaries = []
        self.num_boundary = 2048
        
        # 生成 Cage (與 MNIST/Omniglot 一致)
        rest_cage = utils.generate_circular_cage(128, radius=1.2, device='cpu')
        cage_batch = rest_cage.unsqueeze(0)
        
        # --- 1. 載入資料列表 ---
        self.data_list = []
        self.labels = []
        
        # 讀取 label_mapping.json
        import json
        label_mapping_path = os.path.join(base_dir, "label_mapping.json")
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r') as f:
                self.label_mapping = json.load(f)
        else:
            self.label_mapping = {}
        
        # 遍歷所有類別資料夾
        category_dirs = sorted([d for d in os.listdir(base_dir) 
                               if os.path.isdir(os.path.join(base_dir, d))])
        
        for cat_dir in category_dirs:
            cat_path = os.path.join(base_dir, cat_dir)
            npy_files = sorted(glob.glob(os.path.join(cat_path, "*.npy")))
            
            # 根據 partition 決定要用哪些樣本
            if partition == 'support':
                selected_files = npy_files[:support_samples]
            elif partition == 'query':
                selected_files = npy_files[support_samples:]
            else:  # 'all'
                selected_files = npy_files
            
            for f in selected_files:
                self.data_list.append(f)
        
        print(f"[QuickDraw] Partition: {partition} | Samples: {len(self.data_list)}")
        
        # --- 2. 載入 Templates (與 MNIST/Omniglot 一致) ---
        template_files = sorted(glob.glob(os.path.join(template_dir, "*.npz")))
        
        # First Pass: Load Masks
        for t_path in template_files:
            try:
                t_name = os.path.splitext(os.path.basename(t_path))[0]
                self.template_names.append(t_name)
                with np.load(t_path) as data:
                    if 'mask' not in data: 
                        continue
                    mask = data['mask']
                t_tensor = torch.from_numpy(mask).float()
                if t_tensor.ndim == 2: 
                    t_tensor = t_tensor.unsqueeze(0)
                elif t_tensor.ndim == 3 and t_tensor.shape[0] != 1:
                    if t_tensor.shape[-1] == 1: 
                        t_tensor = t_tensor.permute(2, 0, 1)
                self.templates.append(t_tensor)
            except Exception as e:
                print(f"Error loading template {t_path}: {e}")
        
        # Second Pass: Compute MVC Weights
        for t_path in template_files:
            try:
                with np.load(t_path) as data:
                    if 'mask' not in data:
                        raise KeyError(f"'mask' key not found in {t_path}")
                    mask = data['mask']
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
                
                h, w = mask.shape
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                b_pts_raw = cnts[0].squeeze().reshape(-1, 2).astype(np.float32)
                b_pts_norm = (b_pts_raw / np.array([w-1, h-1])) * 2 - 1
                b_pts_tensor = torch.tensor(b_pts_norm, dtype=torch.float32)
                total_b = b_pts_tensor.shape[0]
                
                w_b_list = []
                chunk_size = 5000
                for i in range(0, len(b_pts_tensor), chunk_size):
                    chunk = b_pts_tensor[i:i+chunk_size].unsqueeze(0)
                    w_b_list.append(utils.compute_mvc_weights(chunk, cage_batch).squeeze(0))
                weights_boundary = torch.cat(w_b_list, dim=0)
                
                start_idx = np.random.randint(0, total_b)
                if total_b >= self.num_boundary:
                    stride = total_b / self.num_boundary
                    idx_b = np.array([int((start_idx + i * stride) % total_b) for i in range(self.num_boundary)])
                else:
                    idx_b = np.array([int((start_idx + i) % total_b) for i in range(self.num_boundary)])
                idx_b = torch.tensor(idx_b, dtype=torch.long)
                
                self.weights_boundaries.append(weights_boundary[idx_b])
            except Exception as e:
                print(f"Error computing weights for {t_path}: {e}")
        
        self.weights_boundaries = torch.stack(self.weights_boundaries, dim=0)

    def _stack_templates(self, img_tensor):
        stacked_inputs = []
        for t_tensor in self.templates:
            combined = torch.cat([img_tensor, t_tensor], dim=0)
            stacked_inputs.append(combined)
        return torch.stack(stacked_inputs, dim=0)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 1. 載入 .npy 檔案
        data = np.load(self.data_list[index], allow_pickle=True)
        image = data[0]  # (3, 256, 256) float32
        mask = data[1]   # (256, 256) uint8 {0, 1}
        label = int(data[2])
        
        # 2. 轉為 Tensor
        img = torch.from_numpy(image).float()  # (3, 256, 256)
        mask = torch.from_numpy(mask).float()  # (256, 256)
        
        # 3. Resize (如果需要)
        if img.shape[-1] != self.img_size:
            img = TF.resize(img, [self.img_size, self.img_size], 
                           interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask.unsqueeze(0), [self.img_size, self.img_size], 
                            interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        
        # 4. 確保 mask 維度正確 (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        # 5. Masking (Image * Mask)
        img = img * mask
        
        # 6. Stack Templates
        final_img = self._stack_templates(img)
        
        return {
            "image": final_img,
            "label": label,
            "mask": mask,
            "template_names": self.template_names[:],
            "weights_boundary": self.weights_boundaries.clone()
        }


def get_quickdraw_loaders(cfg):
    """
    建立 QuickDraw 的 Support 和 Query DataLoader
    """
    base_dir = cfg.data.quickdraw_root
    
    # 1. Support Set (每類前 N 張)
    qd_support_ds = QuickDrawGeometricDataset(
        base_dir=base_dir,
        template_dir=cfg.data.template_dir,
        partition='support',
        support_samples=cfg.train.get('quickdraw_support_samples', 10),
        img_size=cfg.data.img_size,
    )
    
    qd_support_loader = DataLoader(
        qd_support_ds,
        batch_size=cfg.data.val.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 2. Query Set (每類剩餘的樣本)
    qd_query_ds = QuickDrawGeometricDataset(
        base_dir=base_dir,
        template_dir=cfg.data.template_dir,
        partition='query',
        support_samples=cfg.train.get('quickdraw_support_samples', 10),
        img_size=cfg.data.img_size,
    )
    
    qd_query_loader = DataLoader(
        qd_query_ds,
        batch_size=cfg.data.val.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return qd_support_loader, qd_query_loader

    
def get_subset_dataloader(dataset, ratio, batch_size, num_workers, seed=None):
    """
    根據比例隨機從 Dataset 中採樣，建立一個新的 DataLoader。
    
    Args:
        dataset: 原始 Dataset
        ratio: 採樣比例 (0.0 ~ 1.0)
        batch_size, num_workers: DataLoader 參數
    """
    total_len = len(dataset)
    subset_len = int(total_len * ratio)
    
    # 如果比例是 1.0 或大於總數，直接回傳完整的 loader
    if ratio >= 1.0 or subset_len >= total_len:
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    
    # 生成隨機索引
    # 注意：如果不指定 generator，預設使用 torch 的 global RNG，
    # 這意味著每次呼叫這個函數，只要沒有固定 seed，都會拿到不同的子集。
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(total_len, generator=g)[:subset_len]
    else:
        indices = torch.randperm(total_len)[:subset_len]
        
    # 建立 Subset
    subset = Subset(dataset, indices)
    
    # 建立並回傳 DataLoader
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False, # 驗證集通常不用 shuffle，因為 indices 已經是隨機的了
        num_workers=num_workers,
        prefetch_factor=3,
    )

def show_img(img, name="Image.png", dir="./test_img_dir"):
    """
    Args:
        img: torch.Tensor or np.ndarray. Expected range [0, 1].
             Shapes accepted: (H, W), (C, H, W), (H, W, C), (B, C, H, W)
        name: Filename to save.
        dir: Directory to save.
    """
    # 1. 確保目錄存在
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    # 2. 若是 Tensor，轉為 Numpy (detach gradients, move to cpu)
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # 3. 處理 Batch 維度: 如果是 4D (B, C, H, W)，取第一張
    if img.ndim == 4:
        img = img[0]

    # 4. 處理 Channel 維度 (C, H, W) -> (H, W, C)
    # 判斷邏輯：如果維度是 3，且第 0 維是 1 或 3 (且小於後面的維度)，判定為 Channel-first
    if img.ndim == 3:
        if img.shape[0] in [1, 3] and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0)) # (C, H, W) -> (H, W, C)
    
    # 5. 處理單 Channel (H, W, 1) -> (H, W) 變成純灰階
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    # 6. 數值轉換 [0, 1] -> [0, 255] uint8
    # 使用 clip 確保數值不會因為浮點數誤差溢出
    img = (img * 255).clip(0, 255).astype(np.uint8)

    # 7. 存檔
    try:
        pil_img = Image.fromarray(img)
        save_path = os.path.join(dir, name)
        pil_img.save(save_path)
        print(f"[Debug] Saved {save_path} (Shape: {img.shape})")
    except Exception as e:
        print(f"[Error] Failed to save image: {e}, shape: {img.shape}, dtype: {img.dtype}")
# ==========================================
# 4. Usage Example (測試與 Config 對應範例)
# ==========================================
if __name__ == "__main__":
    set_random_seed(42, deterministic=True)
    # 假設這是在 main.py 或 train.py 中定義 transform 的方式
    
    # 訓練用 Transform
    train_transform = Compose([
        # 1. 讀取與拆分: data_path -> image, mask, label
        LoadGeometricData(load_key="data_path", output_keys=["image", "mask", "label"]),
        
        # 2. 轉 Tensor
        ToTensord(keys=["image", "mask", "label"]),
        
        # 3. Augmentation (只對 image 和 mask 做，label 不動)
        RandomGeoAugment(keys=["image", "mask"], translate_range=0.05, degrees=15)
    ])

    # 驗證用 Transform (不做 Augmentation)
    val_transform = Compose([
        LoadGeometricData(load_key="data_path", output_keys=["image", "mask", "label"]),
        ToTensord(keys=["image", "mask", "label"]),
    ])

    # 模擬測試
    dataset = GeometricDataset(base_dir="./data/train", template_dir="./template", transform=train_transform, two_views=True)
    for i in range(10):
        sample = dataset[i]
        print(sample['image'].shape)
        print(sample['mask'].shape)
        for j in range(2):
            show_img(sample['image'][j, 0, 0:3, :, :], name=f"train_image_{i}_{j}.png")
            show_img(sample['mask'][j], name=f"train_mask_{i}_{j}.png")
            print(f"Sample {i}_{j} - Label: {sample['label'][j].item()}")
    sample = dataset[0]["image"]
    for t_idx in range(sample.shape[1]):
        show_img(sample[0, t_idx, 3, :, :], name=f"template_{t_idx}_image.png")