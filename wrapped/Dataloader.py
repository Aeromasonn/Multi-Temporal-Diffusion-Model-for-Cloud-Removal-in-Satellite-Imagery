import os
import re
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Local Dataloader & Data Preparation Pipeline
# ----------------------------

class SatImage_Dataloader(Dataset):
    """
    Sen2-MTC dataset with patch extraction: 3 cloudy v.s. 1 clean.
    Sen2-MTC:
        Per-image: Resolution: 256*256; 4 channel: RGB + 1 grayscale
        [128, 128, 4]

    For each time index n:
        cloudy[n]    -> total 3 *.tif files
        cloudless[n] -> one *.tif file (clean)

    For each logical sample i:

        self.cloudy[i] = [cloudy_t1, cloudy_t2, cloudy_t3]
        self.clean[i]  = clean_image

    If patch_size is provided, stride not:
        If center_crop True : return crop from the center of image.
        If center_crop False: return a random crop on the image.

    If patch_size is provided, also stride:
        Enumerate all patches of size patch_size with given stride.
        Default patch_size = stride = 128 -> 4*4 tiles of original image
        [Recommended to provide both patch_size and stride]

    Output size: [3, 4, patch_size, patch_size]
    """

    def __init__(self, route,
                 patch_size=128, stride=128,
                 center_crop=False, transform=None):

        super().__init__()
        self.root_dir = route
        self.patch_size = patch_size
        self.center_crop = center_crop
        self.transform = transform
        self.stride = stride
        self.samples = []

        cloudy_pattern    = r"(.+?)_(\d+)_(\d+)\.tif"  # match: <tile>_<n>_<k>.tif
        cloudless_pattern = r"(.+?)_(\d+)\.tif"        # match: <tile>_<n>.tif

        for tile_name in sorted(os.listdir(route)):
            tile_path = os.path.join(route, tile_name)
            cloud_dir = os.path.join(tile_path, "cloud")
            clean_dir = os.path.join(tile_path, "cloudless")

            if not (os.path.isdir(cloud_dir) and os.path.isdir(clean_dir)):
                continue

            # 1: Parse cloudy files grouped by time index n
            cloudy_by_n = {}
            for fname in sorted(os.listdir(cloud_dir)):
                if not fname.endswith(".tif"):
                    continue
                m = re.match(cloudy_pattern, fname)
                if not m:
                    continue

                n = int(m.group(2))
                path = os.path.join(cloud_dir, fname)
                cloudy_by_n.setdefault(n, []).append(path)

            # 2: Parse cloudless (clean) files by time index n
            clean_by_n = {}
            for fname in sorted(os.listdir(clean_dir)):
                if not fname.endswith(".tif"):
                    continue
                m = re.match(cloudless_pattern, fname)
                if not m:
                    continue

                n = int(m.group(2))
                clean_by_n[n] = os.path.join(clean_dir, fname)

            # 3: For each n, use ALL cloudy paths for that n + matching clean[n]
            for n in sorted(cloudy_by_n.keys()):
                if n not in clean_by_n:
                    print(f"[WARNING] Tile {tile_name}: time index {n} has cloudy but no clean.")
                    continue

                cloudy_paths = sorted(cloudy_by_n[n])   # list of paths for this n

                # Enforce exactly 3 cloudy views:
                if len(cloudy_paths) < 3:
                    print(f"[WARNING] Tile {tile_name} time {n}: "
                          f"only {len(cloudy_paths)} cloudy files, skipping.")
                    continue
                elif len(cloudy_paths) > 3:
                    cloudy_paths = cloudy_paths[:3]

                clean_path = clean_by_n[n]

                # If stride is not provided -> 1 sample per image set
                if self.stride is None:
                    self.samples.append((cloudy_paths, clean_path, None, None))
                    continue

                # Otherwise, enumerate patches using stride.
                tmp = tiff.imread(cloudy_paths[0])  # (H, W, C)
                H, W, _ = tmp.shape

                ps = self.patch_size
                st = self.stride

                for top in range(0, H - ps + 1, st):
                    for left in range(0, W - ps + 1, st):
                        self.samples.append((cloudy_paths, clean_path, top, left))

        print(f"[Sen2-MTC Data loaded] Total samples: {len(self.samples)}")

    def load_tif(self, path):
        arr = tiff.imread(path)  # (H, W, C)
        arr = np.array(arr, dtype=np.float32)
        return arr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cloudy_paths, clean_path, top, left = self.samples[idx]

        # 1: Load clean image
        clean_np = self.load_tif(clean_path)  # (H, W, C)
        clean = torch.from_numpy(clean_np.transpose(2, 0, 1))  # (C, H, W)

        # 2: Load all cloudy images for this sample
        cloudy_list = []
        for p in cloudy_paths:
            c_np = self.load_tif(p)  # (H, W, C)
            c_t  = torch.from_numpy(c_np.transpose(2, 0, 1))  # (C, H, W)
            cloudy_list.append(c_t)

        # stack into (T, C, H, W)
        cloudy_seq = torch.stack(cloudy_list, dim=0)
        # first view as "cloudy"
        cloudy = cloudy_seq[0]

        # 3: Patch extraction
        if self.patch_size is not None:
            ps = self.patch_size

            if self.stride is not None:
                #Crop via patch_size & stride
                cloudy      = cloudy[:, top:top+ps, left:left+ps]
                clean       = clean[:, top:top+ps, left:left+ps]
                cloudy_seq  = cloudy_seq[:, :, top:top+ps, left:left+ps]
            else:
                # center or random crop
                _, H, W = cloudy.shape
                if ps > H or ps > W:
                    raise ValueError(f"Patch size {ps} > image size {(H, W)}")

                if self.center_crop:
                    top = (H - ps) // 2
                    left = (W - ps) // 2
                else:
                    top = np.random.randint(0, H - ps + 1)
                    left = np.random.randint(0, W - ps + 1)

                clean      = clean[:, top:top+ps, left:left+ps]
                cloudy_seq = cloudy_seq[:, :, top:top+ps, left:left+ps]

        sample = {
            "cloudy_seq": cloudy_seq,  # (T, 4, H, W), T = 3
            "clean": clean
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalization:
    """
    Normalize data patches
    """
    def __init__(self, mean, std):
        self.mean = mean.reshape(-1, 1, 1)
        self.std = std.reshape(-1, 1, 1)

    def __call__(self, sample):
        clean = sample["clean"]  # (C,H,W)
        seq = sample["cloudy_seq"]  # (T,C,H,W)

        clean_n = (clean - self.mean) / (self.std + 1e-6)
        seq_n = (seq - self.mean) / (self.std + 1e-6)

        return {
            "clean": clean_n,
            "cloudy_seq": seq_n
        }

def compute_global_stats(dataset, batch_size=32):
    """
    Compute mean and std on the first cloudy view of every sequence
    Passed to Normalization.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    channel_sum = None
    channel_sq_sum = None
    total_pixels = 0

    for batch in loader:
        # use first cloudy view from the sequence: (B,T,C,H,W) -> (B,C,H,W)
        x = batch["cloudy_seq"][0].double()
        B, C, H, W = x.shape

        if channel_sum is None:
            channel_sum = torch.zeros(C, dtype=torch.float64)
            channel_sq_sum = torch.zeros(C, dtype=torch.float64)

        channel_sum    += x.sum(dim=[0, 2, 3])
        channel_sq_sum += (x ** 2).sum(dim=[0, 2, 3])
        total_pixels   += B * H * W

    mean = channel_sum / total_pixels
    std  = torch.sqrt(channel_sq_sum / total_pixels - mean ** 2)

    print("Global mean:", mean)
    print("Global std:", std)
    return mean.float(), std.float()

def prep_data_local(path,
                    patch_size=128, stride=128, batch_size=16,
                    train_ratio=.7, val_ratio=.15,
                    return_loaders = True):
    """
    The full wrapper for local data preparation.
    """
    dataset_raw = SatImage_Dataloader(
        route=path,
        patch_size=patch_size,
        stride=stride,
        transform=None
    )

    mean, std = compute_global_stats(dataset_raw, batch_size=batch_size)

    dataset_norm = SatImage_Dataloader(
        route=path,
        patch_size=patch_size,
        stride=stride,
        transform=Normalization(mean, std)
    )

    total = len(dataset_norm)
    train_len = int(total * train_ratio)
    val_len = int(total * val_ratio)
    test_len = total - train_len - val_len

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset_norm,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(2025)
    )

    if return_loaders:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, test_loader
    else:
        return train_set, val_set, test_set

# ----------------------------
# Locally Pre-compute Dataloader
# &
# On Cloud Load Pre-computed Dataloader & Data Preparation Pipeline
# ----------------------------

def pack_and_save(path,
                  file_name,
                  patch_size=128, stride=128, batch_size=16,
                  train_ratio=.7, val_ratio=.15):
    train_set, val_set, test_set = prep_data_local(path, patch_size,
                                                            stride, batch_size,
                                                            train_ratio, val_ratio,
                                                            return_loaders=False)
    def pack_split(split):
        """Convert a Subset into a list of {cloudy_seq, clean} dicts."""
        out = []
        for i in range(len(split)):
            item = split[i]
            out.append({
                "cloudy_seq": item["cloudy_seq"],
                "clean": item["clean"]
            })
        return out

    data = {
        "train": pack_split(train_set),
        "val": pack_split(val_set),
        "test": pack_split(test_set)
    }

    torch.save(data, f"./Ckpts/{file_name}.pt")
    print(f"Saved as {file_name}.pt")

class PrecomputedSen2MTC(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "cloudy_seq": item["cloudy_seq"],
            "clean": item["clean"]
        }

def collate_precomputed(batch):

    cloudy = [b["cloudy_seq"] for b in batch]
    clean  = [b["clean"]      for b in batch]

    cloudy = torch.stack(cloudy, dim=0)    # (B,T,C,H,W)
    clean  = torch.stack(clean,  dim=0)    # (B,C,H,W)

    return {
        "cloudy_seq": cloudy,
        "clean": clean
    }

def prep_data_precomputed(file_name, batch_size=16):
    """
    Pre-computed Dataloader already includes all necessary config info,
    no need to specify once more.
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')  # Mount Google Drive
        path = f"/content/drive/MyDrive/Colab Notebooks/{file_name}"
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print('Not running in Colab, please switch to local data prep instead!')
        return

    train_set = PrecomputedSen2MTC(data["train"])
    val_set = PrecomputedSen2MTC(data["val"])
    test_set = PrecomputedSen2MTC(data["test"])

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_precomputed)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_precomputed)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_precomputed)

    return train_loader, val_loader, test_loader