import os
import sys
import json
import random
import argparse
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2, RandomHorizontalFlip, RandomVerticalFlip, InterpolationMode
from tqdm.auto import tqdm

# Required for the real Mask2Former backbone
from transformers import Mask2FormerForUniversalSegmentation

# --- REPRODUCIBILITY ---

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# --- LOGGER UTILITY ---

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    # Add these two methods to fix compatibility error
    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


def get_args():
    parser = argparse.ArgumentParser(description="Mask2Former Fine-tuning")

    # Paths
    parser.add_argument("--data_path", type=str, default="/home/satoshi.tsutsui/projects/wbcas/dataset_txt/pbc_attr_v1_ccrop_all.csv")
    parser.add_argument("--data_root", type=str, default="/home/satoshi.tsutsui/satoshissd/PBC/pbcseg_final_v1/")

    # Model & Resolution
    parser.add_argument("--model_name", type=str, default="facebook/mask2former-swin-tiny-ade-semantic",
                        help="Hugging Face model checkpoint")
    parser.add_argument("--resolution", type=int, default=1024, help="Input image resolution")
    parser.add_argument("--out_resolution", type=int, default=360, help="Output/Dataset resolution")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of target classes")
    parser.add_argument("--ignore_index", type=int, default=0, help="Class index to ignore")

    # Augmentation options (new)
    parser.add_argument("--no_flip", action="store_true", help="Disable flips")
    parser.add_argument("--use_crop", action="store_true", help="Enable random crop + resize")
    parser.add_argument("--use_color", action="store_true", help="Enable color jitter")

    # Freezing Options
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    parser.add_argument("--freeze_decoder", action="store_true", default=False)

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pflip", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Paths & Device
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default="m2f_finetune")

    return parser.parse_args()

# --- MASK2FORMER WRAPPER ---

class Mask2FormerWrapper(nn.Module):
    def __init__(self, model_name, num_classes, out_resolution):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.out_resolution = out_resolution
        self.num_classes = num_classes

    def forward(self, images):
        outputs = self.model(pixel_values=images)
        cls_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits

        cls_probs = F.softmax(cls_logits, dim=-1)
        mask_probs = torch.sigmoid(mask_logits)

        b, q, h_small, w_small = mask_probs.shape
        mask_probs_flat = mask_probs.view(b, q, h_small * w_small)

        # Reconstruct semantic map from queries
        semantic_map = torch.bmm(cls_probs[:, :, :self.num_classes].transpose(1, 2), mask_probs_flat)
        semantic_map = semantic_map.view(b, self.num_classes, h_small, w_small)

        # Resize to 360x360 for loss and metrics
        return F.interpolate(semantic_map, size=(self.out_resolution, self.out_resolution),
                             mode="bilinear", align_corners=False)

# --- DATASET ---

class SegDataset(Dataset):
    def __init__(self, df, img_col="img_path", mask_col="mask_path",
                 backbone_res=512,  # ← NEW: input resolution
                 transform=None, pflip=0.0, flip=True, crop=False, color=False):
        self.df = df
        self.img_col = img_col
        self.mask_col = mask_col

        # Augmentation flags
        self.flip = flip and pflip > 0
        self.crop = crop
        self.color = color
        self.backbone_res = backbone_res  # store for resizing

        # Flip transforms (synced)
        if flip and pflip > 0:
            self.flip_transforms = v2.Compose([
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5)
            ])
        else:
            self.flip_transforms = lambda x: x

    def __len__(self):
        return len(self.df)

    # 🔥 Custom random resized crop → always outputs to out_resolution (360)
    def random_resized_crop(self, img, mask,
                            scale=(0.4, 1.0), ratio=(0.75, 1.33),
                            out_size=360):
        _, h, w = img.shape
        area = h * w

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            new_w = int(round((target_area * aspect_ratio) ** 0.5))
            new_h = int(round((target_area / aspect_ratio) ** 0.5))

            if new_w <= w and new_h <= h:
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)

                img_crop = v2.functional.crop(img, top, left, new_h, new_w)
                mask_crop = v2.functional.crop(mask, top, left, new_h, new_w)

                img_resized = v2.functional.resize(
                    img_crop,
                    (out_size, out_size),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True
                )
                mask_resized = v2.functional.resize(
                    mask_crop,
                    (out_size, out_size),
                    interpolation=InterpolationMode.NEAREST_EXACT
                )
                return img_resized, mask_resized

        # fallback center crop
        min_side = min(h, w)
        top = (h - min_side) // 2
        left = (w - min_side) // 2

        img_crop = v2.functional.crop(img, top, left, min_side, min_side)
        mask_crop = v2.functional.crop(mask, top, left, min_side, min_side)

        img_resized = v2.functional.resize(
            img_crop,
            (out_size, out_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        mask_resized = v2.functional.resize(
            mask_crop,
            (out_size, out_size),
            interpolation=InterpolationMode.NEAREST_EXACT
        )
        return img_resized, mask_resized

    def __getitem__(self, idx):
        img = v2.functional.to_image(default_loader(self.df.iloc[idx][self.img_col]))
        mask = v2.functional.to_image(default_loader(self.df.iloc[idx][self.mask_col]))

        # Sync flip
        state = torch.get_rng_state()
        img = self.flip_transforms(img)
        torch.set_rng_state(state)
        mask = self.flip_transforms(mask)

        # Custom crop (output = 360)
        if self.crop:
            img, mask = self.random_resized_crop(img, mask)

        # Color jitter
        if self.color:
            img = v2.ColorJitter(brightness=0.2, contrast=0.2)(img)

        # Normalize & dtype
        img = v2.functional.to_dtype(img, torch.float32, scale=True)
        img = v2.functional.normalize(
            img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # ✅ FIXED: use configurable backbone_res instead of hardcoded 512
        img = v2.functional.resize(img, (self.backbone_res, self.backbone_res), antialias=True)

        mask = mask.long()[0]
        return {"input": img, "mask": mask}

# --- METRICS HELPERS ---

def compute_conf_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    return torch.bincount(
        num_classes * target[mask].view(-1) + pred[mask].view(-1),
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)

def calculate_metrics(conf_matrix, ignore_index=None):
    ious = []
    conf_matrix = conf_matrix.float()
    num_classes = conf_matrix.shape[0]
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else torch.tensor(float('nan'))
        ious.append(iou.item())
    relevant_ious = [iou for i, iou in enumerate(ious) if i != ignore_index and not np.isnan(iou)]
    miou = np.mean(relevant_ious) if relevant_ious else 0
    return miou, ious

def validate(model, loader, criterion, device, num_classes, ignore_index, stage="val"):
    model.eval()
    total_loss, conf_matrix = 0, torch.zeros(num_classes, num_classes, device=device)
    with torch.no_grad():
        for item in tqdm(loader, desc=f"evaluating_{stage}", leave=False):
            images, masks = item['input'].to(device), item['mask'].to(device).long()
            with torch.autocast(device, dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, masks)
            total_loss += loss.item()
            conf_matrix += compute_conf_matrix(torch.argmax(outputs, dim=1), masks, num_classes)
    avg_loss = total_loss / len(loader)
    miou, class_ious = calculate_metrics(conf_matrix, ignore_index=ignore_index)
    return avg_loss, miou, class_ious

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Folder naming based on augmentations ---
    aug_suffix = ""
    if args.use_crop:   aug_suffix += "_crop"
    if args.use_color: aug_suffix += "_color"
    if args.no_flip:   aug_suffix += "_noflip"

    exp_name = f"{args.exp_name}{aug_suffix}"

    run_dir = os.path.join(args.save_dir, f"{exp_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(__file__, os.path.join(run_dir, os.path.basename(__file__)))
    sys.stdout = Logger(os.path.join(run_dir, "log.txt"))

    print(f"--- Experiment: {exp_name} ---")
    print(f"Arguments: {json.dumps(vars(args), indent=4)}")

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    writer = SummaryWriter(log_dir=run_dir)

    df = pd.read_csv(args.data_path)
    df['img_path'] = args.data_root + df['img_name']
    df['mask_path'] = df['img_path'].apply(lambda x: x.replace(".jpg", "_mask.png"))

    # Build transforms for input images (note: resized to 512 as before)
    transform = v2.Compose([
        v2.ToImage(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    seg_model = Mask2FormerWrapper(args.model_name, args.num_classes, args.out_resolution).to(args.device)

    if args.freeze_encoder:
        for p in seg_model.model.model.backbone.parameters(): p.requires_grad = False
    if args.freeze_decoder:
        for p in seg_model.model.model.pixel_decoder.parameters(): p.requires_grad = False
        for p in seg_model.model.model.transformer_module.parameters(): p.requires_grad = False

    trainable_params = [p for p in seg_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # simplified per request

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=min(1e-6, args.lr / 100))

    # Pass augmentation flags to dataset
    train_loader = DataLoader(
        SegDataset(df[df['split']=="train"],
                   backbone_res=args.resolution,  # ← use CLI arg
                   pflip=args.pflip if not args.no_flip else 0.0,
                   flip=not args.no_flip,
                   crop=args.use_crop, color=args.use_color),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        SegDataset(df[df['split']=="val"],
                   backbone_res=args.resolution,
                   pflip=0, flip=False, crop=False, color=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        SegDataset(df[df['split']=="test"],
                   backbone_res=args.resolution,
                   pflip=0, flip=False, crop=False, color=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    global_step = 0
    for epoch in range(args.epochs):
        seg_model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0

        for item in pbar:
            images, masks = item['input'].to(args.device), item['mask'].to(args.device).long()
            optimizer.zero_grad()

            with torch.autocast(args.device, dtype=torch.bfloat16):
                loss = criterion(seg_model(images), masks)

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

            writer.add_scalar("loss_train_step", loss.item(), global_step)
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        val_loss, val_miou, val_ious = validate(seg_model, val_loader, criterion,
                                                args.device, args.num_classes, args.ignore_index, "val")
        test_loss, test_miou, test_ious = validate(seg_model, test_loader, criterion,
                                                   args.device, args.num_classes, args.ignore_index, "test")

        writer.add_scalar("loss_train_epoch", avg_train_loss, epoch)
        writer.add_scalar("loss_val", val_loss, epoch)
        writer.add_scalar("loss_test", test_loss, epoch)
        writer.add_scalar("miou_val", val_miou, epoch)
        writer.add_scalar("miou_test", test_miou, epoch)

        for i, iou in enumerate(val_ious):
            if i != args.ignore_index:
                writer.add_scalar(f"iou_val_class_{i}", iou, epoch)
        for i, iou in enumerate(test_ious):
            if i != args.ignore_index:
                writer.add_scalar(f"iou_test_class_{i}", iou, epoch)

        log_msg = (f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | "
                   f"Test mIoU: {test_miou:.4f}")
        print(log_msg)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': seg_model.state_dict(),
            'val_miou': val_miou
        }, os.path.join(run_dir, f"model_epoch={epoch+1:03d}.ckpt"))

    writer.close()