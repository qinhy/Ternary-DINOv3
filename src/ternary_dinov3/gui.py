from PIL import Image
from matplotlib.path import Path
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def denorm_to_numpy(img_chw, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    img_chw: torch.Tensor (3, H, W), normalized
    returns: numpy (H, W, 3), float32 in [0,1]
    """
    img = img_chw.detach().cpu().float().clone()
    mean_t = torch.tensor(mean, dtype=img.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(3, 1, 1)
    img = img * std_t + mean_t
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()  # HWC
    return img

def patchtokens_to_featmap(patchtokens, H, W, patch_size=16):
    """
    patchtokens: (B, N, C) where N = (H/P)*(W/P)
    returns:     (B, C, H/P, W/P)
    """
    B, N, C = patchtokens.shape
    Gh, Gw = H // patch_size, W // patch_size
    assert N == Gh * Gw, f"N={N}, expected {Gh*Gw} for H={H}, W={W}, P={patch_size}"
    featmap = patchtokens.reshape(B, Gh, Gw, C).permute(0, 3, 1, 2).contiguous()
    return featmap

class DinoPatchSimilarityViewer:
    """
    Interactive viewer for patch-level self-similarity maps.

    Inputs:
      image:  HxWx3 numpy array (uint8 or float in [0,1])
      feats:  torch.Tensor of shape (B, C, Gh, Gw)
    """
    def __init__(self, image, feats, batch_idx=0, title="DINO patch similarity"):
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array (H, W, 3)")
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("image must have shape (H, W, 3) or (H, W, 4)")
        if not torch.is_tensor(feats):
            raise TypeError("feats must be a torch.Tensor")
        if feats.ndim != 4:
            raise ValueError("feats must have shape (B, C, Gh, Gw)")

        self.image = image[..., :3].copy()
        self.H, self.W = self.image.shape[:2]

        # keep features on original device (GPU is fine)
        self.feats = feats
        self.B, self.C, self.Gh, self.Gw = feats.shape
        self.batch_idx = int(np.clip(batch_idx, 0, self.B - 1))

        # normalize once for cosine similarity
        self.feats_n = F.normalize(self.feats, dim=1)

        # state
        self.gx = self.Gw // 2
        self.gy = self.Gh // 2
        self.alpha = 0.45
        self.show_overlay = True
        self.percentile_clip = (2, 98)  # for nicer visualization

        # figure
        self.fig, (self.ax_img, self.ax_map) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title(title) if hasattr(self.fig.canvas.manager, "set_window_title") else None
        self.fig.suptitle(title)

        self.ax_img.set_title("Image (click to select patch)")
        self.ax_map.set_title("Similarity map")

        # base image
        self.im_img = self.ax_img.imshow(self.image)
        self.im_overlay = self.ax_img.imshow(
            np.zeros((self.H, self.W), dtype=np.float32),
            cmap="jet",
            alpha=self.alpha,
            vmin=0.0,
            vmax=1.0,
        )
        self.im_overlay.set_visible(self.show_overlay)

        # patch grid marker
        self.marker = self.ax_img.plot([], [], marker="o", markersize=8, markeredgecolor="white",
                                       markerfacecolor="none", markeredgewidth=2)[0]

        # map panel
        self.im_map = self.ax_map.imshow(
            np.zeros((self.Gh, self.Gw), dtype=np.float32),
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest"
        )
        self.marker_map = self.ax_map.plot([], [], marker="s", markersize=9, markeredgecolor="white",
                                           markerfacecolor="none", markeredgewidth=2)[0]
        self.cbar = self.fig.colorbar(self.im_map, ax=self.ax_map, fraction=0.046, pad=0.04)
        self.cbar.set_label("relative score")

        self.ax_img.set_xlim([0, self.W - 1])
        self.ax_img.set_ylim([self.H - 1, 0])
        self.ax_map.set_xlim([-0.5, self.Gw - 0.5])
        self.ax_map.set_ylim([self.Gh - 0.5, -0.5])

        # help text
        help_text = (
            "Click image/map: select patch\n"
            "←/→: batch   +/-: overlay alpha\n"
            "o: toggle overlay   c: reset contrast"
        )
        self.text = self.fig.text(0.02, 0.02, help_text, fontsize=9)

        # connect events
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # initial draw
        self._update_view()

    def _grid_to_image_xy(self, gx, gy):
        # patch center in image pixel coords
        x = (gx + 0.5) * (self.W / self.Gw)
        y = (gy + 0.5) * (self.H / self.Gh)
        return x, y

    def _image_xy_to_grid(self, x, y):
        gx = int(np.clip(np.floor(x * self.Gw / self.W), 0, self.Gw - 1))
        gy = int(np.clip(np.floor(y * self.Gh / self.H), 0, self.Gh - 1))
        return gx, gy

    def _compute_score(self):
        """
        Returns:
          score_grid: (Gh, Gw) float tensor in [-1,1]
          score_vis_grid: (Gh, Gw) float tensor in [0,1] (for display)
          score_vis_img: (H, W) float tensor in [0,1] (upsampled)
        """
        b = self.batch_idx
        # feats_n[b]: (C, Gh, Gw)
        ref = self.feats_n[b, :, self.gy, self.gx]  # (C,)
        score = (self.feats_n[b] * ref[:, None, None]).sum(dim=0)  # (Gh, Gw), cosine sim

        # robust normalization for visualization
        s = score.detach().float()
        lo = torch.quantile(s.flatten(), self.percentile_clip[0] / 100.0)
        hi = torch.quantile(s.flatten(), self.percentile_clip[1] / 100.0)
        s_vis = (s - lo) / (hi - lo + 1e-6)
        s_vis = s_vis.clamp(0, 1)

        # upscale to image size for overlay
        s_img = F.interpolate(
            s_vis[None, None],
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        return score, s_vis, s_img

    def _update_view(self):
        score, s_vis, s_img = self._compute_score()

        # update overlay on image
        self.im_overlay.set_data(s_img.detach().cpu().numpy())
        self.im_overlay.set_alpha(self.alpha)
        self.im_overlay.set_visible(self.show_overlay)

        # update map panel
        self.im_map.set_data(s_vis.detach().cpu().numpy())

        # markers
        x, y = self._grid_to_image_xy(self.gx, self.gy)
        self.marker.set_data([x], [y])
        self.marker_map.set_data([self.gx], [self.gy])

        # titles
        raw_val = float(score[self.gy, self.gx].item())  # should be ~1.0
        self.ax_img.set_title(
            f"Image (B={self.batch_idx}/{self.B-1}) | patch=({self.gx},{self.gy})"
        )
        self.ax_map.set_title(f"Similarity map (self={raw_val:.3f})")

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        if event.inaxes == self.ax_img:
            gx, gy = self._image_xy_to_grid(event.xdata, event.ydata)
            self.gx, self.gy = gx, gy
            self._update_view()
        elif event.inaxes == self.ax_map:
            gx = int(np.clip(round(event.xdata), 0, self.Gw - 1))
            gy = int(np.clip(round(event.ydata), 0, self.Gh - 1))
            self.gx, self.gy = gx, gy
            self._update_view()

    def _on_key(self, event):
        key = event.key
        if key in ("left", "right"):
            if key == "left":
                self.batch_idx = (self.batch_idx - 1) % self.B
            else:
                self.batch_idx = (self.batch_idx + 1) % self.B
            self._update_view()

        elif key in ("+", "="):
            self.alpha = min(1.0, self.alpha + 0.05)
            self._update_view()

        elif key == "-":
            self.alpha = max(0.0, self.alpha - 0.05)
            self._update_view()

        elif key == "o":
            self.show_overlay = not self.show_overlay
            self._update_view()

        elif key == "c":
            self.percentile_clip = (2, 98)
            self._update_view()

    def show(self):
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_images(model, images:list[Path],
                    size=224,batch_size=8):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        class ImagesSet(Dataset):
            def __init__(self, imgs: list[Path], size: int):
                self.paths = sorted(imgs)
                if not self.paths:
                    raise FileNotFoundError(f"No .jpg files found")
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize((size, size), antialias=True),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
                )

            def __len__(self) -> int:
                return len(self.paths)

            def __getitem__(self, index: int):
                with Image.open(self.paths[index]) as im:
                    x = self.transform(im.convert("RGB"))
                return index, x
            
        model.eval()        
        loader = DataLoader(
            ImagesSet(imgs=images,size=size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        for step, (row_ids, images) in enumerate(loader):
            with torch.no_grad():
                out = model.forward_features(images)
                patchtokens = out["x_norm_patchtokens"]   # usually (B, N, C)

            # Get image size from tensor
            _, _, H, W = images.shape

            # Reshape tokens -> (B, C, Gh, Gw)
            featmaps = patchtokens_to_featmap(patchtokens, H, W, patch_size=16)

            # Visualize the first sample
            image_np = denorm_to_numpy(images[0])        # (H, W, 3)
            feats_1 = featmaps[0:1]                      # keep batch dim: (1, C, Gh, Gw)

            viewer = DinoPatchSimilarityViewer(image_np, feats_1)
            viewer.show()