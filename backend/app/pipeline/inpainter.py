import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


def get_onnx_providers() -> List[str]:
    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


class LamaInpainter:
    """
    Full-resolution patch-based LaMa ONNX inpainter.
    Auto-downloads `lama_fp32.onnx` (~208 MB) on first use and caches it under `settings.models_dir`.
    """

    def __init__(self) -> None:
        self._session = None
        self._model_path: Optional[Path] = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_failed = False

    def model_path(self) -> Path:
        return settings.models_dir / "lama" / settings.lama_hf_filename

    def is_downloaded(self) -> bool:
        p = self.model_path()
        return p.is_file() and p.stat().st_size > 1_000_000

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if self._load_failed:
            return False

        with self._lock:
            if self._session is not None:
                return True

            target_path = self.model_path()
            if not self.is_downloaded():
                try:
                    from huggingface_hub import hf_hub_download

                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        "Downloading LaMa ONNX model (%s/%s)...",
                        settings.lama_hf_repo,
                        settings.lama_hf_filename,
                    )
                    downloaded = hf_hub_download(
                        repo_id=settings.lama_hf_repo,
                        filename=settings.lama_hf_filename,
                        local_dir=str(target_path.parent),
                    )
                    self._model_path = Path(downloaded)
                except Exception as exc:
                    logger.warning("Could not download LaMa ONNX model (%s); using OpenCV fallback.", exc)
                    self._load_failed = True
                    return False
            else:
                self._model_path = target_path

            try:
                import onnxruntime as ort

                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
                sess_opts.inter_op_num_threads = 1
                providers = get_onnx_providers()
                logger.info("Loading LaMa ONNX session on %s...", providers[0])
                self._session = ort.InferenceSession(
                    str(self._model_path),
                    sess_options=sess_opts,
                    providers=providers,
                )
                return True
            except Exception as exc:
                logger.warning("Failed initializing LaMa ONNX session (%s); using OpenCV fallback.", exc)
                self._load_failed = True
                return False

    def _run_lama_512(self, patch_rgb: np.ndarray, patch_mask: np.ndarray) -> np.ndarray:
        """Run fixed 512x512 LaMa ONNX inference on a single patch and return uint8 RGB at original patch size."""
        orig_h, orig_w = patch_rgb.shape[:2]

        img_512 = cv2.resize(patch_rgb, (512, 512), interpolation=cv2.INTER_AREA if orig_w > 512 else cv2.INTER_CUBIC)
        mask_512 = cv2.resize(patch_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        img_in = (img_512.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        mask_in = (mask_512 > 127).astype(np.float32)[None, None, ...]

        input_names = [inp.name for inp in self._session.get_inputs()]
        feed = {input_names[0]: img_in, input_names[1]: mask_in}
        # Serialize heavy FP32 LaMa inference to prevent multi-GB RAM spikes & OOM on 8GB hosts
        with self._infer_lock:
            out = self._session.run(None, feed)[0][0]  # (3, 512, 512)

        # Normalize output range cleanly whether model outputs [0..255] or [0..1]
        if out.max() <= 1.5:
            out = out * 255.0
        out_rgb = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)

        if (orig_h, orig_w) != (512, 512):
            out_rgb = cv2.resize(out_rgb, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        return out_rgb

    def _find_mask_clusters(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Group nearby mask regions into patch bounding boxes for high-res local inpainting."""
        h, w = mask.shape[:2]
        # Scale merge kernel with image resolution so high-res manga pages don't spawn 15+ patches
        k_size = max(45, min(115, int(max(h, w) * 0.065))) | 1
        merge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        merged = cv2.dilate(mask, merge_k)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

        boxes: List[Tuple[int, int, int, int]] = []
        for idx in range(1, num_labels):
            x, y, bw, bh, area = stats[idx]
            if area < 4:
                continue
            boxes.append((int(x), int(y), int(x + bw), int(y + bh)))

        # If the whole image is modest in size (<= 768px max side), one full-frame pass is faster and cleaner
        if max(h, w) <= 768 and len(boxes) > 0:
            return [(0, 0, w, h)]

        # Cap max LaMa patches per page at 5 by iteratively merging the two closest boxes
        while len(boxes) > 5:
            best_pair = (0, 1)
            best_dist = float("inf")
            for i in range(len(boxes)):
                cx_i = (boxes[i][0] + boxes[i][2]) * 0.5
                cy_i = (boxes[i][1] + boxes[i][3]) * 0.5
                for j in range(i + 1, len(boxes)):
                    cx_j = (boxes[j][0] + boxes[j][2]) * 0.5
                    cy_j = (boxes[j][1] + boxes[j][3]) * 0.5
                    d = (cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_pair = (i, j)
            i, j = best_pair
            a, b = boxes[i], boxes[j]
            merged_box = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
            boxes[i] = merged_box
            boxes.pop(j)

        return boxes

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        use_lama: bool = True,
    ) -> np.ndarray:
        """
        Inpaint masked pixels (`mask > 127`) while leaving all unmasked pixels 100% untouched.
        Uses patch-based LaMa ONNX for high-res fidelity, with OpenCV Telea fallback if offline.
        """
        if mask is None or np.count_nonzero(mask > 127) == 0:
            return image_rgb.copy()

        binary_mask = ((mask > 127).astype(np.uint8)) * 255

        if use_lama and self._ensure_loaded():
            h, w = image_rgb.shape[:2]
            result = image_rgb.copy()
            clusters = self._find_mask_clusters(binary_mask)

            for cx1, cy1, cx2, cy2 in clusters:
                cw = cx2 - cx1
                ch = cy2 - cy1
                # Create a square context crop with generous surrounding background (>= 1.8x mask size, min 384px)
                side = max(384, int(max(cw, ch) * 1.85))
                side = min(side, max(h, w))

                center_x = (cx1 + cx2) // 2
                center_y = (cy1 + cy2) // 2
                px1 = max(0, center_x - side // 2)
                py1 = max(0, center_y - side // 2)
                px2 = min(w, px1 + side)
                py2 = min(h, py1 + side)
                px1 = max(0, px2 - side)
                py1 = max(0, py2 - side)

                patch_img = result[py1:py2, px1:px2]
                patch_mask = binary_mask[py1:py2, px1:px2]
                if np.count_nonzero(patch_mask) == 0:
                    continue

                inpainted_patch = self._run_lama_512(patch_img, patch_mask)

                # Soft 2px feather at mask boundary so transition into original pixels is invisible
                alpha = cv2.GaussianBlur(patch_mask.astype(np.float32) / 255.0, (5, 5), 0)
                alpha = np.clip(alpha * 1.15, 0.0, 1.0)[..., None]
                blended = (inpainted_patch.astype(np.float32) * alpha + patch_img.astype(np.float32) * (1.0 - alpha))
                result[py1:py2, px1:px2] = np.clip(blended, 0, 255).astype(np.uint8)

            return result

        # OpenCV Telea fallback (fast, zero-download, works offline)
        img_bgr = image_rgb[:, :, ::-1]
        inpainted_bgr = cv2.inpaint(img_bgr, binary_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return inpainted_bgr[:, :, ::-1].copy()


inpainter = LamaInpainter()
