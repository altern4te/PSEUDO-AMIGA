"""
Runs SAM2 propagation over extracted frames for 2 object classes
at once (e.g. crop vs weed), then opens an interactive window so user can
inspect every frame, click corrections for whichever class is active, and
save accepted masks as U-Net training data.

Controls:
    1, 2         -> switch the "active" class (the one your clicks edit)
    Left click   -> add positive point for the active class
    Right click  -> add negative point for the active class
    Enter        -> accept current combined mask, save pair, advance
    X            -> reject frame (saves an all-background mask), advance
    C            -> clear your correction points for the ACTIVE class only,nreverting it to the original propagated mask
    Backspace    -> go back one frame (to re-review / re-correct)
    R            -> re-propagate forward from this frame for ALL classes
    Q            -> save progress and quit

I screwed up some of the mask naming in the original so do a run and double check to make sure its outputtig correctly. 
I would test if my fix in this worked but I just really dont want to and already have all my masks fixed. Sorry not sorry.
"""

import os
import json
import shutil
import copy

import numpy as np
import torch
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass  # fall back to whatever default backend is available

import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# ----------------------------- CONFIG ---------------------------------
sam2_checkpoint = r"$PATH\sam2-main\sam2-main\checkpoints\sam2.1_hiera_large.pt"
model_cfg = r"$PATH\sam2-main\sam2-main\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
video_dir = r"$PATH\sam2-main\processed_images\GF-NIRNDVI"
output_dir = r"$PATH\sam2-main\unet_dataset"

IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG")

CLASSES = {
    1: {"name": "crop", "color": (0.0, 1.0, 0.0, 0.5)},   # green
    2: {"name": "weed", "color": (1.0, 0.0, 0.0, 0.5)},   # red
}
SEED_FRAME_IDX = 0


OBJ_IDS = sorted(CLASSES.keys())

os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
progress_path = os.path.join(output_dir, "progress.json")

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

predictor = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=sam2_checkpoint)

frame_names = [p for p in os.listdir(video_dir) if p.endswith(IMG_EXTS)]
frame_names.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[1])) # Change SAM V2 misc.py to match this notation if error occurs

def load_progress():
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {"accepted": [], "rejected": [], "current_idx": 0}


def save_progress(state):
    with open(progress_path, "w") as f:
        json.dump(state, f, indent=2)


def save_pair(frame_idx, class_masks):
    name = frame_names[frame_idx]
    stem = os.path.splitext(name)[0]
    shutil.copy2(os.path.join(video_dir, name), os.path.join(output_dir, "images", name))

    h, w = Image.open(os.path.join(video_dir, name)).size[::-1]
    combined = np.zeros((h, w), dtype=np.uint8)
    assigned = np.zeros((h, w), dtype=bool)
    for obj_id in OBJ_IDS:
        m = class_masks.get(obj_id)
        if m is None:
            continue
        m2d = np.squeeze(np.asarray(m)).astype(bool)
        overlap = np.logical_and(assigned, m2d).sum()
        if overlap > 0:
            print(f"  warning: {overlap}px overlap on frame {frame_idx} -- "
                  f"class {obj_id} ({CLASSES[obj_id]['name']}) overwrote a previous class there")
        combined[m2d] = obj_id
        assigned |= m2d

    Image.fromarray(combined, mode="L").save(os.path.join(output_dir, "masks", f"{stem}_mask.png"))


def _draw_points(ax, points, labels):
    if not len(points):
        return
    pos = points[labels == 1]
    neg = points[labels == 0]
    if len(pos):
        ax.scatter(pos[:, 0], pos[:, 1], c="cyan", marker="o", s=35, edgecolor="black", linewidth=0.7)
    if len(neg):
        ax.scatter(neg[:, 0], neg[:, 1], c="magenta", marker="o", s=35, edgecolor="black", linewidth=0.7)


class SeedSelector:
    def __init__(self, inference_state, seed_frame_idx):
        self.state = inference_state
        self.frame_idx = seed_frame_idx
        self.active_obj_id = OBJ_IDS[0]
        self.points = {obj_id: np.empty((0, 2), dtype=np.float32) for obj_id in OBJ_IDS}
        self.labels = {obj_id: np.empty((0,), dtype=np.int32) for obj_id in OBJ_IDS}
        self.current_masks = {}
        self.confirmed = False

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()
        plt.show()

    def redraw(self):
        self.ax.clear()
        img = Image.open(os.path.join(video_dir, frame_names[self.frame_idx]))
        self.ax.imshow(img)

        for obj_id, mask in self.current_masks.items():
            arr = np.asarray(mask)
            h, w = arr.shape[-2:]
            color = np.array(CLASSES[obj_id]["color"])
            overlay = arr.reshape(h, w, 1) * color.reshape(1, 1, -1)
            self.ax.imshow(overlay)

        _draw_points(self.ax, self.points[self.active_obj_id], self.labels[self.active_obj_id])

        active_name = CLASSES[self.active_obj_id]["name"]
        class_key = "/".join(f"{k}={v['name']}" for k, v in CLASSES.items())
        seeded = ", ".join(f"{CLASSES[o]['name']}={'yes' if len(self.points[o]) else 'no'}" for o in OBJ_IDS)
        self.ax.set_title(
            f"SEED FRAME {self.frame_idx} -- Place starting points, then Enter to propagate\n"
            f"Editing class {self.active_obj_id} ({active_name})  |  switch: {class_key}  |  seeded: {seeded}\n"
            "L-Click=+  R-Click=-  C=Clear Active Class  Enter=Confirm & Propagate  Q=Quit",
            fontsize=8,
        )
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        label = 1 if event.button == 1 else 0
        obj = self.active_obj_id
        pt = np.array([[event.xdata, event.ydata]], dtype=np.float32)
        self.points[obj] = np.concatenate([self.points[obj], pt], axis=0)
        self.labels[obj] = np.concatenate([self.labels[obj], [label]])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=self.state,
                frame_idx=self.frame_idx,
                obj_id=obj,
                points=self.points[obj],
                labels=self.labels[obj],
            )
        for i, oid in enumerate(out_obj_ids):
            self.current_masks[oid] = (out_mask_logits[i] > 0.0).cpu().numpy()
        self.redraw()

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in [str(k) for k in OBJ_IDS]:
            self.active_obj_id = int(key)
            self.redraw()
        elif key == "c":
            obj = self.active_obj_id
            self.points[obj] = np.empty((0, 2), dtype=np.float32)
            self.labels[obj] = np.empty((0,), dtype=np.int32)
            self.current_masks.pop(obj, None)
            self.redraw()
        elif key == "enter":
            if any(len(self.points[o]) for o in OBJ_IDS):
                self.confirmed = True
                plt.close(self.fig)
            else:
                print("Place at least one point for at least one class before confirming.")
        elif key == "q":
            self.confirmed = False
            plt.close(self.fig)


class MaskReviewer:
    def __init__(self, inference_state, video_segments):
        self.state = inference_state
        self.video_segments = video_segments
        self.original_segments = copy.deepcopy(video_segments)
        self.progress = load_progress()
        self.active_obj_id = OBJ_IDS[0]
        self.points = {}
        self.labels = {}
        self.current_masks = {}

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load_frame(self.progress["current_idx"])
        plt.show()

    def load_frame(self, idx):
        if idx >= len(frame_names):
            print("All frames reviewed.")
            self.print_summary()
            plt.close(self.fig)
            return
        self.frame_idx = idx
        self.points = {obj_id: np.empty((0, 2), dtype=np.float32) for obj_id in OBJ_IDS}
        self.labels = {obj_id: np.empty((0,), dtype=np.int32) for obj_id in OBJ_IDS}
        self.current_masks = dict(self.video_segments.get(idx, {}))
        self.redraw()

    def redraw(self):
        self.ax.clear()
        img = Image.open(os.path.join(video_dir, frame_names[self.frame_idx]))
        self.ax.imshow(img)

        for obj_id, mask in self.current_masks.items():
            if mask is None:
                continue
            arr = np.asarray(mask)
            h, w = arr.shape[-2:]
            color = np.array(CLASSES[obj_id]["color"])
            overlay = arr.reshape(h, w, 1) * color.reshape(1, 1, -1)
            self.ax.imshow(overlay)

        _draw_points(self.ax, self.points[self.active_obj_id], self.labels[self.active_obj_id])

        n_done = len(self.progress["accepted"]) + len(self.progress["rejected"])
        active_name = CLASSES[self.active_obj_id]["name"]
        class_key = "/".join(f"{k}={v['name']}" for k, v in CLASSES.items())
        self.ax.set_title(
            f"Frame {self.frame_idx} ({frame_names[self.frame_idx]})  [{n_done}/{len(frame_names)} reviewed]\n"
            f"Editing class {self.active_obj_id} ({active_name})  |  switch: {class_key}\n"
            "L-click=+  R-click=-  Enter=accept  X=reject  C=clear active class  "
            "Backspace=prev  R=repropagate  Q=save&quit",
            fontsize=8,
        )
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        label = 1 if event.button == 1 else 0  # left = positive, right = negative
        obj = self.active_obj_id
        pt = np.array([[event.xdata, event.ydata]], dtype=np.float32)
        self.points[obj] = np.concatenate([self.points[obj], pt], axis=0)
        self.labels[obj] = np.concatenate([self.labels[obj], [label]])
        self.refine(obj)

    def refine(self, obj_id):
        if len(self.points[obj_id]) == 0:
            return
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=self.state,
                frame_idx=self.frame_idx,
                obj_id=obj_id,
                points=self.points[obj_id],
                labels=self.labels[obj_id],
            )
        for i, oid in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            self.current_masks[oid] = mask
            self.video_segments.setdefault(self.frame_idx, {})[oid] = mask
        self.redraw()

    def accept(self):
        save_pair(self.frame_idx, self.current_masks)
        if self.frame_idx not in self.progress["accepted"]:
            self.progress["accepted"].append(self.frame_idx)
        self.advance()

    def reject(self):
        save_pair(self.frame_idx, {})
        if self.frame_idx not in self.progress["rejected"]:
            self.progress["rejected"].append(self.frame_idx)
        self.advance()

    def repropagate(self):
        print(f"Re-propagating forward from frame {self.frame_idx} for all classes...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state=self.state, start_frame_idx=self.frame_idx
            ):
                self.video_segments[out_frame_idx] = {
                    oid: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, oid in enumerate(out_obj_ids)
                }
        self.current_masks = dict(self.video_segments.get(self.frame_idx, {}))
        self.redraw()
        print("Done.")

    def advance(self):
        self.progress["current_idx"] = self.frame_idx + 1
        save_progress(self.progress)
        self.load_frame(self.frame_idx + 1)

    def print_summary(self):
        print(f"Accepted: {len(self.progress['accepted'])}  Rejected: {len(self.progress['rejected'])}")
        print(f"Dataset written to: {output_dir}")

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in [str(k) for k in OBJ_IDS]:
            self.active_obj_id = int(key)
            self.redraw()
        elif key == "enter":
            self.accept()
        elif key == "x":
            self.reject()
        elif key == "c":
            obj = self.active_obj_id
            self.points[obj] = np.empty((0, 2), dtype=np.float32)
            self.labels[obj] = np.empty((0,), dtype=np.int32)
            orig = self.original_segments.get(self.frame_idx, {}).get(obj)
            if orig is not None:
                self.current_masks[obj] = orig
                self.video_segments.setdefault(self.frame_idx, {})[obj] = orig
            else:
                self.current_masks.pop(obj, None)
                self.video_segments.get(self.frame_idx, {}).pop(obj, None)
            self.redraw()
        elif key == "backspace":
            if self.frame_idx > 0:
                self.load_frame(self.frame_idx - 1)
        elif key == "r":
            self.repropagate()
        elif key == "q":
            save_progress(self.progress)
            print("Progress saved. Closing.")
            plt.close(self.fig)


if __name__ == "__main__":
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)

    seeder = SeedSelector(inference_state, seed_frame_idx=SEED_FRAME_IDX)

    if not seeder.confirmed:
        print("No seeds confirmed -- exiting without running propagation.")
    else:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    oid: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, oid in enumerate(out_obj_ids)
                }
        MaskReviewer(inference_state, video_segments)