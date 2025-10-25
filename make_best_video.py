#!/usr/bin/env python3
"""
Crisp, squashed, slow-motion Atari recorder with solid colors (no blur).
- Squashes by altering output height (no cropping)
- Slows playback by duplicating frames (SLOW_FACTOR)
- Quantizes colors to a small palette (PALETTE_SIZE) using k-means -> solid color blocks
- Upscales using INTER_NEAREST to keep pixels crisp (no blur)
- Uses imageio/ffmpeg if available; falls back to OpenCV
"""

# ! This code is AI-generated

import os
from pathlib import Path
import shutil
import numpy as np
import cv2
import imageio
import torch
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing
from collections import deque

from actor_critic import ActorCritic  # local module

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_MAP = [0, 2, 3]  # NOOP, LEFT, RIGHT

FPS = 30  # original logical FPS (we'll duplicate frames for slowdown)
SCALE = 3  # integer upscale factor (nearest neighbor - crisp pixels)
SLOW_FACTOR = 2  # duplicate each frame this many times -> 4 = quarter speed
PALETTE_SIZE = 10  # number of solid colors to quantize to
SAVE_THRESHOLD = 40  # save every episode (set to 30 if you want threshold)
OUT_DIR = Path("videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Squash: user wants vertical squashed look. We'll reduce output height proportionally.
# For example, squashed_height_factor 0.6 means final height = original_height * 0.6 * SCALE
SQUASH_HEIGHT_FACTOR = 0.6  # 0.6 => squashed vertically (less tall)


# ---------------- HELPERS ----------------
def preprocess_obs(obs, crop_top=18):
    """Policy preprocessing (unchanged)."""
    obs = cv2.resize(obs, (84, 110))
    return obs[crop_top : crop_top + 84, :]


def validate_frames(frames):
    if len(frames) == 0:
        return False, "no frames"
    shapes = [f.shape for f in frames]
    if len(set(shapes)) != 1:
        return False, f"inconsistent shapes: {set(shapes)}"
    if frames[0].dtype != np.uint8:
        return False, f"unexpected dtype {frames[0].dtype}, expected uint8"
    mn = int(np.min(frames[0]))
    mx = int(np.max(frames[0]))
    if mn < 0 or mx > 255:
        return False, f"pixel values out of 0-255 (min {mn}, max {mx})"
    return True, "ok"


def quantize_frame_kmeans(rgb_frame, k=6, subsample=1000, rng_seed=42):
    """
    Quantize RGB frame to k colors using k-means.
    - subsample: how many pixels to sample for kmeans initialization (performance).
    Returns quantized uint8 RGB image with exact cluster centroids (solid colors).
    """
    h, w, c = rgb_frame.shape
    assert c == 3
    img_flat = rgb_frame.reshape((-1, 3)).astype(np.float32)

    # Subsample for kmeans init if image large
    rng = np.random.default_rng(rng_seed)
    if subsample and img_flat.shape[0] > subsample:
        idx = rng.choice(img_flat.shape[0], size=subsample, replace=False)
        sample = img_flat[idx]
    else:
        sample = img_flat

    # OpenCV kmeans expects float32
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    attempts = 3
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels_sample, centers = cv2.kmeans(
        sample, k, None, term_crit, attempts, flags
    )

    # assign every pixel to nearest center (fast by computing distances)
    # centers shape (k,3)
    # compute distances in vectorized form
    # note: for small images this is fine; optimize later if necessary
    dists = np.linalg.norm(
        img_flat[:, None, :] - centers[None, :, :], axis=2
    )  # (n_pixels, k)
    labels = np.argmin(dists, axis=1)
    quant_flat = centers[labels].astype(np.uint8)
    quant = quant_flat.reshape((h, w, 3))
    return quant


def upscale_nearest(frame, scale=1, squash_factor=1.0):
    """
    Nearest-neighbor upscale to keep pixels crisp.
    squash_factor reduces height proportionally (value <1 squashes vertically).
    """
    if scale == 1 and abs(squash_factor - 1.0) < 1e-6:
        return frame
    h, w = frame.shape[:2]
    new_h = int(round(h * squash_factor * scale))
    new_w = w * scale
    # use INTER_NEAREST to avoid blur
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def duplicate_frames(frames, factor=1):
    if factor <= 1:
        return frames
    dup = []
    for f in frames:
        for _ in range(factor):
            dup.append(f)
    return dup


def write_video_imageio(fname, frames_rgb, fps=30):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install ffmpeg for best results.")
    # imageio writer expects RGB uint8 frames
    writer = imageio.get_writer(
        str(fname),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"],
    )
    try:
        for fr in frames_rgb:
            writer.append_data(fr)
    finally:
        writer.close()


def write_video_opencv(fname, frames_bgr, fps=30):
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(fname), fourcc, fps, (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()


# ---------------- ENV & POLICY ----------------
gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = AtariPreprocessing(
    env, grayscale_obs=True, scale_obs=True, frame_skip=1, terminal_on_life_loss=True
)

policy = ActorCritic().to(DEVICE)
model_path = Path("models/model_56250.pth")
if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found at {model_path}. Place trained model there."
    )
policy.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
policy.eval()

# ---------------- MAIN LOOP ----------------
episode_idx = 0
obs, info = env.reset()
# FIRE to start
obs, _, _, _, _ = env.step(1)
obs, _, _, _, _ = env.step(1)

frame_stack = deque([preprocess_obs(obs)] * 4, maxlen=4)

total_reward = 0.0
frames = []

print(
    "Recording (Ctrl+C to stop). Settings: SCALE={}, SLOW_FACTOR={}, PALETTE_SIZE={}, SQUASH_HEIGHT_FACTOR={}".format(
        SCALE, SLOW_FACTOR, PALETTE_SIZE, SQUASH_HEIGHT_FACTOR
    )
)
try:
    while True:
        rendered = env.render()
        if rendered is not None:
            # ensure uint8 RGB
            if rendered.dtype != np.uint8:
                rendered = np.clip(rendered, 0, 255).astype(np.uint8)
            frames.append(rendered.copy())

        state = np.stack(frame_stack, axis=0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action, _, _ = policy(state_tensor)

        action_env = ACTION_MAP[int(action.item())]
        obs, reward, terminated, truncated, info = env.step(action_env)
        done = terminated or truncated

        reward -= 0.01
        if done:
            reward -= 1.0

        frame_stack.append(preprocess_obs(obs))
        total_reward += reward

        if done:
            episode_idx += 1
            print(
                f"Episode {episode_idx} reward: {total_reward:.2f}  frames: {len(frames)}"
            )

            if total_reward > SAVE_THRESHOLD and len(frames) > 0:
                fname = (
                    OUT_DIR
                    / f"episode_{episode_idx}_reward_{int(total_reward)}_squashed.mp4"
                )
                print("Saving to:", fname)

                ok, reason = validate_frames(frames)
                if not ok:
                    print("Frame validation failed:", reason)
                    # Attempt to coerce: resize to first and dtype uint8
                    target_shape = frames[0].shape
                    coerced = []
                    for f in frames:
                        arr = f
                        if arr.dtype != np.uint8:
                            arr = np.clip(arr, 0, 255).astype(np.uint8)
                        if arr.shape != target_shape:
                            arr = cv2.resize(
                                arr,
                                (target_shape[1], target_shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        coerced.append(arr)
                    frames = coerced
                    ok, reason = validate_frames(frames)
                    print("Re-validation:", ok, reason)

                if ok:
                    # 1) Quantize each frame to PALETTE_SIZE solid colors (RGB)
                    quant_frames = []
                    for idx, f in enumerate(frames):
                        q = quantize_frame_kmeans(
                            f, k=PALETTE_SIZE, subsample=1500, rng_seed=idx + 1
                        )
                        quant_frames.append(q)

                    # 2) Upscale with nearest neighbor and apply vertical squash factor
                    upscaled = [
                        upscale_nearest(
                            f, scale=SCALE, squash_factor=SQUASH_HEIGHT_FACTOR
                        )
                        for f in quant_frames
                    ]

                    # 3) Duplicate frames to slow down playback
                    slowed = duplicate_frames(upscaled, factor=SLOW_FACTOR)

                    # Final validation
                    ok2, reason2 = validate_frames(slowed)
                    if not ok2:
                        print("Final validation failed:", reason2)
                    else:
                        # Write with imageio/ffmpeg if available, else fallback to OpenCV
                        try:
                            write_video_imageio(fname, slowed, fps=FPS)
                            print("Saved (imageio/ffmpeg):", fname)
                        except Exception as e:
                            print("imageio write failed:", e)
                            try:
                                # convert to BGR for OpenCV
                                slowed_bgr = [
                                    cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in slowed
                                ]
                                write_video_opencv(fname, slowed_bgr, fps=FPS)
                                print("Saved (OpenCV):", fname)
                            except Exception as e2:
                                print("OpenCV save failed:", e2)
                                print("Aborting save to avoid corrupt file.")

            # reset
            total_reward = 0.0
            frames = []
            obs, info = env.reset()
            obs, _, _, _, _ = env.step(1)
            obs, _, _, _, _ = env.step(1)
            frame_stack = deque([preprocess_obs(obs)] * 4, maxlen=4)

except KeyboardInterrupt:
    print("Interrupted by user, exiting.")
finally:
    try:
        env.close()
    except Exception:
        pass
