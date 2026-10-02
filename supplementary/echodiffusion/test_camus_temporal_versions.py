import os
import cv2
import nibabel as nib
import numpy as np

# ============================================================
# SETTINGS — patient0451
# ============================================================

INPUT_NIFTI = r"C:\insigneo\data\CAMUS\database_nifti\patient0451\patient0451_4CH_half_sequence.nii.gz"

OUTPUT_DIR = r"C:\insigneo\data\CAMUS\temporal_test_patient0451"

ORIGINAL_FPS = 55.4
TARGET_FRAMES = 64
OUTPUT_SIZE = (112, 112)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD ORIGINAL CAMUS ED -> ES HALF-SEQUENCE
# ============================================================

img = nib.load(INPUT_NIFTI)
data = img.get_fdata().astype(np.float32)

print("Original NIfTI shape:", data.shape)

# Expected shape: H x W x T
if data.ndim != 3:
    raise ValueError(f"Expected 3D HxWxT data, got {data.shape}")

n_original = data.shape[2]

print("Original frames:", n_original)
print("CAMUS frame rate:", ORIGINAL_FPS)
print("Approx duration:", n_original / ORIGINAL_FPS, "seconds")


# ============================================================
# NORMALISE IMAGE INTENSITIES
# ============================================================

# Global normalisation across the whole cine.
low = np.percentile(data, 1)
high = np.percentile(data, 99)

data = np.clip(data, low, high)
data = (data - low) / (high - low + 1e-8)
data = (data * 255).astype(np.uint8)


# ============================================================
# IMAGE PREPROCESSING
# Resize each frame to 112 x 112
# ============================================================

def preprocess_frame(frame):
    frame = cv2.resize(
        frame,
        OUTPUT_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Convert grayscale -> RGB/BGR-style 3 channel image.
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame


original_frames = [
    preprocess_frame(data[:, :, i])
    for i in range(n_original)
]


# ============================================================
# TEMPORAL INTERPOLATION FUNCTION
# ============================================================

def interpolate_frames(frames, target_length):

    source_length = len(frames)

    old_positions = np.linspace(0, 1, source_length)
    new_positions = np.linspace(0, 1, target_length)

    result = []

    for p in new_positions:

        right = np.searchsorted(old_positions, p)

        if right == 0:
            frame = frames[0]

        elif right >= source_length:
            frame = frames[-1]

        else:
            left = right - 1

            denom = old_positions[right] - old_positions[left]

            if denom == 0:
                alpha = 0.0
            else:
                alpha = (
                    (p - old_positions[left]) / denom
                )

            frame = (
                (1.0 - alpha) * frames[left].astype(np.float32)
                + alpha * frames[right].astype(np.float32)
            )

            frame = np.clip(frame, 0, 255).astype(np.uint8)

        result.append(frame)

    return result


# ============================================================
# VIDEO WRITER
# ============================================================

def write_video(path, frames, fps):

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        path,
        fourcc,
        fps,
        OUTPUT_SIZE
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    for frame in frames:
        writer.write(frame)

    writer.release()

    print("\nCreated:", path)
    print("Frames:", len(frames))
    print("FPS:", fps)
    print("Playback duration:", len(frames) / fps, "seconds")


# ============================================================
# VERSION 1
# Current preprocessing
#
# 23 original frames
#       ->
# 64 interpolated frames
#       ->
# encoded at 30 fps
# ============================================================

frames_v1 = interpolate_frames(
    original_frames,
    TARGET_FRAMES
)

v1_path = os.path.join(
    OUTPUT_DIR,
    "V1_current_64frames_30fps.avi"
)

write_video(
    v1_path,
    frames_v1,
    30.0
)


# ============================================================
# VERSION 2
# Same 64 interpolated frames BUT preserve approximate
# original CAMUS duration.
#
# New FPS scales with increase in frame count.
# ============================================================

timing_preserved_fps = (
    ORIGINAL_FPS * TARGET_FRAMES / n_original
)

v2_path = os.path.join(
    OUTPUT_DIR,
    "V2_timing_preserved_64frames.avi"
)

write_video(
    v2_path,
    frames_v1,
    timing_preserved_fps
)


# ============================================================
# VERSION 3
# Experimental synthetic full cardiac cycle:
#
# ED -> ES -> ED
#
# Forward original sequence:
# 1, 2, ... 22, 23
#
# Then reverse WITHOUT duplicating ES:
# 22, 21, ... 2, 1
#
# This is NOT physiological reconstruction.
# It is only a temporal preprocessing experiment.
# ============================================================

synthetic_full_cycle = (
    original_frames
    + original_frames[-2::-1]
)

v3_path = os.path.join(
    OUTPUT_DIR,
    "V3_synthetic_ED_ES_ED_nativefps.avi"
)

write_video(
    v3_path,
    synthetic_full_cycle,
    ORIGINAL_FPS
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("TEMPORAL TEST COMPLETE")
print("============================================")

print("\nOriginal CAMUS:")
print(
    f"{n_original} frames @ {ORIGINAL_FPS:.2f} fps "
    f"= {n_original / ORIGINAL_FPS:.3f} s"
)

print("\nV1 — CURRENT:")
print(
    f"{len(frames_v1)} frames @ 30 fps "
    f"= {len(frames_v1) / 30:.3f} s"
)

print("\nV2 — TIMING PRESERVED:")
print(
    f"{len(frames_v1)} frames @ "
    f"{timing_preserved_fps:.2f} fps "
    f"= {len(frames_v1) / timing_preserved_fps:.3f} s"
)

print("\nV3 — SYNTHETIC FULL CYCLE:")
print(
    f"{len(synthetic_full_cycle)} frames @ "
    f"{ORIGINAL_FPS:.2f} fps "
    f"= {len(synthetic_full_cycle) / ORIGINAL_FPS:.3f} s"
)

print("\nOutput folder:")
print(OUTPUT_DIR)

print(
    "\nIMPORTANT: V3 is an experimental synthetic ED->ES->ED "
    "sequence, NOT a genuine measured full cardiac cycle."
)