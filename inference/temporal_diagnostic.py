import os
import cv2
import numpy as np
import torch
import torchvision
import echonet

# ============================================================
# PATHS
# ============================================================

import argparse

parser = argparse.ArgumentParser(
    description="Run the temporal diagnostic experiment on patient 0451."
)

parser.add_argument(
    "--video-dir",
    required=True,
    help="Directory containing the temporal diagnostic AVI videos."
)

parser.add_argument(
    "--checkpoint",
    required=True,
    help="Path to the trained EF model checkpoint."
)

args = parser.parse_args()

VIDEO_DIR = args.video_dir
CHECKPOINT = args.checkpoint

VIDEOS = [
    "V1_current_64frames_30fps.avi",
    "V2_timing_preserved_64frames.avi",
    "V3_synthetic_ED_ES_ED_nativefps.avi",
]

# Same normalization used in our working CAMUS inference
MEAN = np.array(
    [33.10984, 33.41836, 33.91615],
    dtype=np.float32
)

STD = np.array(
    [49.97386, 50.183193, 50.74082],
    dtype=np.float32
)

LENGTH = 32
PERIOD = 2


# ============================================================
# MODEL
# ============================================================

print("Loading Ouyang R(2+1)D-18...")

model = torchvision.models.video.r2plus1d_18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    1
)

# Same initialization convention as EchoNet
model.fc.bias.data[0] = 55.6

# Our checkpoint was trained with DataParallel
model = torch.nn.DataParallel(model)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cuda"
)

model.load_state_dict(checkpoint["state_dict"])

model = model.cuda()
model.eval()

print("Model loaded.")


# ============================================================
# LOAD AVI
# ============================================================

def load_video(path):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV gives BGR.
        # Convert to RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {path}")

    frames = np.stack(frames)

    print(
        f"\n{os.path.basename(path)}:"
        f"\n  decoded frames = {len(frames)}"
        f"\n  AVI FPS        = {fps:.3f}"
    )

    return frames


# ============================================================
# ECHONET TEMPORAL SAMPLING
#
# length=32
# period=2
#
# For videos >=63 frames:
# 0,2,4,...,62
#
# For shorter videos such as V3, we create all valid clips
# after padding the final frame, then average predictions.
# ============================================================

def make_clips(frames):

    n = len(frames)

    required_span = 1 + (LENGTH - 1) * PERIOD
    # = 63 frames from first sampled index to last

    if n < required_span:

        pad_count = required_span - n

        print(
            f"  Video shorter than required {required_span} "
            f"frames; padding final frame by {pad_count} frames."
        )

        padding = np.repeat(
            frames[-1][None, ...],
            pad_count,
            axis=0
        )

        frames = np.concatenate(
            [frames, padding],
            axis=0
        )

        n = len(frames)

    max_start = n - required_span

    clips = []

    for start in range(max_start + 1):

        indices = (
            start
            + np.arange(LENGTH) * PERIOD
        )

        clip = frames[indices]

        # T,H,W,C -> C,T,H,W
        clip = clip.transpose(3, 0, 1, 2)

        clip = clip.astype(np.float32)

        # EchoNet normalization, channel-wise
        clip = (
            clip
            - MEAN[:, None, None, None]
        ) / STD[:, None, None, None]

        clips.append(clip)

    return clips


# ============================================================
# INFERENCE
# ============================================================

def predict(path):

    frames = load_video(path)

    clips = make_clips(frames)

    print(f"  Number of clips = {len(clips)}")

    predictions = []

    with torch.no_grad():

        for clip in clips:

            x = torch.from_numpy(
                clip[None, ...]
            ).float().cuda()

            output = model(x)

            prediction = output.detach().cpu().numpy().reshape(-1)[0]

            predictions.append(float(prediction))

    mean_prediction = float(np.mean(predictions))

    return mean_prediction, predictions


# ============================================================
# RUN ALL THREE
# ============================================================

results = {}

for video in VIDEOS:

    path = os.path.join(
        VIDEO_DIR,
        video
    )

    ef, clip_predictions = predict(path)

    results[video] = ef

    print(f"  Predicted EF = {ef:.3f}%")

    if len(clip_predictions) > 1:
        print(
            "  Clip predictions:",
            [round(x, 3) for x in clip_predictions]
        )


# ============================================================
# RESULTS
# ============================================================

REFERENCE_EF = 20.0

print("\n")
print("=" * 65)
print("PATIENT0451 TEMPORAL PREPROCESSING TEST")
print("=" * 65)

print(f"CAMUS reference EF: {REFERENCE_EF:.1f}%\n")

for video, prediction in results.items():

    error = prediction - REFERENCE_EF

    print(video)
    print(f"  Predicted EF : {prediction:.3f}%")
    print(f"  Error        : {error:+.3f} percentage points")
    print()

print("=" * 65)

print(
    "\nKEY COMPARISON:"
    "\nIf V1 and V2 predictions are identical/nearly identical,"
    "\nAVI FPS metadata is not affecting the model prediction."
)

print(
    "\nV3 is only an experimental synthetic ED->ES->ED sequence."
    "\nIt must NOT be interpreted as a genuine measured cardiac cycle."
)