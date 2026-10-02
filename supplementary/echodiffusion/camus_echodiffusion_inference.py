import os
import argparse

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from imagen_pytorch import (
    Unet3D,
    ElucidatedImagen,
    ImagenTrainer,
    Imagen,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--patient",
    type=str,
    required=True,
)

parser.add_argument(
    "--ef",
    type=float,
    required=True,
    help="EF in percent, e.g. 20.0"
)

parser.add_argument(
    "--model",
    type=str,
    default=r"C:\insigneo\EchoDiffusion\1SCM_windows_test",
)

parser.add_argument(
    "--cond_scale",
    type=float,
    default=5.0,
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

args = parser.parse_args()


# --------------------------------------------------
# Paths
# --------------------------------------------------

IMAGE_PATH = os.path.join(
    r"C:\insigneo\data\CAMUS\echodiffusion_inputs\images",
    f"{args.patient}_4CH_ED.png"
)

OUTPUT_DIR = os.path.join(
    args.model,
    "camus_samples"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Check input
# --------------------------------------------------

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"CAMUS image not found: {IMAGE_PATH}"
    )

print("CAMUS image:", IMAGE_PATH)
print("Conditioning EF:", args.ef)


# --------------------------------------------------
# Load config
# --------------------------------------------------

config = OmegaConf.load(
    os.path.join(args.model, "config.yaml")
)

config.dataset.num_frames = int(
    config.dataset.fps *
    config.dataset.duration
)

print(
    "Target frames:",
    config.dataset.num_frames
)

print(
    "Target resolution:",
    max(config.imagen.image_sizes)
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Build model
# --------------------------------------------------

unets = []

for i, (k, v) in enumerate(config.unets.items()):

    unets.append(
        Unet3D(
            **v,
            lowres_cond=(i > 0)
        )
    )


imagen_class = (
    ElucidatedImagen
    if config.imagen.elucidated
    else Imagen
)

del config.imagen.elucidated


imagen = imagen_class(
    unets=unets,
    **OmegaConf.to_container(config.imagen)
)


trainer = ImagenTrainer(
    imagen=imagen,
    **config.trainer
).to(device)


# --------------------------------------------------
# Load pretrained weights
# --------------------------------------------------

weights_path = os.path.join(
    args.model,
    "merged.pt"
)

if not os.path.exists(weights_path):
    raise FileNotFoundError(
        f"Model weights not found: {weights_path}"
    )

trainer.load(weights_path)

print("Loaded model:", weights_path)

trainer.eval()


# --------------------------------------------------
# Load CAMUS conditioning image
#
# EchoDiffusion's own dataset converts images
# to float tensors in [0,1].
# --------------------------------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

if image.size != (112, 112):
    image = image.resize(
        (112, 112),
        Image.Resampling.BILINEAR
    )

image_array = np.asarray(
    image,
    dtype=np.float32
) / 255.0

cond_image = torch.from_numpy(
    image_array
).permute(2, 0, 1).unsqueeze(0)

cond_image = cond_image.to(device)


# --------------------------------------------------
# EF embedding
#
# EchoDiffusion expects EF normalised to 0-1.
# --------------------------------------------------

ef_normalised = args.ef / 100.0

embedding = torch.tensor(
    [[[ef_normalised]]],
    dtype=torch.float32,
    device=device
)


print(
    "EF embedding:",
    ef_normalised
)

print(
    "Condition image shape:",
    tuple(cond_image.shape)
)


# --------------------------------------------------
# Generate
# --------------------------------------------------

print()
print("Starting EchoDiffusion sampling...")
print("Random seed:", args.seed)
print()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

with torch.no_grad():

    gen_video = trainer.sample(
        batch_size=1,
        video_frames=config.dataset.num_frames,
        text_embeds=embedding,
        cond_scale=args.cond_scale,
        cond_images=cond_image,
        use_tqdm=True,
    ).detach().cpu()


# --------------------------------------------------
# Save generated video as GIF
# --------------------------------------------------

# Expected:
# B x C x T x H x W

print()
print(
    "Generated tensor shape:",
    tuple(gen_video.shape)
)

video = gen_video[0]

# C x T x H x W
video = (
    video
    .clamp(0, 1)
    .multiply(255)
    .byte()
    .permute(1, 2, 3, 0)
)

frames = [
    Image.fromarray(frame.numpy())
    for frame in video
]


output_path = os.path.join(
    OUTPUT_DIR,
    f"{args.patient}_EF{int(args.ef)}_generated.gif"
)

fps = config.dataset.fps

frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=1000 / fps,
    loop=0
)


print()
print("========================================")
print("ECHODIFFUSION COMPLETE")
print("========================================")
print("Patient:", args.patient)
print("Conditioning EF:", args.ef)
print("Generated frames:", len(frames))
print("Output:", output_path)
print("========================================")