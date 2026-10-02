import os
import csv
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

# ------------------------------------------------------------
# Default paths
# ------------------------------------------------------------

DEFAULT_MODEL = r"C:\insigneo\EchoDiffusion\1SCM_windows_test"

DEFAULT_METADATA = (
    r"C:\insigneo\data\CAMUS\echodiffusion_inputs"
    r"\camus_echodiffusion_metadata.csv"
)

DEFAULT_IMAGE_DIR = (
    r"C:\insigneo\data\CAMUS\echodiffusion_inputs\images"
)


# ------------------------------------------------------------
# Load EchoDiffusion once
# ------------------------------------------------------------

def load_model(model_dir, device):

    config = OmegaConf.load(
        os.path.join(model_dir, "config.yaml")
    )

    config.dataset.num_frames = int(
        config.dataset.fps *
        config.dataset.duration
    )

    unets = []

    for i, (_, v) in enumerate(
        config.unets.items()
    ):
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
        **OmegaConf.to_container(
            config.imagen
        )
    )

    trainer = ImagenTrainer(
        imagen=imagen,
        **config.trainer
    ).to(device)

    weights = os.path.join(
        model_dir,
        "merged.pt"
    )

    trainer.load(weights)
    trainer.eval()

    return config, trainer


# ------------------------------------------------------------
# Read CAMUS metadata
# ------------------------------------------------------------

def read_metadata(path):

    rows = []

    with open(path, newline="") as f:

        reader = csv.DictReader(f)

        required = {
            "patient",
            "image",
            "reference_ef"
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                f"Missing metadata columns: {sorted(missing)}"
            )

        for row in reader:
            rows.append(row)

    return rows


# ------------------------------------------------------------
# Read Ouyang predictions
#
# Expected CSV:
#
# patient,predicted_ef
# patient0451,53.397
# patient0452,XX.XXX
# ...
# ------------------------------------------------------------

def read_predictions(path):

    predictions = {}

    with open(path, newline="") as f:

        reader = csv.DictReader(f)

        required = {
            "patient",
            "predicted_ef"
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                f"Missing prediction columns: {sorted(missing)}"
            )

        for row in reader:

            patient = row["patient"].strip()

            if patient in predictions:
                raise RuntimeError(
                    f"Duplicate prediction for {patient}"
                )

            predictions[patient] = float(
                row["predicted_ef"]
            )

    return predictions


# ------------------------------------------------------------
# Fixed random seed
#
# Same seed is reused for the reference and predicted-EF
# runs for a given patient.
# ------------------------------------------------------------

def set_seed(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Generate one video
# ------------------------------------------------------------

def generate_video(
    trainer,
    config,
    device,
    image_path,
    ef,
    seed,
    output_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    if image.size != (112, 112):
        image = image.resize(
            (112, 112),
            Image.Resampling.BILINEAR
        )

    image_array = (
        np.asarray(
            image,
            dtype=np.float32
        ) / 255.0
    )

    cond_image = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    # EF must be 0–1 for EchoDiffusion
    ef_normalised = ef / 100.0

    embedding = torch.tensor(
        [[[ef_normalised]]],
        dtype=torch.float32,
        device=device
    )

    # Reset RNG immediately before generation
    set_seed(seed)

    with torch.no_grad():

        gen_video = trainer.sample(
            batch_size=1,
            video_frames=config.dataset.num_frames,
            text_embeds=embedding,
            cond_scale=5.0,
            cond_images=cond_image,
            use_tqdm=True
        ).detach().cpu()

    # B x C x T x H x W
    video = (
        gen_video[0]
        .clamp(0, 1)
        .multiply(255)
        .byte()
        .permute(1, 2, 3, 0)
    )

    frames = [
        Image.fromarray(
            frame.numpy()
        )
        for frame in video
    ]

    fps = config.dataset.fps

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 / fps,
        loop=0
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["reference", "predicted"],
        required=True
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL
    )

    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA
    )

    parser.add_argument(
        "--image_dir",
        default=DEFAULT_IMAGE_DIR
    )

    parser.add_argument(
        "--predictions_csv",
        default=None
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    rows = read_metadata(
        args.metadata
    )

    if len(rows) != 50:

        raise RuntimeError(
            f"Expected 50 CAMUS patients, found {len(rows)}"
        )

    # --------------------------------------------------------
    # Ouyang predictions
    # --------------------------------------------------------

    predictions = None

    if args.mode == "predicted":

        if args.predictions_csv is None:

            raise RuntimeError(
                "--predictions_csv is required "
                "when mode=predicted"
            )

        predictions = read_predictions(
            args.predictions_csv
        )

        missing = [
            row["patient"]
            for row in rows
            if row["patient"] not in predictions
        ]

        if missing:

            raise RuntimeError(
                "Missing Ouyang predictions for: "
                + ", ".join(missing)
            )

    # --------------------------------------------------------
    # Load model once
    # --------------------------------------------------------

    print()
    print("Loading EchoDiffusion...")
    print()

    config, trainer = load_model(
        args.model,
        device
    )

    print(
        "Model loaded."
    )

    print(
        "Generated frames:",
        config.dataset.num_frames
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    manifest = []

    # --------------------------------------------------------
    # Generate 50 patients
    # --------------------------------------------------------

    for i, row in enumerate(rows):

        patient = row["patient"]

        image_path = os.path.join(
            args.image_dir,
            row["image"]
        )

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        reference_ef = float(
            row["reference_ef"]
        )

        if args.mode == "reference":

            conditioning_ef = reference_ef

        else:

            conditioning_ef = predictions[
                patient
            ]

        output_path = os.path.join(
            args.output_dir,
            f"{patient}_EF{conditioning_ef:.3f}_generated.gif"
        )

        print()
        print(
            f"[{i+1}/50] {patient}"
        )
        print(
            f"Reference EF: {reference_ef:.3f}"
        )
        print(
            f"Conditioning EF: {conditioning_ef:.3f}"
        )
        print(
            f"Seed: {args.seed}"
        )

        generate_video(
            trainer=trainer,
            config=config,
            device=device,
            image_path=image_path,
            ef=conditioning_ef,
            seed=args.seed,
            output_path=output_path
        )

        manifest.append({
            "patient": patient,
            "reference_ef": reference_ef,
            "conditioning_ef": conditioning_ef,
            "mode": args.mode,
            "seed": args.seed,
            "output": output_path
        })

        print(
            "Saved:",
            output_path
        )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest_path = os.path.join(
        args.output_dir,
        "run_manifest.csv"
    )

    with open(
        manifest_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "reference_ef",
                "conditioning_ef",
                "mode",
                "seed",
                "output"
            ]
        )

        writer.writeheader()
        writer.writerows(manifest)

    print()
    print("==========================================")
    print("BATCH COMPLETE")
    print("Patients generated:", len(manifest))
    print("Manifest:", manifest_path)
    print("==========================================")


if __name__ == "__main__":
    main()