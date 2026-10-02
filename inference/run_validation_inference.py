import os
import csv
import gc
import argparse

import numpy as np
import torch
import torchvision

import echonet


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run LVEF inference on the converted validation-data videos."
)

parser.add_argument(
    "--original-data-dir",
    required=True,
    help="Path to the original-data directory required by the model code."
)

parser.add_argument(
    "--validation-video-dir",
    required=True,
    help="Directory containing the converted validation-data AVI videos."
)

parser.add_argument(
    "--validation-data-dir",
    required=True,
    help="Root directory containing the validation data and metadata."
)

parser.add_argument(
    "--results-dir",
    required=True,
    help="Root directory containing the trained model checkpoint."
)

parser.add_argument(
    "--output-dir",
    required=True,
    help="Directory in which validation predictions will be saved."
)

args = parser.parse_args()

ORIGINAL_DATA_DIR = args.original_data_dir
VALIDATION_VIDEO_DIR = args.validation_video_dir
VALIDATION_DATA_DIR = args.validation_data_dir
RESULTS_DIR = args.results_dir
OUTPUT_DIR = args.output_dir


# --------------------------------------------------
# PyTorch compatibility
# --------------------------------------------------

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


# --------------------------------------------------
# Paths
# --------------------------------------------------

ECHONET_ROOT = ORIGINAL_DATA_DIR

CAMUS_VIDEO_DIR = VALIDATION_VIDEO_DIR

CAMUS_METADATA = os.path.join(
    VALIDATION_DATA_DIR,
    "echodiffusion_inputs",
    "camus_echodiffusion_metadata.csv"
)

WEIGHTS = os.path.join(
    RESULTS_DIR,
    "ouyang_from_scratch",
    "best.pt"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "camus_ouyang_predictions.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# EchoNet normalisation values
# --------------------------------------------------

MEAN = np.array(
    [33.10984, 33.41836, 33.91615],
    dtype=np.float32
)

STD = np.array(
    [49.97386, 50.183193, 50.74082],
    dtype=np.float32
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Build the model
# --------------------------------------------------

model = torchvision.models.video.r2plus1d_18(
    pretrained=False
)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    1
)

model.fc.bias.data[0] = 55.6

if device.type == "cuda":
    model = torch.nn.DataParallel(model)

model.to(device)


# --------------------------------------------------
# Load trained checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    WEIGHTS
)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.eval()

print(
    "Loaded:",
    WEIGHTS
)


# --------------------------------------------------
# Load reference EF lookup
# --------------------------------------------------

reference_ef = {}

with open(
    CAMUS_METADATA,
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        reference_ef[
            row["patient"]
        ] = float(
            row["reference_ef"]
        )


# --------------------------------------------------
# Resume support
# --------------------------------------------------

completed = set()

if os.path.exists(OUTPUT_CSV):

    with open(
        OUTPUT_CSV,
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            completed.add(
                row["patient"]
            )


if not os.path.exists(OUTPUT_CSV):

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "reference_ef",
                "predicted_ef",
                "absolute_error",
                "num_clips"
            ]
        )

        writer.writeheader()


# --------------------------------------------------
# Process patients
# --------------------------------------------------

for patient_num in range(451, 501):

    patient = f"patient{patient_num:04d}"

    if patient in completed:

        print(
            f"{patient}: already complete, skipping"
        )

        continue


    video_path = os.path.join(
        CAMUS_VIDEO_DIR,
        f"{patient}_4CH.avi"
    )

    if not os.path.exists(video_path):

        print(
            f"{patient}: video missing"
        )

        continue


    if patient not in reference_ef:

        print(
            f"{patient}: reference EF missing"
        )

        continue


    print()
    print("========================================")
    print(f"Processing {patient}")
    print("========================================")


    # --------------------------------------------------
    # Create validation dataset
    # --------------------------------------------------

    dataset = echonet.datasets.Echo(

        root=ECHONET_ROOT,

        split="EXTERNAL_TEST",

        external_test_location=CAMUS_VIDEO_DIR,

        target_type="EF",

        mean=MEAN,

        std=STD,

        length=32,

        period=2,

        clips="all"
    )


    # --------------------------------------------------
    # Find this patient's index
    # --------------------------------------------------

    try:

        index = dataset.fnames.index(
            f"{patient}_4CH.avi"
        )

    except ValueError:

        print(
            f"{patient}: not found in dataset"
        )

        del dataset

        continue


    # --------------------------------------------------
    # Load all available clips
    #
    # Shape:
    # N x C x 32 x 112 x 112
    # --------------------------------------------------

    X, _ = dataset[index]

    if X.ndim == 4:

        X = np.expand_dims(
            X,
            axis=0
        )


    predictions = []


    # --------------------------------------------------
    # Run clips one at a time
    # --------------------------------------------------

    for clip_num, clip in enumerate(X):

        clip_tensor = torch.from_numpy(
            clip
        ).unsqueeze(0).float()

        clip_tensor = clip_tensor.to(
            device,
            non_blocking=False
        )

        with torch.no_grad():

            prediction = model(
                clip_tensor
            )

        prediction = float(
            prediction
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)[0]
        )

        predictions.append(
            prediction
        )

        del clip_tensor
        del prediction

        if device.type == "cuda":
            torch.cuda.empty_cache()


    # --------------------------------------------------
    # Average clip predictions
    # --------------------------------------------------

    predicted = float(
        np.mean(predictions)
    )

    reference = reference_ef[
        patient
    ]

    error = abs(
        predicted - reference
    )


    print(
        f"Reference EF: {reference:.3f}"
    )

    print(
        "Clip predictions: "
        + ", ".join(
            f"{x:.3f}"
            for x in predictions
        )
    )

    print(
        f"Predicted EF: {predicted:.3f}"
    )

    print(
        f"Absolute error: {error:.3f}"
    )


    # --------------------------------------------------
    # Save result immediately
    # --------------------------------------------------

    with open(
        OUTPUT_CSV,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "reference_ef",
                "predicted_ef",
                "absolute_error",
                "num_clips"
            ]
        )

        writer.writerow({
            "patient": patient,
            "reference_ef": reference,
            "predicted_ef": predicted,
            "absolute_error": error,
            "num_clips": len(
                predictions
            )
        })


    # --------------------------------------------------
    # Free memory
    # --------------------------------------------------

    del X
    del dataset

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


print()
print("========================================")
print("VALIDATION INFERENCE COMPLETE")
print("========================================")
print("Results:", OUTPUT_CSV)