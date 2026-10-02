import os
import csv
import nibabel as nib
import numpy as np
import cv2

CAMUS_ROOT = r"C:\insigneo\data\CAMUS\database_nifti"

OUTPUT_ROOT = r"C:\insigneo\data\CAMUS\echodiffusion_inputs"
IMAGE_DIR = os.path.join(OUTPUT_ROOT, "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

CSV_PATH = os.path.join(
    OUTPUT_ROOT,
    "camus_echodiffusion_metadata.csv"
)

records = []

for patient_num in range(451, 501):

    patient = f"patient{patient_num:04d}"
    patient_dir = os.path.join(CAMUS_ROOT, patient)

    nii_path = os.path.join(
        patient_dir,
        f"{patient}_4CH_half_sequence.nii.gz"
    )

    cfg_path = os.path.join(
        patient_dir,
        "Info_4CH.cfg"
    )

    if not os.path.exists(nii_path):
        print(f"SKIP {patient}: NIfTI missing")
        continue

    if not os.path.exists(cfg_path):
        print(f"SKIP {patient}: config missing")
        continue

    # ----------------------------
    # Read EF from CAMUS metadata
    # ----------------------------
    reference_ef = None

    with open(cfg_path, "r") as f:
        for line in f:

            if line.startswith("EF:"):
                reference_ef = float(
                    line.split(":")[1].strip()
                )

    if reference_ef is None:
        print(f"SKIP {patient}: EF missing")
        continue

    # ----------------------------
    # Load cine
    # Shape = H x W x T
    # ----------------------------
    nii = nib.load(nii_path)
    video = np.asarray(nii.dataobj).astype(np.float32)

    H, W, T = video.shape

    # ----------------------------
    # Global intensity normalisation
    # ----------------------------
    vmin = video.min()
    vmax = video.max()

    if vmax <= vmin:
        print(f"SKIP {patient}: invalid intensity range")
        continue

    video = (
        (video - vmin)
        / (vmax - vmin)
        * 255.0
    )

    video = np.clip(video, 0, 255)

    # ----------------------------
    # One consistent crop across
    # the entire cine
    # ----------------------------
    mask = np.any(video > 8, axis=2)

    ys, xs = np.where(mask)

    if len(xs) == 0:
        print(f"SKIP {patient}: no ultrasound region found")
        continue

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    margin = 5

    x1 = max(0, x1 - margin)
    x2 = min(W - 1, x2 + margin)
    y1 = max(0, y1 - margin)
    y2 = min(H - 1, y2 + margin)

    video = video[
        y1:y2 + 1,
        x1:x2 + 1,
        :
    ]

    # ----------------------------
    # Use ED frame.
    #
    # CAMUS metadata gives ED as a
    # 1-based frame index.
    # ----------------------------
    ed_frame_index = 1

    frame = video[:, :, ed_frame_index - 1]

    frame = frame.astype(np.uint8)

    # ----------------------------
    # Resize to EchoDiffusion
    # conditioning image size
    # ----------------------------
    frame = cv2.resize(
        frame,
        (112, 112),
        interpolation=cv2.INTER_AREA
    )

    # ----------------------------
    # Grayscale -> 3 channel RGB
    # ----------------------------
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_GRAY2RGB
    )

    # ----------------------------
    # Save
    # ----------------------------
    image_name = f"{patient}_4CH_ED.png"

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    cv2.imwrite(
        image_path,
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    )

    records.append({
        "patient": patient,
        "image": image_name,
        "reference_ef": reference_ef,
        "source": os.path.basename(nii_path),
        "source_frames": T,
        "conditioning_frame": "ED"
    })

    print(
        f"{patient}: "
        f"EF={reference_ef:.1f}% "
        f"frames={T}"
    )

# ----------------------------
# Save metadata
# ----------------------------
with open(
    CSV_PATH,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "patient",
            "image",
            "reference_ef",
            "source",
            "source_frames",
            "conditioning_frame"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

print()
print("================================")
print("CAMUS PREPARATION COMPLETE")
print("================================")
print("Patients processed:", len(records))
print("Images:", IMAGE_DIR)
print("Metadata:", CSV_PATH)