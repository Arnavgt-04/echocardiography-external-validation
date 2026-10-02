
import os
import re
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import nibabel as nib


# ============================================================
# CAMUS / EchoDiffusion paired analysis
#
# Main comparison:
#   Reference EF -> EchoDiffusion
#   Ouyang EF    -> EchoDiffusion
#
# Optional generated-EF CSVs can be supplied to add:
#   generated EF fidelity
#   paired Wilcoxon test
#   upstream-error vs downstream-error analysis
#
# Visual metrics are calculated directly against the
# corresponding CAMUS 4CH half-sequence after temporal
# resampling to 16 frames.
# ============================================================


CAMUS_ROOT = Path(
    r"C:\insigneo\data\CAMUS\database_nifti"
)

METADATA_CSV = Path(
    r"C:\insigneo\data\CAMUS\echodiffusion_inputs"
    r"\camus_echodiffusion_metadata.csv"
)

OUYANG_CSV = Path(
    r"C:\insigneo\results\camus_ouyang"
    r"\camus_ouyang_predictions.csv"
)

REFERENCE_BATCH = Path(
    r"C:\insigneo\EchoDiffusion\1SCM_windows_test"
    r"\camus_reference_batch"
)

PREDICTED_BATCH = Path(
    r"C:\insigneo\EchoDiffusion\1SCM_windows_test"
    r"\camus_ouyang_batch"
)

OUTPUT_ROOT = Path(
    r"C:\insigneo\results\camus_evaluation"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
STATS_DIR = OUTPUT_ROOT / "statistics"
DATA_DIR = OUTPUT_ROOT / "data"


# ------------------------------------------------------------
# Optional: generated-EF score CSVs
#
# Set these once you have scored the generated GIFs with the
# EchoDiffusion reference EF regressor.
#
# The script accepts patient + one generated-EF column.
# It recognises columns such as:
#   patient / Patient / patient_id
#   Gen EF / Generated EF / EstimatedEF / generated_ef
# ------------------------------------------------------------

REFERENCE_GEN_EF_CSV = None
PREDICTED_GEN_EF_CSV = None


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

SEED = 42
N_GENERATED_FRAMES = 16


# ============================================================
# Helpers
# ============================================================

def ensure_dirs():
    for folder in [
        OUTPUT_ROOT,
        TABLE_DIR,
        FIGURE_DIR,
        STATS_DIR,
        DATA_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def patient_key(value):
    match = re.search(r"patient(\d{4})", str(value), re.I)
    if match:
        return f"patient{match.group(1)}"
    return None


def load_camus_metadata():
    df = pd.read_csv(METADATA_CSV)

    required = {"patient", "reference_ef"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"CAMUS metadata missing columns: {sorted(missing)}"
        )

    df["patient"] = df["patient"].astype(str)
    df["reference_ef"] = pd.to_numeric(
        df["reference_ef"],
        errors="raise"
    )

    return df


def load_ouyang():
    df = pd.read_csv(OUYANG_CSV)

    required = {
        "patient",
        "predicted_ef"
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Ouyang CSV missing columns: {sorted(missing)}"
        )

    df = df[[
        "patient",
        "predicted_ef"
    ]].copy()

    df["patient"] = df["patient"].astype(str)

    df["predicted_ef"] = pd.to_numeric(
        df["predicted_ef"],
        errors="raise"
    )

    if df["patient"].duplicated().any():
        raise RuntimeError(
            "Duplicate patients found in Ouyang CSV."
        )

    return df


def load_generated_ef_csv(path):
    path = Path(path)

    df = pd.read_csv(path)

    # Find patient column or infer patient from any text column.
    patient_col = None

    for candidate in [
        "patient",
        "Patient",
        "patient_id",
        "PatientID"
    ]:
        if candidate in df.columns:
            patient_col = candidate
            break

    if patient_col is not None:
        patients = df[patient_col].map(patient_key)
    else:
        patients = pd.Series(
            [None] * len(df),
            index=df.index
        )

        for col in df.columns:
            if df[col].dtype == object:
                extracted = df[col].map(patient_key)
                patients = patients.fillna(extracted)

    # Find generated EF column.
    ef_col = None

    candidates = [
        "Gen EF",
        "Generated EF",
        "EstimatedEF",
        "generated_ef",
        "gen_ef",
        "GenEF"
    ]

    normalised = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalised:
            ef_col = normalised[key]
            break

    if ef_col is None:
        raise RuntimeError(
            f"Could not find generated-EF column in {path}. "
            f"Columns found: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "patient": patients,
        "generated_ef": pd.to_numeric(
            df[ef_col],
            errors="raise"
        )
    })

    out = out.dropna(subset=["patient"])

    if out.empty:
        raise RuntimeError(
            f"Could not identify patient IDs in {path}."
        )

    out["patient"] = out["patient"].astype(str)

    # If multiple rows per patient exist, average them.
    out = (
        out.groupby("patient", as_index=False)
        ["generated_ef"]
        .mean()
    )

    return out


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred)
    }


def paired_wilcoxon(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    diff = a - b

    if np.allclose(diff, 0):
        return {
            "n": len(a),
            "statistic": 0.0,
            "p_value": 1.0
        }

    stat, p = wilcoxon(
        a,
        b,
        zero_method="wilcox",
        alternative="two-sided"
    )

    return {
        "n": len(a),
        "statistic": float(stat),
        "p_value": float(p)
    }


def save_figure(fig, name):
    path = FIGURE_DIR / name
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# Visual processing
# ============================================================

def load_camus_sequence(patient):
    nii_path = (
        CAMUS_ROOT
        / patient
        / f"{patient}_4CH_half_sequence.nii.gz"
    )

    if not nii_path.exists():
        raise FileNotFoundError(
            f"Missing CAMUS sequence: {nii_path}"
        )

    video = np.asarray(
        nib.load(str(nii_path)).dataobj,
        dtype=np.float32
    )

    # H x W x T

    vmin = video.min()
    vmax = video.max()

    if vmax <= vmin:
        raise RuntimeError(
            f"{patient}: invalid intensity range"
        )

    video = (
        (video - vmin)
        / (vmax - vmin)
        * 255.0
    )

    video = np.clip(video, 0, 255)

    # Same consistent crop approach used for
    # the project-specific CAMUS conversion.
    mask = np.any(
        video > 8,
        axis=2
    )

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError(
            f"{patient}: no ultrasound region found"
        )

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    margin = 5

    x1 = max(0, x1 - margin)
    x2 = min(video.shape[1] - 1, x2 + margin)
    y1 = max(0, y1 - margin)
    y2 = min(video.shape[0] - 1, y2 + margin)

    video = video[
        y1:y2 + 1,
        x1:x2 + 1,
        :
    ]

    # Temporal interpolation to match generated 16 frames.
    old_t = np.linspace(
        0,
        1,
        video.shape[2]
    )

    new_t = np.linspace(
        0,
        1,
        N_GENERATED_FRAMES
    )

    resampled = np.empty(
        (
            video.shape[0],
            video.shape[1],
            N_GENERATED_FRAMES
        ),
        dtype=np.float32
    )

    for i in range(video.shape[0]):
        for j in range(video.shape[1]):
            resampled[i, j, :] = np.interp(
                new_t,
                old_t,
                video[i, j, :]
            )

    frames = []

    for t in range(N_GENERATED_FRAMES):

        frame = resampled[:, :, t].astype(
            np.uint8
        )

        frame = np.asarray(
            Image.fromarray(frame).resize(
                (112, 112),
                Image.Resampling.BILINEAR
            )
        )

        frames.append(frame)

    return np.stack(frames, axis=0)


def load_gif_frames(path):
    frames = []

    with Image.open(path) as gif:

        for i in range(gif.n_frames):

            gif.seek(i)

            frame = gif.convert("L").resize(
                (112, 112),
                Image.Resampling.BILINEAR
            )

            frames.append(
                np.asarray(
                    frame,
                    dtype=np.uint8
                )
            )

    if len(frames) == 0:
        raise RuntimeError(
            f"No frames found in {path}"
        )

    frames = np.stack(frames, axis=0)

    # Match 16 frames if necessary.
    if len(frames) != N_GENERATED_FRAMES:

        old_t = np.linspace(
            0,
            1,
            len(frames)
        )

        new_t = np.linspace(
            0,
            1,
            N_GENERATED_FRAMES
        )

        resized = []

        for t in new_t:

            lo = int(
                np.floor(
                    t * (len(frames) - 1)
                )
            )

            hi = int(
                np.ceil(
                    t * (len(frames) - 1)
                )
            )

            if lo == hi:
                resized.append(
                    frames[lo]
                )
            else:
                alpha = (
                    t * (len(frames) - 1)
                    - lo
                )

                interp = (
                    frames[lo].astype(float)
                    * (1 - alpha)
                    +
                    frames[hi].astype(float)
                    * alpha
                )

                resized.append(
                    np.clip(
                        interp,
                        0,
                        255
                    ).astype(np.uint8)
                )

        frames = np.stack(
            resized,
            axis=0
        )

    return frames


def find_gif(directory, patient):
    candidates = sorted(
        directory.glob(
            f"{patient}_EF*_generated.gif"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"No generated GIF found for {patient} "
            f"in {directory}"
        )

    # There should be one file per patient per batch.
    return candidates[0]


def compute_visual_metrics(reference_frames, generated_frames):
    ssim_values = []

    for ref, gen in zip(
        reference_frames,
        generated_frames
    ):
        value = ssim(
            ref,
            gen,
            data_range=255
        )
        ssim_values.append(value)

    return {
        "SSIM_mean": float(
            np.mean(ssim_values)
        ),
        "SSIM_median": float(
            np.median(ssim_values)
        )
    }


# ============================================================
# Main
# ============================================================

def main():

    ensure_dirs()

    print("Loading metadata...")
    metadata = load_camus_metadata()

    print("Loading Ouyang predictions...")
    ouyang = load_ouyang()

    df = metadata.merge(
        ouyang,
        on="patient",
        how="inner",
        validate="one_to_one"
    )

    if len(df) != 50:
        raise RuntimeError(
            f"Expected 50 matched CAMUS patients; "
            f"found {len(df)}."
        )

    # --------------------------------------------------------
    # Upstream Ouyang performance
    # --------------------------------------------------------

    upstream = calculate_metrics(
        df["reference_ef"],
        df["predicted_ef"]
    )

    upstream_table = pd.DataFrame([upstream])

    upstream_table.to_csv(
        TABLE_DIR / "01_upstream_ouyang_metrics.csv",
        index=False
    )

    df["ouyang_abs_error"] = (
        df["predicted_ef"]
        - df["reference_ef"]
    ).abs()

    # --------------------------------------------------------
    # Figures: upstream scatter
    # --------------------------------------------------------

    fig = plt.figure(figsize=(7, 7))

    plt.scatter(
        df["reference_ef"],
        df["predicted_ef"]
    )

    lo = min(
        df["reference_ef"].min(),
        df["predicted_ef"].min()
    )

    hi = max(
        df["reference_ef"].max(),
        df["predicted_ef"].max()
    )

    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--"
    )

    plt.xlabel("CAMUS reference EF (%)")
    plt.ylabel("Ouyang predicted EF (%)")

    plt.title(
        "Ouyang EF prediction on CAMUS"
    )

    plt.text(
        0.05,
        0.95,
        f"MAE = {upstream['MAE']:.2f}%\n"
        f"RMSE = {upstream['RMSE']:.2f}%\n"
        f"R² = {upstream['R2']:.2f}",
        transform=plt.gca().transAxes,
        va="top"
    )

    save_figure(
        fig,
        "01_Ouyang_EF_reference_vs_predicted.png"
    )

    # --------------------------------------------------------
    # Bland-Altman
    # --------------------------------------------------------

    means = (
        df["reference_ef"]
        + df["predicted_ef"]
    ) / 2

    differences = (
        df["predicted_ef"]
        - df["reference_ef"]
    )

    bias = differences.mean()
    sd = differences.std(
        ddof=1
    )

    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig = plt.figure(figsize=(8, 6))

    plt.scatter(
        means,
        differences
    )

    plt.axhline(
        bias,
        linestyle="--"
    )

    plt.axhline(
        loa_upper,
        linestyle=":"
    )

    plt.axhline(
        loa_lower,
        linestyle=":"
    )

    plt.xlabel(
        "Mean of reference and predicted EF (%)"
    )

    plt.ylabel(
        "Predicted − reference EF (%)"
    )

    plt.title(
        "Bland–Altman: Ouyang vs CAMUS reference EF"
    )

    plt.text(
        0.03,
        0.95,
        f"Bias = {bias:.2f}%\n"
        f"95% limits = {loa_lower:.2f} to {loa_upper:.2f}%",
        transform=plt.gca().transAxes,
        va="top"
    )

    save_figure(
        fig,
        "02_Ouyang_Bland_Altman.png"
    )

    # --------------------------------------------------------
    # Visual evaluation
    # --------------------------------------------------------

    print()
    print("Computing visual metrics...")
    print()

    visual_rows = []

    for i, row in df.iterrows():

        patient = row["patient"]

        reference_gif = find_gif(
            REFERENCE_BATCH,
            patient
        )

        predicted_gif = find_gif(
            PREDICTED_BATCH,
            patient
        )

        camus_frames = load_camus_sequence(
            patient
        )

        ref_gen_frames = load_gif_frames(
            reference_gif
        )

        pred_gen_frames = load_gif_frames(
            predicted_gif
        )

        ref_metrics = compute_visual_metrics(
            camus_frames,
            ref_gen_frames
        )

        pred_metrics = compute_visual_metrics(
            camus_frames,
            pred_gen_frames
        )

        visual_rows.append({
            "patient": patient,
            "SSIM_reference_EF": ref_metrics[
                "SSIM_mean"
            ],
            "SSIM_Ouyang_EF": pred_metrics[
                "SSIM_mean"
            ]
        })

        print(
            f"{patient}: "
            f"reference SSIM={ref_metrics['SSIM_mean']:.4f}, "
            f"Ouyang SSIM={pred_metrics['SSIM_mean']:.4f}"
        )

    visual = pd.DataFrame(
        visual_rows
    )

    df = df.merge(
        visual,
        on="patient",
        how="left"
    )

    visual_long = pd.DataFrame({
        "condition": (
            ["Reference EF"] * len(df)
            +
            ["Ouyang EF"] * len(df)
        ),
        "SSIM": (
            df["SSIM_reference_EF"].tolist()
            +
            df["SSIM_Ouyang_EF"].tolist()
        )
    })

    visual.to_csv(
        TABLE_DIR / "02_visual_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Paired SSIM test
    # --------------------------------------------------------

    ssim_test = paired_wilcoxon(
        df["SSIM_reference_EF"],
        df["SSIM_Ouyang_EF"]
    )

    pd.DataFrame([{
        "test": "Wilcoxon signed-rank",
        "metric": "SSIM",
        **ssim_test
    }]).to_csv(
        STATS_DIR / "01_paired_SSIM_test.csv",
        index=False
    )

    # --------------------------------------------------------
    # SSIM figure
    # --------------------------------------------------------

    fig = plt.figure(figsize=(7, 6))

    rng = np.random.default_rng(SEED)

    x1 = np.ones(len(df)) * 1
    x2 = np.ones(len(df)) * 2

    plt.plot(
        [x1, x2],
        [
            df["SSIM_reference_EF"].values,
            df["SSIM_Ouyang_EF"].values
        ],
        linewidth=0.7,
        alpha=0.35
    )

    plt.scatter(
        x1 + rng.normal(0, 0.035, len(df)),
        df["SSIM_reference_EF"]
    )

    plt.scatter(
        x2 + rng.normal(0, 0.035, len(df)),
        df["SSIM_Ouyang_EF"]
    )

    plt.xticks(
        [1, 2],
        ["Reference EF", "Ouyang EF"]
    )

    plt.ylabel("SSIM")
    plt.title("Paired generated-video SSIM")

    save_figure(
        fig,
        "03_Paired_SSIM.png"
    )

    # --------------------------------------------------------
    # Optional LPIPS
    # --------------------------------------------------------

    lpips_available = False

    try:
        import lpips
        import torch

        lpips_model = lpips.LPIPS(
            net="alex"
        )

        lpips_available = True

    except Exception as exc:
        warnings.warn(
            "LPIPS not available. "
            "SSIM and all non-LPIPS analyses will still run.\n"
            f"Reason: {exc}"
        )

    if lpips_available:

        print()
        print("Computing LPIPS...")
        print()

        lpips_rows = []

        # Use CPU to avoid competing with any remaining GPU job.
        lpips_model = lpips_model.cpu()
        lpips_model.eval()

        for _, row in df.iterrows():

            patient = row["patient"]

            reference_gif = find_gif(
                REFERENCE_BATCH,
                patient
            )

            predicted_gif = find_gif(
                PREDICTED_BATCH,
                patient
            )

            camus_frames = load_camus_sequence(
                patient
            )

            ref_gen = load_gif_frames(
                reference_gif
            )

            pred_gen = load_gif_frames(
                predicted_gif
            )

            def lpips_score(generated, reference):

                values = []

                for g, r in zip(
                    generated,
                    reference
                ):

                    g_rgb = np.repeat(
                        g[:, :, None],
                        3,
                        axis=2
                    )

                    r_rgb = np.repeat(
                        r[:, :, None],
                        3,
                        axis=2
                    )

                    gt = torch.from_numpy(
                        g_rgb
                    ).permute(
                        2, 0, 1
                    ).float() / 127.5 - 1.0

                    rt = torch.from_numpy(
                        r_rgb
                    ).permute(
                        2, 0, 1
                    ).float() / 127.5 - 1.0

                    with torch.no_grad():

                        score = lpips_model(
                            gt.unsqueeze(0),
                            rt.unsqueeze(0)
                        )

                    values.append(
                        float(
                            score.item()
                        )
                    )

                return float(
                    np.mean(values)
                )

            lpips_rows.append({
                "patient": patient,
                "LPIPS_reference_EF": lpips_score(
                    ref_gen,
                    camus_frames
                ),
                "LPIPS_Ouyang_EF": lpips_score(
                    pred_gen,
                    camus_frames
                )
            })

        lpips_df = pd.DataFrame(
            lpips_rows
        )

        df = df.merge(
            lpips_df,
            on="patient",
            how="left"
        )

        lpips_df.to_csv(
            TABLE_DIR / "03_LPIPS_metrics.csv",
            index=False
        )

        lpips_test = paired_wilcoxon(
            df["LPIPS_reference_EF"],
            df["LPIPS_Ouyang_EF"]
        )

        pd.DataFrame([{
            "test": "Wilcoxon signed-rank",
            "metric": "LPIPS",
            **lpips_test
        }]).to_csv(
            STATS_DIR / "02_paired_LPIPS_test.csv",
            index=False
        )

        fig = plt.figure(figsize=(7, 6))

        rng = np.random.default_rng(SEED + 1)

        x1 = np.ones(len(df)) * 1
        x2 = np.ones(len(df)) * 2

        plt.plot(
            [x1, x2],
            [
                df["LPIPS_reference_EF"].values,
                df["LPIPS_Ouyang_EF"].values
            ],
            linewidth=0.7,
            alpha=0.35
        )

        plt.scatter(
            x1 + rng.normal(0, 0.035, len(df)),
            df["LPIPS_reference_EF"]
        )

        plt.scatter(
            x2 + rng.normal(0, 0.035, len(df)),
            df["LPIPS_Ouyang_EF"]
        )

        plt.xticks(
            [1, 2],
            ["Reference EF", "Ouyang EF"]
        )

        plt.ylabel("LPIPS (lower = more similar)")
        plt.title("Paired generated-video LPIPS")

        save_figure(
            fig,
            "04_Paired_LPIPS.png"
        )

    # --------------------------------------------------------
    # Optional generated-EF analysis
    # --------------------------------------------------------

    if (
        REFERENCE_GEN_EF_CSV is not None
        and PREDICTED_GEN_EF_CSV is not None
    ):

        print()
        print("Loading generated-EF scores...")
        print()

        gen_ref = load_generated_ef_csv(
            REFERENCE_GEN_EF_CSV
        )

        gen_pred = load_generated_ef_csv(
            PREDICTED_GEN_EF_CSV
        )

        gen_ref = gen_ref.rename(
            columns={
                "generated_ef":
                    "generated_ef_reference_condition"
            }
        )

        gen_pred = gen_pred.rename(
            columns={
                "generated_ef":
                    "generated_ef_ouyang_condition"
            }
        )

        df = df.merge(
            gen_ref,
            on="patient",
            how="inner"
        )

        df = df.merge(
            gen_pred,
            on="patient",
            how="inner"
        )

        # Generator fidelity to the requested EF
        df["generator_error_reference"] = (
            df["generated_ef_reference_condition"]
            - df["reference_ef"]
        ).abs()

        df["generator_error_ouyang"] = (
            df["generated_ef_ouyang_condition"]
            - df["reference_ef"]
        ).abs()

        # How closely generated EF follows the supplied Ouyang EF.
        df["conditioning_adherence_ouyang"] = (
            df["generated_ef_ouyang_condition"]
            - df["predicted_ef"]
        ).abs()

        df["downstream_error_change"] = (
            df["generator_error_ouyang"]
            - df["generator_error_reference"]
        )

        generated_summary = pd.DataFrame([
            {
                "condition": "Reference EF",
                **calculate_metrics(
                    df["reference_ef"],
                    df[
                        "generated_ef_reference_condition"
                    ]
                )
            },
            {
                "condition": "Ouyang EF",
                **calculate_metrics(
                    df["reference_ef"],
                    df[
                        "generated_ef_ouyang_condition"
                    ]
                )
            }
        ])

        generated_summary.to_csv(
            TABLE_DIR / "04_generated_EF_metrics.csv",
            index=False
        )

        df[
            [
                "patient",
                "reference_ef",
                "predicted_ef",
                "ouyang_abs_error",
                "generated_ef_reference_condition",
                "generated_ef_ouyang_condition",
                "generator_error_reference",
                "generator_error_ouyang",
                "conditioning_adherence_ouyang",
                "downstream_error_change"
            ]
        ].to_csv(
            TABLE_DIR / "05_paired_generation_results.csv",
            index=False
        )

        # Paired downstream test
        downstream_test = paired_wilcoxon(
            df["generator_error_reference"],
            df["generator_error_ouyang"]
        )

        pd.DataFrame([{
            "test": "Wilcoxon signed-rank",
            "metric": "Absolute generated-EF error vs CAMUS reference",
            **downstream_test
        }]).to_csv(
            STATS_DIR / "03_paired_generated_EF_error_test.csv",
            index=False
        )

        # Upstream error -> downstream change
        rho, p = spearmanr(
            df["ouyang_abs_error"],
            df["downstream_error_change"]
        )

        pd.DataFrame([{
            "test": "Spearman correlation",
            "x": "Ouyang absolute EF error",
            "y": "Change in downstream generation EF error",
            "rho": rho,
            "p_value": p,
            "n": len(df)
        }]).to_csv(
            STATS_DIR / "04_error_propagation_spearman.csv",
            index=False
        )

        # Paired downstream figure
        fig = plt.figure(figsize=(7, 6))

        rng = np.random.default_rng(SEED + 2)

        x1 = np.ones(len(df)) * 1
        x2 = np.ones(len(df)) * 2

        plt.plot(
            [x1, x2],
            [
                df["generator_error_reference"].values,
                df["generator_error_ouyang"].values
            ],
            linewidth=0.7,
            alpha=0.35
        )

        plt.scatter(
            x1 + rng.normal(0, 0.035, len(df)),
            df["generator_error_reference"]
        )

        plt.scatter(
            x2 + rng.normal(0, 0.035, len(df)),
            df["generator_error_ouyang"]
        )

        plt.xticks(
            [1, 2],
            ["Reference EF", "Ouyang EF"]
        )

        plt.ylabel(
            "|Generated EF − CAMUS EF| (%)"
        )

        plt.title(
            "Paired downstream EF fidelity"
        )

        save_figure(
            fig,
            "05_Paired_generated_EF_error.png"
        )

        # Error propagation scatter
        fig = plt.figure(figsize=(7, 6))

        plt.scatter(
            df["ouyang_abs_error"],
            df["downstream_error_change"]
        )

        plt.axhline(
            0,
            linestyle="--"
        )

        plt.xlabel(
            "Ouyang EF absolute error (%)"
        )

        plt.ylabel(
            "Change in downstream EF error (%)"
        )

        plt.title(
            "Upstream EF error vs downstream generation error"
        )

        plt.text(
            0.05,
            0.95,
            f"Spearman ρ = {rho:.2f}\n"
            f"p = {p:.3g}",
            transform=plt.gca().transAxes,
            va="top"
        )

        save_figure(
            fig,
            "06_EF_error_propagation.png"
        )

    else:
        print()
        print(
            "Generated-EF CSVs not supplied; "
            "downstream EF fidelity tests were skipped."
        )

    # --------------------------------------------------------
    # Master data export
    # --------------------------------------------------------

    df.to_csv(
        DATA_DIR / "master_camus_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # Statistical summary
    # --------------------------------------------------------

    stats_rows = [
        {
            "analysis": "Ouyang vs CAMUS reference EF",
            "metric": "MAE (%)",
            "value": upstream["MAE"]
        },
        {
            "analysis": "Ouyang vs CAMUS reference EF",
            "metric": "RMSE (%)",
            "value": upstream["RMSE"]
        },
        {
            "analysis": "Ouyang vs CAMUS reference EF",
            "metric": "R2",
            "value": upstream["R2"]
        },
        {
            "analysis": "Visual quality",
            "metric": "SSIM reference-EF mean",
            "value": df[
                "SSIM_reference_EF"
            ].mean()
        },
        {
            "analysis": "Visual quality",
            "metric": "SSIM Ouyang-EF mean",
            "value": df[
                "SSIM_Ouyang_EF"
            ].mean()
        }
    ]

    if (
        "LPIPS_reference_EF" in df.columns
        and "LPIPS_Ouyang_EF" in df.columns
    ):
        stats_rows.extend([
            {
                "analysis": "Visual quality",
                "metric": "LPIPS reference-EF mean",
                "value": df[
                    "LPIPS_reference_EF"
                ].mean()
            },
            {
                "analysis": "Visual quality",
                "metric": "LPIPS Ouyang-EF mean",
                "value": df[
                    "LPIPS_Ouyang_EF"
                ].mean()
            }
        ])

    pd.DataFrame(stats_rows).to_csv(
        STATS_DIR / "00_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    readme = f"""
CAMUS / EchoDiffusion Analysis Output
======================================

Output root:
{OUTPUT_ROOT}

FIGURES
-------
01_Ouyang_EF_reference_vs_predicted.png
    Reference CAMUS EF vs Ouyang predicted EF.

02_Ouyang_Bland_Altman.png
    Agreement between CAMUS reference EF and Ouyang EF.

03_Paired_SSIM.png
    Patient-matched SSIM comparison:
    reference-EF generation vs Ouyang-EF generation.

04_Paired_LPIPS.png
    Patient-matched LPIPS comparison, when lpips is installed.

05_Paired_generated_EF_error.png
    Patient-matched downstream generated-EF error.
    Created only when generated-EF score CSVs are supplied.

06_EF_error_propagation.png
    Ouyang upstream EF error vs change in downstream generation EF error.
    Created only when generated-EF score CSVs are supplied.

TABLES
------
01_upstream_ouyang_metrics.csv
02_visual_metrics.csv
03_LPIPS_metrics.csv
04_generated_EF_metrics.csv
05_paired_generation_results.csv

STATISTICS
----------
01_paired_SSIM_test.csv
02_paired_LPIPS_test.csv
03_paired_generated_EF_error_test.csv
04_error_propagation_spearman.csv
00_summary.csv

DATA
----
master_camus_analysis.csv

PRIMARY INTERPRETATION
---------------------
The central comparison is paired within patient:
    CAMUS reference EF -> 1SCM_v2
vs
    Ouyang predicted EF -> 1SCM_v2

Primary question:
    Does replacing reference EF with model-predicted EF change
    downstream functional fidelity and visual quality?

The downstream generated-EF analysis should be interpreted separately
from image-similarity metrics. SSIM/LPIPS measure visual similarity,
not clinical correctness.

IMPORTANT
---------
CAMUS sequences are the half-sequence ED->ES acquisitions used in this
project. They are temporally resampled to 16 frames only for the
patient-level visual comparison with the 1SCM_v2 generated videos.

Seed used for paired generation:
{SEED}
"""

    (OUTPUT_ROOT / "README.txt").write_text(
        readme.strip(),
        encoding="utf-8"
    )

    print()
    print("============================================")
    print("CAMUS ANALYSIS COMPLETE")
    print("============================================")
    print("Results:", OUTPUT_ROOT)
    print("Figures:", FIGURE_DIR)
    print("Tables:", TABLE_DIR)
    print("Statistics:", STATS_DIR)
    print("Data:", DATA_DIR)
    print("============================================")


if __name__ == "__main__":
    main()
