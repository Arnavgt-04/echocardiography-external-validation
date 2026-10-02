import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT = (
    r"C:\insigneo\results\camus_evaluation"
    r"\data\master_camus_analysis.csv"
)

OUTPUT_DIR = (
    r"C:\insigneo\results\camus_evaluation"
    r"\ef_error_propagation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT)

print("Patients:", len(df))

# --------------------------------------------------
# Calculate deterioration caused by predicted EF
# --------------------------------------------------

# SSIM: HIGHER is better
# Positive value = predicted-EF conditioning made SSIM worse
df["SSIM_deterioration"] = (
    df["SSIM_reference_EF"]
    - df["SSIM_Ouyang_EF"]
)

# LPIPS: LOWER is better
# Positive value = predicted-EF conditioning made LPIPS worse
df["LPIPS_deterioration"] = (
    df["LPIPS_Ouyang_EF"]
    - df["LPIPS_reference_EF"]
)

# --------------------------------------------------
# Spearman correlations
# --------------------------------------------------

rho_ssim, p_ssim = spearmanr(
    df["ouyang_abs_error"],
    df["SSIM_deterioration"]
)

rho_lpips, p_lpips = spearmanr(
    df["ouyang_abs_error"],
    df["LPIPS_deterioration"]
)

print()
print("========================================")
print("EF ERROR PROPAGATION")
print("========================================")

print()
print("EF error vs SSIM deterioration")
print(f"Spearman rho: {rho_ssim:.4f}")
print(f"p-value:      {p_ssim:.6g}")

print()
print("EF error vs LPIPS deterioration")
print(f"Spearman rho: {rho_lpips:.4f}")
print(f"p-value:      {p_lpips:.6g}")

# --------------------------------------------------
# Save statistics
# --------------------------------------------------

stats = pd.DataFrame([
    {
        "comparison": "Ouyang EF error vs SSIM deterioration",
        "n": len(df),
        "spearman_rho": rho_ssim,
        "p_value": p_ssim
    },
    {
        "comparison": "Ouyang EF error vs LPIPS deterioration",
        "n": len(df),
        "spearman_rho": rho_lpips,
        "p_value": p_lpips
    }
])

stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "EF_error_propagation_statistics.csv"
    ),
    index=False
)

# Save patient-level values too
df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "EF_error_propagation_patient_data.csv"
    ),
    index=False
)

# --------------------------------------------------
# GRAPH 1 — EF error vs SSIM deterioration
# --------------------------------------------------

plt.figure(figsize=(6, 5))

plt.scatter(
    df["ouyang_abs_error"],
    df["SSIM_deterioration"],
    alpha=0.75
)

# Visual trend line
x = df["ouyang_abs_error"].values
y = df["SSIM_deterioration"].values

coef = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)

plt.plot(
    x_line,
    np.polyval(coef, x_line),
    linestyle="--"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Absolute LVEF prediction error (percentage points)"
)

plt.ylabel(
    "SSIM deterioration\n"
    "(Reference-EF SSIM − Predicted-EF SSIM)"
)

plt.title(
    "LVEF prediction error vs SSIM deterioration"
)

plt.text(
    0.05,
    0.95,
    f"Spearman ρ = {rho_ssim:.2f}\n"
    f"p = {p_ssim:.3g}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_EF_error_vs_SSIM_deterioration.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# GRAPH 2 — EF error vs LPIPS deterioration
# --------------------------------------------------

plt.figure(figsize=(6, 5))

plt.scatter(
    df["ouyang_abs_error"],
    df["LPIPS_deterioration"],
    alpha=0.75
)

x = df["ouyang_abs_error"].values
y = df["LPIPS_deterioration"].values

coef = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)

plt.plot(
    x_line,
    np.polyval(coef, x_line),
    linestyle="--"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Absolute LVEF prediction error (percentage points)"
)

plt.ylabel(
    "LPIPS deterioration\n"
    "(Predicted-EF LPIPS − Reference-EF LPIPS)"
)

plt.title(
    "LVEF prediction error vs LPIPS deterioration"
)

plt.text(
    0.05,
    0.95,
    f"Spearman ρ = {rho_lpips:.2f}\n"
    f"p = {p_lpips:.3g}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_EF_error_vs_LPIPS_deterioration.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved results to:")
print(OUTPUT_DIR)