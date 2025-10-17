import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ----------------------------
# Konstanta fisika & parameter
# ----------------------------
h = 6.626e-34
c = 3.0e8
eV = 1.602e-19
# default kisi difraksi (600 lines/mm)
d_default = 1.0 / (600e3)  # m (600 lines/mm -> 600000 lines/m)

# ----------------------------
# Fungsi utilitas
# ----------------------------
def wavelength_to_rgb(wl):
    """Approximate visible wavelength (nm) -> RGB (0..1)."""
    wl = float(wl)
    if wl < 380 or wl > 780:
        return (0.0, 0.0, 0.0)
    if wl < 440:
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0
    # intensity scaling near edges
    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl > 700:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    else:
        factor = 1.0
    return (max(0.0, r*factor), max(0.0, g*factor), max(0.0, b*factor))

def photon_energy_eV(wl_nm):
    wl_m = wl_nm * 1e-9
    E_J = h * c / wl_m
    return E_J / eV

def zone_label(wl):
    if np.isnan(wl):
        return "None"
    w = float(wl)
    if 380 <= w < 450:
        return "Ungu"
    if 450 <= w < 495:
        return "Biru"
    if 495 <= w < 570:
        return "Hijau"
    if 570 <= w < 590:
        return "Kuning"
    if 590 <= w < 620:
        return "Jingga"
    if 620 <= w <= 780:
        return "Merah"
    return "None"

# ----------------------------
# UI konfigurasi
# ----------------------------
st.set_page_config(page_title="LED Spectrum Mixer & Analyzer", layout="wide")
st.title("LED Spectrum Mixer & Analyzer")
st.caption("Masukkan nilai X–Y (mm) lalu lihat λ, E, dan spektrum warna; grafik bergaya action-spectrum.")

# sidebar controls
st.sidebar.header("Pengaturan")
d_input = st.sidebar.number_input("Jarak kisi d (m)", value=float(d_default), format="%.6e")
sigma_nm = st.sidebar.slider("Lebar Gaussian per-peak (σ, nm)", 2.0, 80.0, 12.0, step=1.0)
normalize = st.sidebar.checkbox("Normalisasi kurva (maks = 1)", value=True)
show_peaks = st.sidebar.checkbox("Tandai λ tiap data pada grafik", value=True)
download = st.sidebar.checkbox("Tampilkan tombol ekspor CSV", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Rentang warna yang digunakan: Ungu, Biru, Hijau, Kuning, Jingga, Merah (380–780 nm).")

# ----------------------------
# Input default data (editable)
# ----------------------------
st.subheader("Input data X–Y (mm)")
default = pd.DataFrame({
    "X (mm)": [100, 100, 100, 100, 100],
    "Y (mm)": [20, 25, 30, 35, 40]
})
data = st.data_editor(default, num_rows="dynamic", use_container_width=True, key="xy_table")

# ----------------------------
# Perhitungan fisika
# ----------------------------
X_mm = np.array(data["X (mm)"], dtype=float)
Y_mm = np.array(data["Y (mm)"], dtype=float)
X = X_mm * 1e-3
Y = Y_mm * 1e-3

# compute theta safely (avoid division by zero)
with np.errstate(divide='ignore', invalid='ignore'):
    theta_rad = np.arctan2(Y, X)  # rad
    theta_deg = np.degrees(theta_rad)

# compute lambda via d * sin(theta)
lambda_m = d_input * np.sin(theta_rad)
lambda_nm = lambda_m * 1e9
# mark invalid or negative lambdas as NaN
lambda_nm = np.where(lambda_nm > 0, lambda_nm, np.nan)

# energy
E_eV = np.array([photon_energy_eV(wl) if not np.isnan(wl) else np.nan for wl in lambda_nm])

# assemble results table
results = pd.DataFrame({
    "X (mm)": X_mm,
    "Y (mm)": Y_mm,
    "θ (°)": theta_deg,
    "λ (nm)": lambda_nm,
    "E (eV)": E_eV
})
results["Zona Warna"] = results["λ (nm)"].apply(zone_label)

# display results (formatted)
st.subheader("Hasil Perhitungan Spektrum Emisi LED")
st.dataframe(results.style.format({"θ (°)": "{:.2f}", "λ (nm)": "{:.1f}", "E (eV)": "{:.2f}"}), use_container_width=True)

# color blocks for each data point
st.markdown("### Warna dominan tiap titik data")
cols = st.columns(len(results))
for i, (_, row) in enumerate(results.iterrows()):
    wl = row["λ (nm)"]
    if pd.isna(wl):
        rgb = (0.9, 0.9, 0.9)
        label = "Out of range"
    else:
        rgb = wavelength_to_rgb(max(380, min(780, wl)))
        label = f"{row['Zona Warna']} ({wl:.0f} nm)"
    color_css = f"background: rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}); height:60px; border-radius:6px; border:1px solid #ddd"
    with cols[i]:
        st.markdown(f"<div style='{color_css}'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center'><small>{label}</small></div>", unsafe_allow_html=True)

# ----------------------------
# Build continuous spectrum background (380-780 nm)
# and intensity curve as sum of Gaussians centered at λ values
# ----------------------------
wavelengths = np.linspace(380, 780, 1000)
# create RGB background image (height small)
bg_rgb = np.array([wavelength_to_rgb(w) for w in wavelengths])  # shape (N,3)
bg_img = np.tile(bg_rgb.reshape(1, len(wavelengths), 3), (40, 1, 1))  # small height for imshow

# intensity: sum of Gaussians for each valid lambda
I = np.zeros_like(wavelengths)
valid_wls = [wl for wl in lambda_nm if not np.isnan(wl)]
for wl in valid_wls:
    I += np.exp(-0.5 * ((wavelengths - wl) / sigma_nm)**2)

if normalize and I.max() > 0:
    I = I / I.max()

# create figure similar to action-spectrum style
st.subheader("Spektrum emisi (aksi visual bergradasi warna)")

fig, ax = plt.subplots(figsize=(12, 4))
# show color gradient background
ax.imshow(bg_img, extent=[wavelengths[0], wavelengths[-1], 0, 1], aspect='auto', origin='lower')

# plot curve (use dark line) and filled area with white alpha to emulate action spectrum contrast
ax.plot(wavelengths, I, color='k', linewidth=1.6)
ax.fill_between(wavelengths, I, color='white', alpha=0.35)

# optionally mark peaks and annotate
if show_peaks:
    for wl in valid_wls:
        ax.axvline(wl, color=wavelength_to_rgb(wl), linewidth=1.2, alpha=0.9)
        ax.text(wl, min(1.02, 0.98*I.max()), f"{wl:.1f} nm", rotation=90, va='bottom', ha='center', fontsize=8)

ax.set_xlim(380, 780)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Panjang gelombang (nm)")
ax.set_ylabel("Intensitas relatif (arb. unit)")
ax.set_title("Spektrum Emisi (gradasi warna + kurva intensitas)")
ax.grid(alpha=0.25)
st.pyplot(fig)

# ----------------------------
# Energi vs lambda plot (colored markers)
# ----------------------------
st.subheader("Energi foton terhadap panjang gelombang")
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(wavelengths, photon_energy_eV(wavelengths), color='gray', linestyle='--', alpha=0.6, label="E = hc/λ")
ax2.scatter(results["λ (nm)"], results["E (eV)"],
            color=[wavelength_to_rgb(max(380, min(780, wl))) if not np.isnan(wl) else (0.5,0.5,0.5) for wl in results["λ (nm)"]],
            edgecolors='k', s=90)
ax2.set_xlim(380, 780)
ax2.set_xlabel("Panjang gelombang (nm)")
ax2.set_ylabel("Energi (eV)")
ax2.set_title("Hubungan energi foton dan panjang gelombang")
ax2.grid(alpha=0.25)
st.pyplot(fig2)

# ----------------------------
# Export CSV
# ----------------------------
if download:
    csv_buf = results.to_csv(index=False).encode('utf-8')
    st.download_button("Unduh hasil (CSV)", data=csv_buf, file_name="hasil_spektrum.csv", mime="text/csv")

# ----------------------------
# Pedagogical notes
# ----------------------------
st.markdown("---")
st.subheader("Interpretasi singkat")
st.write(
    "- Grafik utama menampilkan latar warna (380–780 nm) dan kurva intensitas yang dihasilkan "
    "dari penjumlahan Gaussian pada panjang gelombang yang dihitung dari data X–Y.\n"
    "- Titik-titik vertikal menandai posisi λ hasil perhitungan. Zona warna (Ungu→Merah) memberikan label visual.\n"
    "- Energi foton dihitung dengan E = hc/λ; kurva terpisah memperlihatkan hubungan kuantitatif."
)
