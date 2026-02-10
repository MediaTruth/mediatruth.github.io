import rawpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os

def run_precision_data_audit(src_path, label, color, out_dir):
    if not os.path.exists(src_path):
        return

    files = [f for f in os.listdir(src_path) if f.lower().endswith('.cr2')]
    os.makedirs(out_dir, exist_ok=True)

    for filename in files:
        img_path = os.path.join(src_path, filename)
        try:
            with rawpy.imread(img_path) as raw:
                raw_data = raw.raw_image_visible.astype(np.float32)
                r, gr, gb, b = raw_data[0::2,0::2], raw_data[0::2,1::2], raw_data[1::2,0::2], raw_data[1::2,1::2]
                
                min_h, min_w = min(r.shape[0], gr.shape[0], gb.shape[0], b.shape[0]), min(r.shape[1], gr.shape[1], gb.shape[1], b.shape[1])
                lum_14bit = (r[:min_h, :min_w] * 0.299 + ((gr[:min_h, :min_w] + gb[:min_h, :min_w]) / 2) * 0.587 + b[:min_h, :min_w] * 0.114)

                fig = plt.figure(figsize=(14, 10), facecolor='#ffffff')
                gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
                
                # 14-Bit Linear Histogram
                ax0 = plt.subplot(gs[0])
                ax0.hist(lum_14bit.ravel(), bins=np.arange(0, 16385, 16), color=color, alpha=0.7, log=True)
                ax0.axvline(x=1023.5, color='black', linestyle='--')
                ax0.set_title(f"14-Bit Linear Audit: {filename} [{label}]")
                ax0.set_xlabel("Signal Intensity (0-16383)")
                ax0.set_ylabel("Pixel Frequency (Log)")

                # DATA TABLE: Starting precisely at 1023-1024
                # Custom bins to isolate the threshold and subsequent 32-level buckets
                bins_table = [1023, 1025, 1057, 1089, 1121, 1153, 1185, 1217, 1249, 1281]
                counts, _ = np.histogram(lum_14bit, bins=bins_table)
                
                bucket_labels = ["Lvl 1023-1024"] + [f"Lvl {bins_table[i]}-{bins_table[i+1]-1}" for i in range(1, len(bins_table)-1)]
                df = pd.DataFrame(list(zip(bucket_labels, counts)), columns=["Sensor Range", "Pixel Count"])

                ax_tbl = plt.subplot(gs[1])
                ax_tbl.axis('off')
                tbl = ax_tbl.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(11)
                tbl.scale(1, 1.8)

                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{label}_DATA_{os.path.splitext(filename)[0]}.png"), dpi=150)
                plt.close()
        except Exception:
            continue

# Target directories
output_folder = "/volumes/data/16bithist"
run_precision_data_audit("/volumes/data/real", "REAL", "blue", output_folder)
run_precision_data_audit("/volumes/data/fake", "JONAS_CLOUD", "red", output_folder)