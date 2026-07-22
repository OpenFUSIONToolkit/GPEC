#!/usr/bin/env python3
"""plot_deltacoil_svd.py — SVD robustness of the delta_coil matrix vs the truncation dmlim.
LEFT: singular-value spectrum. RIGHT: condition number per run (this is what varies — the flat
tolerance-bar panel was removed because it carried no information: the tolerance scan is bit-identical
and is stated in the caption instead). Values from deltacoil_svd_tol.jl on the DIII-D-like case."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

sigma = {
 "nominal":                 [28.99, 15.57, 12.94, 9.612, 6.048, 3.607],
 "dmlim=0.30":              [30.49, 15.55, 13.76, 9.490, 5.997, 3.594],
 "dmlim=0.42":              [29.04, 15.57, 12.99, 9.607, 6.045, 3.607],
 "dmlim=0.50 (q5 dropped)": [15.73, 10.62, 6.220, 4.523, 2.437, 1.395],
}
cols = {"nominal":"#111111","dmlim=0.30":"#2E86C1","dmlim=0.42":"#28B463","dmlim=0.50 (q5 dropped)":"#C0392B"}
# condition number (σ1/σ_min) per run — this genuinely varies
cond_runs  = ["nominal","dmlim=0.30","dmlim=0.42","dmlim=0.50"]
cond_vals  = [26.2, 27.2, 26.2, 11.3]
cond_cols  = ["#111111","#2E86C1","#28B463","#C0392B"]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
fig.suptitle("delta_coil matrix — SVD robustness (DIII-D-like)",fontsize=13,fontweight="bold")

x=np.arange(1,7)
for k,v in sigma.items():
    ax1.plot(x,v,"o-",color=cols[k],lw=2,ms=6,label=k)
ax1.set_yscale("log"); ax1.set_xlabel("singular value index  i"); ax1.set_ylabel("σᵢ")
ax1.set_title("Singular-value spectrum vs truncation\n(stable below threshold; whole matrix shrinks when q5 leaves)",
              fontsize=10.5,fontweight="bold")
ax1.legend(fontsize=8.5); ax1.grid(alpha=0.25,which="both")

bars=ax2.bar(cond_runs,cond_vals,color=cond_cols)
for b,v in zip(bars,cond_vals): ax2.text(b.get_x()+b.get_width()/2,v+0.5,f"{v:.1f}",ha="center",fontsize=9)
ax2.set_ylabel("condition number  σ₁ / σ_min"); ax2.set_ylim(0,40)
ax2.set_title("Conditioning per run\n(well-conditioned ≈ 26; drops to 11 when q5 is dropped)",
              fontsize=10.5,fontweight="bold")
ax2.grid(alpha=0.2,axis="y")
ax2.text(0.5,0.93,"Integration-tolerance scan (1e-8…1e-12): the σ-spectrum,\n"
         "norms and cosine are all BIT-IDENTICAL — omitted from the plot\n"
         "because there is nothing to show (delta_coil is tolerance-independent).",
         transform=ax2.transAxes,ha="center",fontsize=8,style="italic",
         bbox=dict(boxstyle="round",fc="#F4F6FA",ec="#B4BED2"))

out="benchmarks/deltacoil_sensitivity/figures/deltacoil_svd.png"
import os; os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,dpi=140,bbox_inches="tight"); print("wrote",out)
