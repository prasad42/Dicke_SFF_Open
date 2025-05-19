import matplotlib.pyplot as plt
import numpy as np
from dicke_sff_lib import *
from parameters import *
from tqdm import tqdm
import os

g_arr = np.round(np.arange(0.1, 1.05, 0.1), 2)
γ = 2.2
β_vals = [0, 5]
colors = plt.cm.cividis(np.linspace(0, 1, len(g_arr)))  # colour variation for g

def main():
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=False)
    gc = np.round((np.sqrt(ω / ω0 * (γ**2 / 4 + ω**2))) / 2, 2)

    for i, β in enumerate(β_vals):
        ax = axs[i]
        ax.set_xlim(1e-3,1e2)

        for g_idx, g in enumerate(g_arr):
            tlist_open = np.arange(StartTime, LateTime, 0.001 if β==0 else 0.01)
            if method == "mcsolve":
                sff_list = sff_open_list_fun(ω, ω0, j, M, g, β, γ, tlist_open, ntraj, nproc)
            elif method == "ssesolve":
                sff_list = sff_open_list_sse_fun(ω, ω0, j, M, g, β, γ, tlist_open, ntraj, nproc)
            else:
                raise ValueError("Unknown method. Choose 'mcsolve' or 'ssesolve'.")

            ax.plot(tlist_open, sff_list, label=r"$g/g_{cγ}$"+f"={np.round(g/gc, 2)}", linewidth=1, color=colors[g_idx])

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Time t", fontsize=9)
        # Place the label inside the subplot
        # ax.text(0.25, 0.5, fr"$\beta\omega={β}$", transform=ax.transAxes,
        #         fontsize=8, va='top', ha='center',
        #         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5))
        # Major ticks (thicker)
        ax.tick_params(
            which='major',
            direction='in',
            length=4,
            width=1,
            colors='black',
            grid_color='gray',
            grid_alpha=0.5
        )

        # Minor ticks (thinner)
        ax.tick_params(
            which='minor',
            direction='in',
            length=2,
            width=0.5,
            colors='black'
        )
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    axs[0].set_ylabel(fr"DSPF($\beta,t$)", fontsize=9)
    # axs[0].legend(fontsize=6, title_fontsize=7, loc='upper right', frameon=True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.06))

    # fig.suptitle(fr"Comparison at $\gamma = {γ}$", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/Dicke_dspf_beta_subplots_j={j}_M={M}_γ={γ}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
