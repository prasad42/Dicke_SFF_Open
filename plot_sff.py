import matplotlib.pyplot as plt
import numpy as np
from dicke_sff_lib import *
import dicke_sff_lib_closed as sff_lib
from parameters import *
from tqdm import tqdm
import os


g_arr = [1.0]
γ_arr = np.array([0.0, 0.01, 0.1, 1.0])
# γ_arr = [0.0]
β = 0

# Time list for open model with MCWF method
StartTime = 0
LateTime = 100
tlist_open1 = np.arange(StartTime, LateTime, 0.01)

def main():
    fig, axs = plt.subplots(1, 1, figsize=(3.4, 2.4), sharey=True)
    # colors = plt.cm.viridis(γ_arr)
    
    N = int((2*j+1)*M)

    ax = axs
    # ax.text(0.33, 0.2, fr"$\gamma={γ}$", transform=ax.transAxes,
    #         fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.tick_params(direction='in', which='both')

    # colors = plt.cm.cividis(np.linspace(0, 1, len(γ_arr)))
    linestyles = ['--','-.', ':', '-']

    # Plot GOE theory (only once per g)
    tlist_GOE = tlist
    deg = None
    unfl_proc = None

    for γ_ind, γ in enumerate(γ_arr):
        gc = np.round((np.sqrt(ω/ω0*(γ**2/4 + ω**2)))/2, 2)
        linestyle = linestyles[γ_ind % len(linestyles)]
        # color = colors[γ_ind]
        for g_ind, g in enumerate(tqdm(g_arr)):
            if method == "mcsolve":
                ntraj = 100
                tlist_open = tlist_open1
                if γ == 0.0:
                    ntraj = 1  
                    sff_list, _ = sff_lib.sff_list_fun(ω, ω0, j, M, g, β, tlist_open, v=None, deg=deg, unfl_proc=unfl_proc)
                    sff_list = sff_rl_fun(tlist_open, sff_list, win=50)
                    ax.plot(tlist_open, sff_list, linestyle=linestyle,
                        label=fr"$\gamma/\omega={γ/2}$", linewidth=1)
                    continue
                else:
                    ntraj = 100
                    sff_list = sff_open_list_fun(ω, ω0, j, M, g, β, γ, tlist_open, ntraj, nproc)
            elif method == "ssesolve":
                sff_list = sff_open_list_sse_fun(ω, ω0, j, M, g, β, γ, tlist_open, ntraj, nproc)
            ax.plot(tlist_open, sff_list, linestyle=linestyle,
                    label=fr"$\gamma/\omega={γ/2}$", linewidth=1)

    # ax.plot(tlist, Kpoi, ':r', label="Poisson", linewidth=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2,1e2)
    ax.tick_params(labelsize=8, direction='in', length=3, width=0.8)
    ax.set_ylabel(fr"DSPF($\beta=0,t$)", fontsize=9)
    # ax.set_xlim(1e-2,1e2)
    handles, labels = axs.get_legend_handles_labels()
    fig.text(
        0.5, 0.01,            # x=center, y=slightly below both subplots
        "Time t",            # your x-label
        ha='center', va='top',
        fontsize=10
        )
    plt.legend(fontsize=7, ncols=2)

    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/Dicke_dspf_open_j={j}_M={M}_β={β}_g={g}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
