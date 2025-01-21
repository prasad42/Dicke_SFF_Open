import matplotlib.pyplot as plt
from dicke_sff_lib import *
from parameters import *

def main():
    # print(sff_goe_list)
    plt.figure(figsize=(10,5))
    # plt.suptitle(f"j={j} M={M} β={β}")
    for γ_ind, γ in enumerate(γ_arr):
        gc={np.round((np.sqrt(ω/ω0*(γ**2/4+ω**2))/2),2)}
        plt.subplot(1,2,γ_ind+1)
        plt.title(f"γ = {γ}")
        print(f"gc={gc}")
        plt.grid()
        for g_ind, g in tqdm(enumerate(g_arr)):
            # overlap = ss_psi0_overlap_fun(ω, ω0, j, M, g, β, γ)
            sff_list = sff_open_list_fun(ω, ω0, j, M, g, β, γ, tlist_open, ntraj, nproc)
            # plt.subplot(10,2,g_ind+1)
            plt.plot(tlist_open,sff_list,label=f"g={g}")
            # plt.axhline(overlap.real, linestyle="--", color="k")
            plt.xscale('log'); plt.yscale('log')
            
    plt.legend()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    
    plt.savefig(f"plots/Dicke_sff_open_j={j}_M={M}_β={β}.png")
    plt.show()

if __name__ == "__main__":
    main()