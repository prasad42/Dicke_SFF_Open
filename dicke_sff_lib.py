import numpy as np
import qutip as qt
import os
import scipy.linalg as sl
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def DH_fun(ω, ω0, j, M, g):
    '''
    Dicke Hamiltonian for the following parameters.
    Args:
    - ω : frequency of the bosonic field
    - ω0 : Energy difference in spin states
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    '''
    a  = qt.tensor(qt.destroy(M), qt.qeye(int(2*j+1)))
    Jp = qt.tensor(qt.qeye(M), qt.jmat(j, '+'))
    Jm = qt.tensor(qt.qeye(M), qt.jmat(j, '-'))
    Jz = qt.tensor(qt.qeye(M), qt.jmat(j, 'z'))
    H0 = ω * a.dag() * a + ω0 * Jz
    H1 = 1.0 / np.sqrt(2*j) * (a + a.dag()) * (Jp + Jm)
    H = H0 + g * H1
    
    return H

def sff_rl_fun(ω, ω0, j, M, g, β, tlist, γ, ntraj, θntraj, nproc, win = 50):
    '''
    This function returns the rolling average of the sff over time.
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - β : Inverse Temperature
    - tlist : pass time list as an array
    - win: Window size for rolling average
    '''
    sff_list = sff_open_list_fun(ω, ω0, j, M, g, β, γ, tlist, ntraj, θntraj, nproc)
    sff_list = np.column_stack(sff_list)
    sff_rl = []
    for t_ind in range(0,len(tlist),1):
        win_start = int(t_ind)
        win_end = int(t_ind+win)
        sff_rl_val = np.average(sff_list[win_start:win_end], axis=0)
        sff_rl.append(sff_rl_val)
    sff_rl = np.array(sff_rl)

    return np.array(sff_rl)

def generate_goe_matrix(N):
    """
    Generate an NxN Gaussian Orthogonal Ensemble (GOE) matrix.
    """
    A = np.random.normal(0, 1, size=(N, N))
    A = (A + A.T) / 2
    return A

def sff_goe_list_fun(j, M, β, tlist, ntraj):
    """
    Compute the Spectral Form Factor (sff) for GOE matrices of size N,
    averaged over `num_realizations` random GOE matrices.
    
    Args:
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - β : Inverse Temperature
    - tlist: Array of time values (T) for which to compute sff.
    - ntraj: Number of GOE realizations to average over.
    
    Returns:
    - sff_list: Array of sff values for each T.
    """
    # N: Size of the GOE matrix.
    N  = int((2*j+1)*M)
    if not os.path.exists("sff_open"):
        os.mkdir("sff_open")
    file_path = f"sff_open/sff_goe_j={j}_M={M}_N={N}_β={β}_ntraj={ntraj}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        sff_list = np.zeros_like(tlist, dtype=np.float64)
        for _ in tqdm(range(ntraj)):
            H = generate_goe_matrix(N)
            eigvals = np.linalg.eigvalsh(H)
            for i, t in enumerate(tlist):
                exp_sum = np.sum(np.exp(-(β + 1j*t) * eigvals))
                sff_list[i] += np.abs(exp_sum)**2/(np.sum(np.exp(-(β) * eigvals)))**2
        sff_list /= ntraj
        np.save(file_path,sff_list)
    else:
        print(f"{file_path} already exists.")
    sff_list = np.load(file_path)

    return sff_list

def psi0_fun(ω, ω0, j, M, g, β):
    '''
    Returns CGS function
    Args:
    - ω : frequency of the bosonic field
    - ω0 : Energy difference in spin states
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - β : Inverse temperature
    '''
    π = np.pi
    H = DH_fun(ω, ω0, j, M, g)
    eigvals, eigvecs = H.eigenstates()
    psi0 = np.sum(np.exp(-β/2*eigvals) * eigvecs)
    psi0 /= np.sqrt(np.sum(np.exp(-β*eigvals)))
    # psi0 = psi0.unit()
    
    return psi0

def sff_open_list_fun(ω, ω0, j, M, g, β, γ, tlist, ntraj, nproc):
    '''
    Returns open sff
    Args:
    - ω : frequency of the bosonic field
    - ω0 : Energy difference in spin states
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    - β : Inverse temperature
    - γ : decay rate of the cavity
    - tlist : time list
    - ntraj : number of trejectories
    '''
    if not os.path.exists("sff_open"):
        os.mkdir(f"sff_open")
    file_path = f"sff_open/sff_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2,2)}_β={β}_γ={γ}_g={g}_ntraj={ntraj}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        H = DH_fun(ω, ω0, j, M, g)
        c_op = np.sqrt(γ) * qt.tensor(qt.destroy(M),qt.qeye(int(2*j+1)))
        psi0 = psi0_fun(ω, ω0, j, M, g, β)
        e_op = psi0 * psi0.dag()
        if ntraj>1:
            result = qt.mcsolve(H, psi0, tlist, c_op, e_op, ntraj = ntraj, options={"map":"loky", "num_cpus":nproc})
        elif ntraj==1 and γ==0.0:
            result = qt.sesolve(H, psi0, tlist, e_op)
        elif ntraj==1:
            result = qt.mcsolve(H, psi0, tlist, c_op, e_op, ntraj = ntraj, options={"map":"loky", "num_cpus":nproc})
        sff = np.abs(result.expect)
        np.save(file_path,sff)
    else:
        print(f"{file_path} already exists.")
    sff = np.load(file_path)
    sff = np.column_stack(sff)
    
    return sff

def ss_psi0_overlap_fun(ω, ω0, j, M, g, β, γ):
    """
    Calculate or load the steady-state overlap with a reference state.
    Args:
        ω (float): Frequency parameter.
        ω0 (float): Reference frequency.
        j (float): Spin quantum number.
        M (int): Dimension of the bosonic mode.
        g (float): Coupling constant.
        β (float): Inverse temperature (1/kT).
        γ (float): Damping rate.
    Returns:
        float: Overlap value between the steady-state density matrix and the reference state.
    """
    file_path = f"sff_open/overlap_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2/4+ω**2))/2,2)}_β={β}_γ={γ}_g={g}.npy"
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist, generating data.")
        H = DH_fun(ω, ω0, j, M, g)
        cop = np.sqrt(γ) * qt.tensor(qt.destroy(M),qt.qeye(int(2*j+1)))
        psi0 = psi0_fun(ω, ω0, j, M, g, β)
        rhoss = qt.steadystate(H,cop)
        overlap = rhoss.overlap(psi0)
        np.save(file_path, overlap)
    else:
        print(f"{file_path} already exists.")
    overlap = np.load(file_path)

    return overlap