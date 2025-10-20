from jax import config
config.update("jax_enable_x64", True)
import numpy as np
from jax import numpy as jnp
from jax import jit
import jax
from sgdml.train import GDMLTrain

#B : number of training samples
#T : number of test samples
#N : number of atoms
#M : number of descriptor entries = N*(N-1)/2
#3N: 3*number of atoms (for forces)


def parse_extended_xyz(
    filename,
    add_random_error=False,
    error_value=0.63,
    energy_convert=1.00,
    force_convert=1.00,
):
    """Parse the extended XYZ file to extract energy, coordinates, and forces with errors from an external file."""
    structures = []

    with open(filename, "r") as file:
        lines = file.readlines()
    i = 0
    structure_index = 0

    while i < len(lines):
        num_atoms = int(lines[i].strip())
        energy_line = lines[i + 1].strip()

        # Parse energy
        energy = float(energy_line) * energy_convert

        species = []
        positions = []
        forces = []

        for j in range(i + 2, i + 2 + num_atoms):
            atom_line = lines[j].split()
            species.append(atom_line[0])
            positions.append(
                [float(atom_line[1]), float(atom_line[2]), float(atom_line[3])]
            )
            forces.append(
                [
                    float(atom_line[4]) * force_convert,
                    float(atom_line[5]) * force_convert,
                    float(atom_line[6]) * force_convert,
                ]
            )

        forces = np.array(forces)

        if add_random_error:
            energy += np.random.normal(0, error_value)
            force_error = np.random.normal(0, error_value, forces.shape)
            forces += force_error

        structures.append(
            {
                "energy": energy,
                "species": species,
                "positions": positions,
                "forces": forces,
            }
        )

        # Move to the next structure in the file
        i += num_atoms + 2
        structure_index += 1
    return structures

@jit
def invdist_descriptor_sgdml(R_flat, eps=1e-12):
    B, threeN = R_flat.shape
    N = threeN // 3
    R = R_flat.reshape(B, N, 3)
    diffs = R[:, :, None, :] - R[:, None, :, :]
    dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + eps)
    i, j = jnp.tril_indices(N, k=-1)
    r_ij = dists[:, i, j]
    return 1.0 / r_ij  # (B, M), with M = N*(N-1)/2


def _kprime_ksecond(d, sigma):
    a = np.sqrt(5.0) / sigma
    expterm = np.exp(-a * d)
    kprime  = expterm * (-(a*a/3.0) * d * (1.0 + a*d))
    ksecond = expterm * (-(a*a/3.0) - (a*a*a/3.0)*d + (a**4/3.0)*d*d)
    return kprime, ksecond  # (B,B) or (T,B)

def assemble_force_kernel(H_xx_all, J):
    B, M = J.shape[0], J.shape[1]
    D = J.shape[2]          #3N
    K_blocks = np.einsum('ami,abmn,bnj->aibj', J, H_xx_all, J)
    K_full = K_blocks.reshape(B * D, B * D)  #B:ntrain
    return K_full


# Get d desc / dR using autodiff
from jax import jacfwd, vmap, jit

@jit
def jac_desc_single(r_flat):
    f = lambda r: invdist_descriptor_sgdml(r[None, :])[0]
    return jacfwd(f)(r_flat)  # (M, 3N)

@jit
def jac_desc_batch(R):
    return vmap(jac_desc_single)(R)

def train(K_train,lamda,F_train):
    n_train = K_train.shape[0]#F_train.shape[0]
    #K_train = K[:n_train, :n_train]
    alpha = -np.linalg.solve(K_train + lamda * np.eye(n_train), F_train)#, rcond=-1)[0]#np.linalg.solve(K_train + lamda * np.eye(n_train), F_train)
    return alpha

def hess_desc_cross_explicit(D_left, D_right, sigma, eps=1e-14):
    #test left, train right
    D_left  = np.asarray(D_left)
    D_right = np.asarray(D_right)
    T, M = D_left.shape
    B = D_right.shape[0]

    diff = D_left[:, None, :] - D_right[None, :, :]  
    d = np.linalg.norm(diff, axis=-1)                            

    mask = (d <= eps)                                            
    invd = np.where(mask, 0.0, 1.0 / d)                          
    u = diff * invd.reshape(T, B, 1)                             

    kprime, ksecond = _kprime_ksecond(d, sigma)                 
    c1 = ksecond - kprime * invd                                
    c2 = kprime * invd                                           

    UUT = np.einsum('tbm,tbn->tbmn', u, u)                       
    I_M = np.eye(M, dtype=D_left.dtype)                          

    term1 = np.einsum('tb,tbmn->tbmn', c1, UUT)                  
    term2 = np.einsum('tb,ij->tbij',   c2, I_M)                  
    H = -(term1 + term2)                                        

    diag_block = (5.0 / (3.0 * sigma * sigma)) * I_M            
    H = np.where(mask.reshape(T, B, 1, 1),
                 diag_block.reshape(1, 1, M, M),
                 H)

    return H

def Pq_from_perm(p):
    p = np.asarray(p)
    N = p.size
    P3 = np.zeros((3*N, 3*N))
    I3 = np.eye(3)
    for i in range(N):
        j = p[i]
        P3[3*i:3*i+3, 3*j:3*j+3] = I3
    return P3

def assemble_force_kernel_sym_train(R_train_jnp, D_left, J_left, sigma, perms_set):
    B, threeN = R_train_jnp.shape
    N  = threeN // 3
    N3 = threeN

    R_xyz = R_train_jnp.reshape(B, N, 3)
    K_blocks = np.zeros((B, B, N3, N3))

    for p in perms_set:
        p = np.asarray(p, dtype=int)                        
        R_perm_flat = R_xyz[:, p, :].reshape(B, N3)          

        D_right = np.asarray(jax.device_get(invdist_descriptor_sgdml(R_perm_flat)))
        J_right = np.asarray(jax.device_get(jac_desc_batch(R_perm_flat)))

        H = hess_desc_cross_explicit(D_left, D_right, sigma) 

        K_tmp = np.einsum('ami,abmn,bnj->abij', J_left, H, J_right) 

        P3 = Pq_from_perm(p)
        K_blocks += np.einsum('abij,jk->abik', K_tmp, P3)

    return K_blocks.transpose(0, 2, 1, 3).reshape(B * N3, B * N3)

def predict_forces_sym(R_test_jnp, D_test, J_test,
                                 R_train_jnp, sigma, alphas_explicit,
                                 y_std, perms_set):
    T, threeN = R_test_jnp.shape
    B         = R_train_jnp.shape[0]
    N         = threeN // 3
    N3        = threeN

    R_train_xyz = R_train_jnp.reshape(B, N, 3)
    F_sum = np.zeros((T, N3), dtype=D_test.dtype)

    for p in perms_set:
        p = np.asarray(p, dtype=int)
        R_perm_flat = R_train_xyz[:, p, :].reshape(B, N3)

        D_right = np.asarray(jax.device_get(invdist_descriptor_sgdml(R_perm_flat)))
        J_right = np.asarray(jax.device_get(jac_desc_batch(R_perm_flat)))
        H = hess_desc_cross_explicit(D_test, D_right, sigma)  

        K_tmp = np.einsum('tmi,tbmn,bnj->tbij', J_test, H, J_right) 

        P3 = Pq_from_perm(p)  
        K_tmp = np.einsum('tbij,jk->tbik', K_tmp, P3)  

        K_fF = K_tmp.transpose(0, 2, 1, 3).reshape(T, N3, B * N3)
        F_sum += np.einsum('tij,j->ti', K_fF, alphas_explicit)

    F_pred = F_sum * y_std
    return F_pred

def predict_energy_no_sym(D_test, D_train, J_train, sigma, alphas_explicit, y_std,c):
    ntest = D_test.shape[0]
    ntrain = D_train.shape[0]
    d = D_test[:, None, :] - D_train[None, :, :]  
    d_norm = np.linalg.norm(d, axis=-1)            
    kprime, _ = _kprime_ksecond(d_norm, sigma)   
    u = d / d_norm[:, :, None]                  
    u[~np.isfinite(u)] = 0.0

    k2 = np.einsum("ut,utd,tdn->utn", kprime, u, J_train)  
    k2 = k2.reshape(ntest,ntrain*J_train.shape[2])  
    E_pred = np.einsum("un,n->u", k2, alphas_explicit)  * y_std
    E_pred += c

    return E_pred

def predict_energy_sym(D_left, R_train_jnp, sig, alphas_explicit, y_std, c, perms):
    ntrain = R_train_jnp.shape[0]
    ntest  = D_left.shape[0]
    ndim = R_train_jnp.shape[1]
    natoms = ndim // 3
    Epred_sym = np.zeros((D_left.shape[0],))
    # print("ntrain:", ntrain, "ntest:", ntest, "ndim:", ndim, "natoms:", natoms)
    for p in perms:
        p = np.asarray(p, dtype=int)                        
        R_perm_flat = R_train_jnp.reshape(ntrain, natoms, 3)[:, p, :].reshape(ntrain, ndim)

        D_right = np.asarray(jax.device_get(invdist_descriptor_sgdml(R_perm_flat)))
        J_right = np.asarray(jax.device_get(jac_desc_batch(R_perm_flat)))
        diff = D_left[:, None, :] - D_right[None, :, :]  
        d = np.linalg.norm(diff, axis=-1)                           
        kprime = _kprime_ksecond(d, sig)[0]  
        u = diff / d[:, :, None]                    
        u[~np.isfinite(u)] = 0.0
        K_tmp = np.einsum("ut,utd,tdn->utn", kprime, u, J_right)  
        pq = Pq_from_perm(p)  
        K_tmp = np.einsum("utn,nk->utk", K_tmp, pq) 
        K_tmp = K_tmp.reshape(ntest,ntrain*J_right.shape[2])  
        Epred_sym += np.einsum("un,n->u", K_tmp, alphas_explicit) 

    Epred_sym = Epred_sym * y_std + c   
    return Epred_sym 

def get_c_nosym(E_train, D_train, J_train, sig, alphas_explicit, y_std):
    E_pred_train = predict_energy_no_sym(D_train, D_train, J_train, sig, alphas_explicit, y_std, 0.0)
    c = np.mean(E_train - E_pred_train)
    return c

def get_c_sym(E_train, D_train, R_train_jnp, sig, alphas_explicit, y_std, perms):
    E_pred_train = predict_energy_sym(D_train, R_train_jnp, sig, alphas_explicit, y_std, 0.0, perms)
    c = np.mean(E_train - E_pred_train)
    return c


filename = "md_traj_new.xyz"
# filename = "uccsd_t_result.xyz"
dataset = parse_extended_xyz(filename, add_random_error=False)
Es = np.array([d["energy"] for d in dataset])
Fs = np.array([d["forces"] for d in dataset])
Rs = np.array([d["positions"] for d in dataset])
species = dataset[0]["species"]

# from ase.io import read
# filename = "md.xyz"
# atoms = read(filename, index=":")
# Es = np.array([a.get_potential_energy()* 23.0605419453293 for a in atoms])
# Rs = np.array([a.positions for a in atoms])
# Fs = np.array([a.get_forces()* 23.0605419453293 for a in atoms])
# species = atoms[0].get_chemical_symbols()


natoms = len(species)
ndim = natoms * 3
ndata = len(Es)

#############################################
seed = 52
np.random.seed(seed)
ntrain = 100
ntest = 80
sig = 10
lam = 1e-10
sym = True
#############################################
# gdml_train = GDMLTrain()
# train_indx = gdml_train.draw_strat_sample(Es, ntrain)
# del gdml_train
train_indx = np.random.choice(ndata, ntrain, replace=False)
test_indx = np.random.choice(list(set(range(ndata)) - set(train_indx)), ntest, replace=False)

R_train = Rs[train_indx].reshape(ntrain, -1)
F_train = Fs[train_indx].reshape(ntrain, -1)
E_train = Es[train_indx]
R_test = Rs[test_indx].reshape(ntest, -1)
F_test = Fs[test_indx].reshape(ntest, -1)
E_test = Es[test_indx]

import jax.numpy as jnp
R_train_jnp = jnp.array(R_train)
R_test_jnp = jnp.array(R_test)
F_train_jnp = jnp.array(F_train)
F_test_jnp = jnp.array(F_test)
E_train_jnp = jnp.array(E_train)
E_test_jnp = jnp.array(E_test)

D_train = invdist_descriptor_sgdml(R_train_jnp)
D_train = np.asarray(jax.device_get(D_train), dtype=np.float64)
D_d_train = jac_desc_batch(R_train_jnp)
D_d_train = np.asarray(jax.device_get(D_d_train), dtype=np.float64)
D_test = invdist_descriptor_sgdml(R_test_jnp)
D_test = np.asarray(jax.device_get(D_test), dtype=np.float64)
J_train = jac_desc_batch(R_train_jnp)
J_train = np.asarray(jax.device_get(J_train), dtype=np.float64)
J_test = jac_desc_batch(R_test_jnp)
J_test = np.asarray(jax.device_get(J_test), dtype=np.float64)


if(not sym):
    H_xx_all_explicit = hess_desc_cross_explicit(D_train, D_train, sig) #hess_desc_kernel_explicit(d_np, sig)
    print(H_xx_all_explicit.shape)
    K_final_explicit = assemble_force_kernel(H_xx_all_explicit, D_d_train)
    print(K_final_explicit.shape)
    y = F_train_jnp.flatten()
    y_std = np.std(y)
    y = y / y_std
    alphas_explicit = train(K_final_explicit, lam, -y)  

    H_cross = hess_desc_cross_explicit(D_test, D_train, sig)

    K_blocks = np.einsum('tmi,tbmn,bnj->tibj', J_test, H_cross, J_train)   
    K_fF = K_blocks.reshape(D_test.shape[0], J_test.shape[2], -1)  
    F_pred = np.einsum('tij,j->ti', K_fF, alphas_explicit) * y_std

    rmse = np.sqrt(np.mean((F_pred - F_test)**2))
    mae = np.mean(np.abs(F_pred - F_test))
    print("Force MAE (explicit):  ", mae)
    print("Force RMSE (explicit): ", rmse)

    c = get_c_nosym(E_train, D_train, J_train, sig, alphas_explicit, y_std)
    E_pred = predict_energy_no_sym(D_test, D_train, J_train, sig, alphas_explicit, y_std, c)
    rmse_e = np.sqrt(np.mean((E_pred - E_test)**2))
    mae_e = np.mean(np.abs(E_pred - E_test))
    print("Energy MAE (explicit):  ", mae_e)
    print("Energy RMSE (explicit): ", rmse_e)

if(sym):
    perms = np.array([[0, 1, 2, 3, 4, 5],
       [0, 1, 3, 2, 4, 5]])
    # perms = perms[0:1]  #Doing this is equivalent to no symmetrization
    K_sym = assemble_force_kernel_sym_train(R_train_jnp, D_train, J_train, sig, perms)
    print(K_sym.shape)
    y = F_train_jnp.flatten()
    y_std = np.std(y)
    y = y / y_std
    alphas_explicit = train(K_sym, lam, -y)
    
    F_pred_sym = predict_forces_sym(R_test_jnp, D_test, J_test,
                                 R_train_jnp, sig, alphas_explicit,
                                 y_std, perms)
    
    rmse_sym = np.sqrt(np.mean((F_pred_sym - F_test)**2))
    mae_sym = np.mean(np.abs(F_pred_sym - F_test))
    print("Force MAE (sym explicit):  ", mae_sym)
    print("Force RMSE (sym explicit): ", rmse_sym)

    c = get_c_sym(E_train, D_train, R_train_jnp, sig, alphas_explicit, y_std, perms)
    Epred_sym = predict_energy_sym(D_test, R_train_jnp, sig, alphas_explicit, y_std, c, perms)
    rmse_e = np.sqrt(np.mean((Epred_sym - E_test)**2))
    mae_e = np.mean(np.abs(Epred_sym - E_test))
    print("Energy MAE (sym explicit):  ", mae_e)
    print("Energy RMSE (sym explicit): ", rmse_e)