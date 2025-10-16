from jax import config
config.update("jax_enable_x64", True)
import numpy as np
from jax import numpy as jnp
from jax import jit
from jax import jacfwd, vmap
import jax

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
    return kprime, ksecond

def Hxx_pair_explicit(xa, xb, sigma):
    diff = xa - xb
    d = np.linalg.norm(diff)

    kprime, ksecond = _kprime_ksecond(d, sigma)     
    invd = 1.0 / d
    c1 = ksecond - kprime * invd                    
    c2 = kprime * invd                              
    u  = diff * invd                                
    UUT = np.outer(u, u)                           
    Mdim = xa.shape[0]
    I = np.eye(Mdim, dtype=xa.dtype)
    H = -(c1 * UUT + c2 * I)
    return H

def hess_desc_kernel_explicit(D, sigma):
    D = np.asarray(D)
    B, M = D.shape
    H = np.zeros((B, B, M, M), dtype=D.dtype)

    diag_block = (5.0 / (3.0 * sigma * sigma)) * np.eye(M, dtype=D.dtype)
    for a in range(B):
        H[a, a] = diag_block
        xa = D[a]
        for b in range(a+1, B):
            xb = D[b]
            Hab = Hxx_pair_explicit(xa, xb, sigma)
            H[a, b] = Hab
            H[b, a] = Hab.T  

    return H

def assemble_force_kernel(H_xx_all, J):
    B, M = J.shape[0], J.shape[1]
    D = J.shape[2]

    K_blocks = np.einsum('ami,abmn,bnj->aibj', J, H_xx_all, J)   # (B, B, D, D)
    K_full = K_blocks.reshape(B * D, B * D)

    return K_full


# Get d desc / dR using autodiff. Still using autodiff because its easier. But can be changed
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

def hess_desc_cross_explicit(D_test, D_train, sigma, eps=1e-12):
    D_test  = np.asarray(D_test)
    D_train = np.asarray(D_train)
    T, M = D_test.shape
    B = D_train.shape[0]

    diff = D_test[:, None, :] - D_train[None, :, :]  # (T,B,M)
    d = np.linalg.norm(diff, axis=-1)                            # (T,B)

    mask = (d <= eps)                                            # (T,B)
    invd = np.where(mask, 0.0, 1.0 / d)                          # (T,B)
    u = diff * invd.reshape(T, B, 1)                             # (T,B,M)

    kprime, ksecond = _kprime_ksecond(d, sigma)                  # (T,B)
    c1 = ksecond - kprime * invd                                 # (T,B)
    c2 = kprime * invd                                           # (T,B)

    UUT = np.einsum('tbm,tbn->tbmn', u, u)                       # (T,B,M,M)
    I_M = np.eye(M, dtype=D_test.dtype)                           # (M,M)

    term1 = np.einsum('tb,tbmn->tbmn', c1, UUT)                  # (T,B,M,M)
    term2 = np.einsum('tb,ij->tbij',   c2, I_M)                  # (T,B,M,M)
    H = -(term1 + term2)                                         # (T,B,M,M)

    diag_block = (5.0 / (3.0 * sigma * sigma)) * I_M             # (M,M)
    H = np.where(mask.reshape(T, B, 1, 1),
                 diag_block.reshape(1, 1, M, M),
                 H)

    return H

dataset = parse_extended_xyz("md_traj_new.xyz", add_random_error=False)
Es = np.array([d["energy"] for d in dataset])
Fs = np.array([d["forces"] for d in dataset])
Rs = np.array([d["positions"] for d in dataset])
species = dataset[0]["species"]

natoms = len(species)
ndim = natoms * 3
ndata = len(Es)

#############################################
seed = 52
np.random.seed(seed)
ntrain = 100
ntest = 100
sig = 10
lam = 1e-10
#############################################

train_indx = np.random.choice(ndata, ntrain, replace=False)
test_indx = np.random.choice(list(set(range(ndata)) - set(train_indx)), ntest, replace=False)

# import pdb;pdb.set_trace()
R_train = Rs[train_indx].reshape(ntrain, -1)
F_train = Fs[train_indx].reshape(ntrain, -1)
E_train = Es[train_indx]
R_test = Rs[test_indx].reshape(ntest, -1)
F_test = Fs[test_indx].reshape(ntest, -1)
E_test = Es[test_indx]

R_train_jnp = jnp.array(R_train)
R_test_jnp = jnp.array(R_test)
F_train_jnp = jnp.array(F_train)
F_test_jnp = jnp.array(F_test)
E_train_jnp = jnp.array(E_train)
E_test_jnp = jnp.array(E_test)


d = invdist_descriptor_sgdml(R_train) #descriptors
d = d.block_until_ready()
d_np = np.asarray(jax.device_get(d), dtype=np.float64)

d_d = jac_desc_batch(R_train_jnp)  #d desc / dR
d_d = d_d.block_until_ready()
d_d_np = np.asarray(jax.device_get(d_d), dtype=np.float64)

#Generating kernel
H_xx_all_explicit = hess_desc_kernel_explicit(d_np, sig)
K_final_explicit = assemble_force_kernel(H_xx_all_explicit, d_d_np) #Converting from descriptor to R 
y = F_train_jnp.flatten()
y_std = np.std(y)
y = y / y_std  #This is how sgdml is training
alphas_explicit = train(K_final_explicit, lam, -y)  
# print(alphas_explicit)

D_test = invdist_descriptor_sgdml(R_test) #descriptors
D_test = D_test.block_until_ready()
D_test = np.asarray(jax.device_get(D_test), dtype=np.float64)
D_train = invdist_descriptor_sgdml(R_train) #descriptors
D_train = D_train.block_until_ready()
D_train = np.asarray(jax.device_get(D_train), dtype=np.float64)

J_test = jac_desc_batch(R_test_jnp) #d(desc_train)/dR
J_test = J_test.block_until_ready()
J_test = np.asarray(jax.device_get(J_test), dtype=np.float64)


# H_cross = hess_desc_cross_explicit(D_test_np[:5,:], D_train_np, sig)
H_cross = hess_desc_cross_explicit(D_test, D_train, sig)
K_blocks = np.einsum('tmi,tbmn,bnj->tibj', J_test, H_cross, d_d_np)   # (T,B,D,D)
K_fF = K_blocks.reshape(D_test.shape[0], J_test.shape[2], -1)  # (T,D,B*D)

F_pred = np.einsum('tij,j->ti', K_fF, alphas_explicit) * y_std


rmse = np.sqrt(np.mean((F_pred - F_test)**2))
mae = np.mean(np.abs(F_pred - F_test))
print("My KRR result:")
print("MAE (force) on test set: ", mae, " [kcal/mol/Ang]")
print("RMSE (force) on test set: ", rmse, " [kcal/mol/Ang]")


################################Test with sgdml#################################
import os
import sys
import numpy as np
from ase import Atoms
from sgdml.train import GDMLTrain
from typing import List
from ase.calculators.calculator import Calculator, all_properties
from sgdml.intf.ase_calc import SGDMLCalculator
from pyscf import gto
from ase.calculators.singlepoint import SinglePointCalculator

seed = 52
np.random.seed(seed)


def atomic_number_to_symbol(atomic_numbers):
    """
    Converts a list of atomic numbers to atomic symbols for elements up to atomic number 20.
    If an atomic number is greater than 20, an error message is printed.

    Args:
    atomic_numbers (list): List of atomic numbers (integers).

    Returns:
    list: List of atomic symbols corresponding to the atomic numbers.
    """
    atomic_symbols = {
        1: 'H',  2: 'He',  3: 'Li',  4: 'Be',  5: 'B',  6: 'C',  7: 'N',  8: 'O',  9: 'F', 10: 'Ne',
       11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca'
    }

    symbols = []
    for number in atomic_numbers:
        if number in atomic_symbols:
            symbols.append(atomic_symbols[number])
        else:
            raise ValueError(f"Error: Atomic number {number} is not supported (only up to 20).")

    return symbols

def write_extended_xyz_data(filename, energies, xyz_coords, forces, atom_symbols):
    """
    Writes an extended XYZ file with energies, atomic coordinates, and forces.

    Args:
    filename (str): Output filename for the extended XYZ file.
    energies (list): List of total energies for each configuration (length = number of configurations).
    xyz_coords (list of lists): List of atomic coordinates for each configuration.
                               Each element is a list of lists (one list per atom: [[x, y, z], ...]).
    forces (list of lists): List of atomic forces for each configuration.
                            Each element is a list of lists (one list per atom: [[fx, fy, fz], ...]).
    atom_symbols (list): List of atomic symbols for each atom (e.g., ['H', 'O', 'O']).
    """

    with open(filename, 'w') as f:
        for i, energy in enumerate(energies):
            num_atoms = len(xyz_coords[i])
            f.write(f"{num_atoms}\n")
            f.write(f"{energy}\n")
            for j in range(num_atoms):
                atom = atom_symbols[j]
                x, y, z = xyz_coords[i][j]
                fx, fy, fz = forces[i][j]
                f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f} {fx:.6f} {fy:.6f} {fz:.6f}\n")

def convert_to_ase_atoms(structures: List[dict]) -> List[Atoms]:
    """
    Convert a list of dictionaries into ASE Atoms objects.

    Args:
        structures (List[dict]): List of dictionaries with keys 'energy', 'species', 'positions', 'forces'.

    Returns:
        List[Atoms]: List of ASE Atoms objects.
    """
    atoms_list = []
    for s in structures:
        atoms = Atoms(symbols=s["species"], positions=s["positions"])
        if "forces" in s and s["forces"] is not None:
            atoms.set_array("forces", s["forces"])
        if "energy" in s and s["energy"] is not None:
            atoms.info["energy"] = s["energy"]
        calc = SinglePointCalculator(atoms, energy=s["energy"], forces=s["forces"])
        atoms.calc = calc
        atoms_list.append(atoms)

    return atoms_list

base_path = "./"


os.system(f"rm -rf {base_path}/structures {base_path}/models *.npz")  #Removes old files
os.system(f"mkdir {base_path}/structures {base_path}/models")

higher_data_file = f"md_traj_new.xyz"  # Assumes units kcal/mol and kcal/mol/A for energy and forces

os.system(f"rm -rf {base_path}/structures {base_path}/models/*.npz")  #Removes old files
# Converts data to sgdml type. Make sure data is in the correct unit, otherwise we will have to change the calculator accordingly
os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/{higher_data_file} --r_unit Ang --e_unit kcal/mol")

dataset = np.load(f'{base_path}/{higher_data_file[:-4]}.npz', allow_pickle=True)
ndata = dataset["E"].shape[0]

# train_indx = np.random.choice(ndata, ntrain, replace=False)
# test_indx = np.random.choice(list(set(range(ndata)) - set(train_indx)), ntest, replace=False)

write_extended_xyz_data(f'{base_path}/train_{ntrain}.xyz', dataset['E'][train_indx], dataset['R'][train_indx],
            dataset['F'][train_indx], atomic_number_to_symbol(dataset['z']))

os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/train_{ntrain}.xyz --r_unit Ang --e_unit kcal/mol")

write_extended_xyz_data(f'{base_path}/test_{ntest}.xyz', dataset['E'][test_indx], dataset['R'][test_indx],
            dataset['F'][test_indx], atomic_number_to_symbol(dataset['z']))

os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/test_{ntest}.xyz --r_unit Ang --e_unit kcal/mol")



#---------------- Training model ---------------#
#np.random.seed(seed)

dataset= np.load(f'{base_path}/train_{ntrain}.npz') #np.load(f'{base_path}/{higher_data_file[:-4]}.npz')
nvalid = 0
model_path = f"{base_path}/models/model_{ntrain}_{sig}_{lam}_{ntrain}.npz"  #Saves this to disk
gdml_train = GDMLTrain()
task = gdml_train.create_task(dataset, ntrain-nvalid,\
        valid_dataset=dataset, n_valid=nvalid,\
        sig=sig, lam=lam,use_sym=False,use_E_cstr=False)

try:
        model = gdml_train.train(task)
except Exception:
        sys.exit()
else:
        np.savez_compressed(model_path, **model)

# del gdml_train

#------------------------------------------------#

#----------------- Predict energies ----------------#
structures = parse_extended_xyz(f"{base_path}/test_{ntest}.xyz", add_random_error=False)
ase_atoms = convert_to_ase_atoms(structures)
model_path = f"{base_path}/models/model_{ntrain}_{sig}_{lam}_{ntrain}.npz"

Eexact = []
Epred_sgdml = []
Fpred_sgdml = []
Fexact = []
for i, atoms in enumerate(ase_atoms):
    E_fromfile = atoms.info["energy"]
    F_fromfile = atoms.get_array("forces")
    Eexact.append(E_fromfile)
    Fexact.append(F_fromfile)
    atoms.info.update({"spin":1,"charge":0})
    atoms.calc = SGDMLCalculator(model_path) 
    energy = atoms.get_potential_energy()[0]*23.06054195
    forces = atoms.get_forces()*23.06054195
    Epred_sgdml.append(energy)
    Fpred_sgdml.append(forces)
    # if(i%10==0): print(i)

# print("Eexact:", Eexact)
# print("Epred:", Epred_sgdml)
# print("Diff:", np.array(Epred_sgdml) - np.array(Eexact))

Fpred_sgdml = np.array(Fpred_sgdml).reshape(ntest,-1)
Fexact = np.array(Fexact).reshape(ntest,-1)
rmse = np.sqrt(np.mean((Fpred_sgdml - Fexact)**2))
mae = np.mean(np.abs(Fpred_sgdml - Fexact))
print("SGDML results:")
print("MAE (force) on test set: ", mae, " [kcal/mol/Ang]")
print("RMSE (force) on test set: ", rmse, " [kcal/mol/Ang]")

alphas_sgdml = model['alphas_F']

#There is a minus sign difference in the definition of alphas
# print("MAE of alphas between my KRR and SGDML: ", np.mean(np.abs(-alphas_explicit - alphas_sgdml)))
# print(alphas_sgdml)
# print(alphas_explicit)

n_train, n_atoms = task['R_train'].shape[:2]
from sgdml.utils.desc import Desc
from functools import partial
desc = Desc(
    n_atoms,
    max_processes=1,
)
R = task['R_train'].reshape(n_train, -1)
R_desc, R_d_desc = desc.from_R(
    R,
    lat_and_inv=None,
    callback=None,
)
tril_perms = np.array([Desc.perm(p) for p in task['perms']])
n_perms = task['perms'].shape[0]
dim_d = desc.dim

perm_offsets = np.arange(n_perms)[:, None] * dim_d
tril_perms_lin = (tril_perms + perm_offsets).flatten('F')

#Negative to match sign convention of SGDML
K = -gdml_train._assemble_kernel_mat(R_desc, R_d_desc,tril_perms_lin,task['sig'],desc,use_E_cstr=False,callback=None)

# def train(K_train,lamda,F_train):
#     n_train = K_train.shape[0]#F_train.shape[0]
#     #K_train = K[:n_train, :n_train]
#     alpha = np.linalg.solve(K_train + lamda * np.eye(n_train), F_train)#, rcond=-1)[0]#np.linalg.solve(K_train + lamda * np.eye(n_train), F_train)
#     return alpha


# y = task['F_train'].flatten()
# E_train = task['E_train']
# E_train_mean = np.mean(E_train)

# # y = np.hstack((y, -E_train + E_train_mean))
# y_std = np.std(y)
# y = y / y_std
# alpha = train(K,task['lam'],-y)

print("MAE of Kernel between my KRR and SGDML: ",np.mean(np.abs(K - K_final_explicit)))
print("First 25 Alphas my KRR:", -alphas_explicit[:25]) #There is a negative sign difference in the definition of alphas
print("First 25 Alphas SGDML:", model['alphas_F'][:25])









