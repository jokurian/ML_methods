import os
import sys
import numpy as np
from ase import Atoms
from sgdml.train import GDMLTrain
from ase.calculators.calculator import Calculator, all_properties
from sgdml.intf.ase_calc import SGDMLCalculator
from pyscf.geomopt import as_pyscf_method, geometric_solver
from pyscf.geomopt import berny_solver
from pyscf import gto

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


class DeltaSGDMLUMACalculator(Calculator):
    #This calculator assumes that SGDML model was created with units kcal/mol(/A). Otherwise, a conversion factor has to be given      
    implemented_properties = ['energy', 'forces'] 

    def __init__(self, model_path_1, calculator_2):
        super().__init__()
        self.calculator_1 = SGDMLCalculator(model_path_1)
        self.calculator_2 = calculator_2

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)

        # Set properties to be calculated
        if properties is None:
            properties = self.implemented_properties

        # Perform calculations using both calculators
        self.calculator_1.calculate(atoms, properties, system_changes)
        self.calculator_2.calculate(atoms, properties, system_changes)

        # Combine results (sum energies and forces from both models)
        self.results = {}
        for prop in properties:
            if prop in self.calculator_1.results and prop in self.calculator_2.results:
                self.results[prop] = (self.calculator_1.results[prop] +
                                      self.calculator_2.results[prop])


base_path = "./"


os.system(f"rm -rf {base_path}/structures {base_path}/models")  #Removes old files
os.system(f"mkdir {base_path}/structures {base_path}/models")

ntrain = 10
sig = 28
lam = 5e-7

higher_data_file = f"afqmc_result.xyz"  # Assumes units kcal/mol and kcal/mol/A for energy and forces

#Converts data to sgdml type. Make sure data is in the correct unit, otherwise we will have to change the calculator accordingly
os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/{higher_data_file} --r_unit Ang --e_unit kcal/mol")  

#---------------- making datafiles ---------------#
# del gdml_train
higher_data = np.load(f"{base_path}/{higher_data_file[:-4]}.npz")
gdml_train = GDMLTrain()
picked_pts = gdml_train.draw_strat_sample(T=higher_data['E'],n= ntrain)        #picks ntrain samples according to their distribution of energy
print("Picked points for training:", picked_pts)

higher_data = np.load(f"{base_path}/{higher_data_file[:-4]}.npz")
write_extended_xyz_data(f'{base_path}/selected_{ntrain}.xyz', higher_data['E'][picked_pts], higher_data['R'][picked_pts],
            higher_data['F'][picked_pts], atomic_number_to_symbol(higher_data['z']))

os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/selected_{ntrain}.xyz --r_unit Ang --e_unit kcal/mol")
os.system(f"mv selected_{ntrain}.npz {base_path}")
#-------------------------------------------------#



#---------------- Training model ---------------#
del gdml_train

# Training the model
#np.random.seed(seed)
dataset= np.load(f'{base_path}/selected_{ntrain}.npz')
nvalid = 0
model_path = f"{base_path}/models/model_{ntrain}_{sig}_{lam}_{ntrain}.npz"  #Saves this to disk
gdml_train = GDMLTrain()
task = gdml_train.create_task(dataset, ntrain,\
        valid_dataset=dataset, n_valid=nvalid,\
        sig=sig, lam=lam)

try:
        model = gdml_train.train(task)
except Exception:
        sys.exit()
else:
        np.savez_compressed(model_path, **model)

del gdml_train

#------------------------------------------------#


def ase_energy_and_grad(mol):
    """
    Returns (energy, gradient) for PySCF geomopt interface
    Energy in Hartree, Gradient in Hartree/Bohr
    """
    coords = mol.atom_coords(unit='Angstrom')  # (N,3)
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    positions = mol.atom_coords() * 0.529177210903
    atoms = Atoms(symbols=symbols, positions=positions)

    atoms.set_positions(coords)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.calc = SGDMLCalculator(model_path)

    # Energy in Hartree
    energy_ha = atoms.get_potential_energy() / 27.211386

    # Forces in eV/Å → Gradient in Hartree/Bohr
    forces_ev = atoms.get_forces()
    gradient_ha_perA = -forces_ev / 27.211386
    gradient_ha_perBohr = gradient_ha_perA / 1.88973  # convert Å → Bohr

    return energy_ha, gradient_ha_perBohr


model_path = f"{base_path}/models/model_{ntrain}_{sig}_{lam}_{ntrain}.npz"
mol = gto.Mole()
mol.atom = "h2o_initial.xyz"
mol.spin = 0
mol.unit = "Angstrom"
mol.build()

geo_conv_params = { # These are the default settings
    'convergence_energy': 1e-6,  # Eh
    'convergence_grms': 3e-5,    # Eh/Bohr
    'convergence_gmax': 4.5e-5,  # Eh/Bohr
    'convergence_drms': 1.2e-3,  # Angstrom
    'convergence_dmax': 1.8e-3,  # Angstrom
}

ase_method = as_pyscf_method(mol, ase_energy_and_grad)

# ts_mol = berny_solver.optimize(
ts_mol = geometric_solver.optimize(
    ase_method,
    transition=False,    # TS optimization
    maxsteps=100,
    verbose=3,
    conv_params=geo_conv_params
)

print("Optimized TS geometry (Angstrom):")
print(ts_mol.tostring(format="xyz"))
print(ts_mol.tostring(format="zmat"))

with open(f"{base_path}/optimized_structure.xyz", "w") as f:
    f.write(ts_mol.tostring(format="xyz"))
