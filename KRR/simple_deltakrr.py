import os
import sys
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_properties
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from sgdml.train import GDMLTrain
from sgdml.intf.ase_calc import SGDMLCalculator
from typing import List
from ase.calculators.singlepoint import SinglePointCalculator

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


class DeltaSGDMLUMACalculator(Calculator):
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




settings = InferenceSettings(
    tf32=True,
    activation_checkpointing=False,
    merge_mole=True,
    compile=False,
    wigner_cuda=False,
    external_graph_gen=False,
    internal_graph_gen_version=2,
)
uma_predictor = load_predict_unit(
    path="/home/jokurian/projects/ML_umol/uma-s-1p1.pt",
    device="cpu",
    inference_settings=settings, 
)

uma_calculator = FAIRChemCalculator(
    uma_predictor,
    task_name="omol", # options: "omol", "omat", "odac", "oc20", "omc"
)


base_path = "./"

os.system(f"rm -rf {base_path}/structures {base_path}/models") 
os.system(f"cp sgdml_from_xyz.py {base_path}/")
os.system(f"mkdir {base_path}/structures {base_path}/models")

ntrain = 100
sig_diff = 23#17
lam_diff = 1e-8#1e-8

higher_data_file = f"afqmc_result.xyz"
lower_data_file = "uma_result.xyz"  #UMA predicted values for the same structures in afqmc_result.xyz: for delta model

os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/{higher_data_file} --r_unit Ang --e_unit kcal/mol")
os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/{lower_data_file} --r_unit Ang --e_unit kcal/mol")
os.system(f"mv {higher_data_file[:-4]}.npz {lower_data_file[:-4]}.npz {base_path}")



#---------------- making datafiles ---------------#
# del gdml_train
lower_data = np.load(f"{base_path}/{lower_data_file[:-4]}.npz")
gdml_train = GDMLTrain()
picked_pts = gdml_train.draw_strat_sample(T=lower_data['E'],n= ntrain)
print("Picked points for training:", picked_pts)

higher_data = np.load(f"{base_path}/{higher_data_file[:-4]}.npz")
write_extended_xyz_data(f'{base_path}/difference_selected_{ntrain}.xyz', higher_data['E'][picked_pts]-lower_data['E'][picked_pts], lower_data['R'][picked_pts],
            higher_data['F'][picked_pts] - lower_data['F'][picked_pts], atomic_number_to_symbol(lower_data['z']))

os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/difference_selected_{ntrain}.xyz --r_unit Ang --e_unit kcal/mol")
os.system(f"mv difference_selected_{ntrain}.npz {base_path}")
del gdml_train

#--------------------------------------------------#

#---------------- Training difference model ----------------#
#np.random.seed(seed)
dataset= np.load(f'{base_path}/difference_selected_{ntrain}.npz')
nvalid = 0
model_path = f"{base_path}/models/model_diff_{ntrain}_{sig_diff}_{lam_diff}_{ntrain}.npz"
gdml_train = GDMLTrain()
task = gdml_train.create_task(dataset, ntrain,\
        valid_dataset=dataset, n_valid=nvalid,\
        sig=sig_diff, lam=lam_diff)

try:
        model = gdml_train.train(task)
except Exception:
        sys.exit()
else:
        np.savez_compressed(model_path, **model)

del gdml_train

#------------------------------------------------------------#

#----------------- Predict energies ----------------#
structures = parse_extended_xyz(f"{base_path}/{higher_data_file}")
ase_atoms = convert_to_ase_atoms(structures)
model_path = f"{base_path}/models/model_diff_{ntrain}_{sig_diff}_{lam_diff}_{ntrain}.npz"
Eexact = []
Epred = []
for i, atoms in enumerate(ase_atoms):
    E_fromfile = atoms.info["energy"]
    Eexact.append(E_fromfile)
    atoms.info.update({"spin":1,"charge":0})
    atoms.calc = DeltaSGDMLUMACalculator(model_path, uma_calculator)
    energy = atoms.get_potential_energy()[0]*23.06054195
    Epred.append(energy)
    if(i>5): break

print("Eexact:", Eexact)
print("Epred:", Epred)
print("Diff:", np.array(Epred) - np.array(Eexact))

