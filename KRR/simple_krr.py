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

ntrain = 100
sig = 10
lam = 1e-12

higher_data_file = f"uccsd_t_result.xyz"  # Assumes units kcal/mol and kcal/mol/A for energy and forces

# Converts data to sgdml type. Make sure data is in the correct unit, otherwise we will have to change the calculator accordingly
os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/{higher_data_file} --r_unit Ang --e_unit kcal/mol")


#---------------- Training model ---------------#
#np.random.seed(seed)
dataset= np.load(f'{base_path}/{higher_data_file[:-4]}.npz')
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

#----------------- Predict energies ----------------#
structures = parse_extended_xyz(f"{base_path}/{higher_data_file}")
ase_atoms = convert_to_ase_atoms(structures)
model_path = f"{base_path}/models/model_{ntrain}_{sig}_{lam}_{ntrain}.npz"
for i, atoms in enumerate(ase_atoms):
    atoms.calc = SGDMLCalculator(model_path)
    energy = atoms.get_potential_energy()[0]  #in eV
    print(f"Energy of structure {i}: {energy} eV")
    if(i>5): break