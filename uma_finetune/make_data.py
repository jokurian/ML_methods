import numpy as np
from sklearn.model_selection import train_test_split
from ase.io import write
import os
import argparse
from ase import Atoms
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

    # --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Split XYZ dataset into training and validation sets.")
parser.add_argument("--ntrain", type=int, required=True, help="Number of training structures.")
parser.add_argument("--add_random_error", action="store_true", help="Add random error to energies.")
parser.add_argument("--error_value", type=float, default=0.0, help="Value of the random error to add.")
parser.add_argument("--input_xyz", type=str, required=True, help="Path to the input XYZ file.")

args = parser.parse_args()

# Constants
energy_convert = 0.0433641153
force_convert = 0.0433641153

# Read and parse
structures_parsed = parse_extended_xyz(
    args.input_xyz,
    add_random_error=args.add_random_error,
    error_value=args.error_value,
    energy_convert=energy_convert,
    force_convert=force_convert,
)

structures_ase = convert_to_ase_atoms(structures_parsed)

structures = [(atoms, i) for i, atoms in enumerate(structures_ase)]

ntrain = args.ntrain
train_ratio = ntrain / len(structures)
train_dir = "./train/"
val_dir = "./val/"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

train_structures, val_structures = train_test_split(
    structures, train_size=train_ratio, random_state=42
)

print(f"Saving {len(train_structures)} training structures...")
for atoms, original_idx in train_structures:
    filename = os.path.join(train_dir, f"structure_{original_idx:04d}.traj")
    write(filename, atoms)

print(f"Saving {len(val_structures)} validation structures...")
for atoms, original_idx in val_structures:
    filename = os.path.join(val_dir, f"structure_{original_idx:04d}.traj")
    write(filename, atoms)



