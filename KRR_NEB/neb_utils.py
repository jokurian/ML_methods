import numpy as np

from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE 
from ase import io
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator, all_changes

from sgdml.train import GDMLTrain
from sgdml.intf.ase_calc import SGDMLCalculator
import os
import sys
import matplotlib.pyplot as plt
from pyscf import scf, gto, cc, grad

from ase.io import read
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from ase.calculators.calculator import Calculator, all_properties
from sgdml.intf.ase_calc import SGDMLCalculator

class DeltaSGDMLCalculator_krr_umol(Calculator):
    implemented_properties = ['energy', 'forces']  # This allows the calculator to return common properties like energy, forces, etc.

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


def parse_extended_xyz(filename, error_file='error_arrays.npz', add_random_error=False):                 
    """Parse the extended XYZ file to extract energy, coordinates, and forces with errors from an external file."""
    structures = []

    # Load the error arrays from the npz file
    error_data = np.load(error_file)
    energy_error = error_data["E_error"]
    force_errors = error_data['errors']  # Assuming the key for errors is 'errors'


    with open(filename, 'r') as file:
        lines = file.readlines()
    i = 0
    structure_index = 0

    while i < len(lines):
        num_atoms = int(lines[i].strip())
        energy_line = lines[i + 1].strip()

        # Parse energy
        energy = float(energy_line)

        species = []
        positions = []
        forces = []

        for j in range(i + 2, i + 2 + num_atoms):
            atom_line = lines[j].split()
            species.append(atom_line[0])
            positions.append([float(atom_line[1]), float(atom_line[2]), float(atom_line[3])])
            forces.append([float(atom_line[4]), float(atom_line[5]), float(atom_line[6])])

        forces = np.array(forces)

        if add_random_error:
            energy += np.random.normal(0, energy_error[structure_index]*627.509) #Convert to kcal/mol
            current_force_error = force_errors[structure_index]*627.509/0.529177  # Convert from Ha/Bohr to kcal/mol/Angstrom
            forces += np.random.normal(0, current_force_error)

        structures.append({
            'energy': energy,
            'species': species,
            'positions': positions,
            'forces': forces,
        })

        # Move to the next structure in the file
        i += num_atoms + 2
        structure_index += 1
    return structures

def write_extended_xyz(structures, filename):
    """Write the extracted data into a new extended XYZ file."""
    with open(filename, 'w') as f:
        for structure in structures:
            num_atoms = len(structure['species'])
            f.write(f"{num_atoms}\n")
            f.write(f"Energy={structure['energy']} Properties=species:S:1:pos:R:3:forces:R:3\n")

            for species, pos, force in zip(structure['species'], structure['positions'], structure['forces']):
                pos_str = ' '.join(f"{x:.4f}" for x in pos)
                force_str = ' '.join(f"{x:.4f}" for x in force)
                f.write(f"{species} {pos_str} {force_str}\n")


def extract_xyz_frames(filename):
    frames = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        frame_lines = []
        for line in lines:
            if line.strip().isdigit():
                if frame_lines:
                    frames.append("\n".join(frame_lines))
                    frame_lines = []
            elif line.startswith('Properties'):
                continue
            else:
                frame_lines.append(line.strip())
        if frame_lines:
            frames.append("\n".join(frame_lines))
    return frames

def make_neb(leftImage, rightImage, nimages=10):
    images = [leftImage]
    images += [leftImage.copy() for _ in range(nimages)]
    images += [rightImage]
    # for image in images:
    #     image.set_calculator(CustomCalculator(custom_energy_force_function))
    neb = NEB(images,remove_rotation_and_translation=True,dynamic_relaxation=True)
    neb.interpolate(method='idpp')
    return neb

def interpolate(leftImage, rightImage, nimages=4):
    images = [leftImage]
    images += [leftImage.copy() for _ in range(nimages)]
    images += [rightImage]

    neb = NEB(images,remove_rotation_and_translation=True,dynamic_relaxation=True)
    neb.interpolate(method='idpp')
    return neb.images

def make_neb_ts(leftImage,tsimage, rightImage, nimages=10):
    im1 = interpolate(leftImage,tsimage,nimages//2)
    im2 = interpolate(tsimage,rightImage,nimages//2)
    images = im1 + im2[1:]
    # for image in images:
    #     image.set_calculator(CustomCalculator(custom_energy_force_function))
    neb = NEB(images,remove_rotation_and_translation=True,dynamic_relaxation=True)
    return neb


def initialize(initFile, finalFile, nimages=10,calculator=None):
    initial = read(initFile)  # Ethanol molecule
    #initial.set_calculator(CustomCalculator(custom_energy_force_function))
    final = read(finalFile)
    #final.set_calculator(CustomCalculator(custom_energy_force_function))
    return initial, final

def extract_energies(file_name):
    energies = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'energy=' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith('energy='):
                        energy = float(part.split('=')[1])
                        energies.append(energy)
                        break
    return energies

def extract_coordinates(data):
    # Split the input string into lines
    lines = data.splitlines()

    # Initialize an empty list to store coordinates
    coordinates = []

def write_extended_xyz_noForce(atomstring, energy_dft,energy_cc, output_file, append=True):
    atom_lines = atomstring.strip().splitlines()
    num_atoms = len(atom_lines)
    comment = f"{energy_dft:.6f} {energy_cc:.6f}"
    #import pdb;pdb.set_trace()
    # Open the file (append or overwrite mode)
    mode = 'a' if append else 'w'
    with open(output_file, mode) as f:
        f.write(f"{num_atoms}\n")        # Number of atoms
        f.write(f"{comment}\n")          # Comment with energy
        f.write(f"{atomstring}\n")       # Atom coordinates



import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.neb import NEB
from ase.optimize import BFGS, FIRE
from pyscf import gto, scf, cc


def run_neb(initial, final,delta_model_path, uma_calculator, nimages=10, fmax=0.05, steps=100, charge=0, spin=1,traj=None):
    neb = make_neb(initial, final, nimages=nimages)
    for image in neb.images:
        image.calc = DeltaSGDMLCalculator_krr_umol(delta_model_path, uma_calculator)
        #image.calc = FAIRChemCalculator(uma_predictor, task_name="omol")
        image.info = {"charge": charge, "spin": spin}
    optimizer = BFGS(neb)#, trajectory=traj, append_trajectory=True)
    optimizer.run(fmax=fmax, steps=steps)
    return neb.images

def refine_neb(images,delta_model_path, uma_calculator, iterations=5, nimages=10, fmax=0.05, steps=100, base_path='./', charge=0, spin=1,traj = None):
    for i in range(1, iterations):
        es_i = [im.get_total_energy() for im in images]
        maxI = np.argmax(es_i)
        neb = make_neb_ts(images[maxI - 1], images[maxI], images[maxI + 1], nimages=nimages)
        for image in neb.images:
            image.calc = DeltaSGDMLCalculator_krr_umol(delta_model_path, uma_calculator)
            image.info = {"charge": charge, "spin": spin}
        optimizer = BFGS(neb)#,trajectory=traj, append_trajectory=True)
        optimizer.run(fmax=fmax, steps=steps)
        images = neb.images
        write(f"{base_path}neb_result{i+1}.xyz", images)
    return images

def final_neb_relax(images, fmax=0.05, steps=100,traj = None):
    neb2 = NEB(images, climb=True,remove_rotation_and_translation=True,dynamic_relaxation=True)
    optimizer2 = FIRE(neb2)#, trajectory=traj, append_trajectory=True)
    optimizer2.run(fmax=fmax, steps=steps)
    return neb2.images

def plot_energies(es, base_path='./'):
    xs = np.arange(len(es))
    plt.plot(xs, es)
    plt.savefig(f"{base_path}neb.png")

def compute_quantum_energies(atomstring):
    mol = gto.M(atom=atomstring, basis='ccpvdz', unit='angstrom', verbose=3)
    mol.build()
    mf_dft = scf.UKS(mol)
    mf_dft.xc = "b3lyp"
    mf_dft.kernel()
    mf = scf.UHF(mol)
    mf.kernel()
    mycc = cc.UCCSD(mf)
    mycc.kernel()
    ccsd_t = mycc.ccsd_t()
    return mf_dft.e_tot, mycc.e_tot + ccsd_t

def write_results(result_file, e_ts, e_r, e_p, de, e_dft, e_ccsd_t):
    with open(result_file, 'a') as f:
        f.write(f"{e_ts:.8f}\t{e_r:.8f}\t{e_p:.8f}\t{de:.8f}\t{e_dft:.8f}\t{e_ccsd_t:.8f}\n")




                
