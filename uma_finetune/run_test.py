import numpy as np
from sklearn.model_selection import train_test_split
from ase.io import write
import os

from ase import Atoms
from typing import List
from ase.calculators.singlepoint import SinglePointCalculator

from ase.io import read
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


base_path = "./"

#filesamples = [13000]#[2600,5200,7800,10400,13000] #2600,5200,7800

filenames = [base_path+"/inference_ckpt.pt"]#.format(sample) for sample in filesamples]
uma_predictors = []
uma_calculators = []
for filename in filenames:
    print(filename)
    uma_predictor = load_predict_unit(
        path=filename,
        device="cpu",
    )
    uma_predictors.append(uma_predictor)
    uma_calculator = FAIRChemCalculator(
        uma_predictors[-1],
        task_name="omol", # options: "omol", "omat", "odac", "oc20", "omc"
    )
    uma_calculators.append(uma_calculator)
    # break


#Find all filenames in ./train folder
train_filenames = [f for f in os.listdir("./train") if f.endswith('.traj')]
val_filenames = [f for f in os.listdir("./val") if f.endswith('.traj')]


import numpy as np
import os
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt

energy_train_mae_epoch = []
force_train_mae_epoch = []
energy_val_mae_epoch = []
force_val_mae_epoch = []

for uma_calculator in uma_calculators:
    print(f"Processing with calculator: {uma_calculator}")
    
    energy_train_mae = []
    force_train_mae = []
    energy_val_mae = []
    force_val_mae = []

    for train_filename in train_filenames:
        atoms = Trajectory(os.path.join("./train/", train_filename))[0]
        atoms.info = {"charge": 0, "spin": 1}

        # Reference
        E_ref = atoms.get_potential_energy() * 23.06054195
        F_ref = atoms.get_forces() * 23.06054195

        # Predicted
        atoms_copy = atoms.copy()
        atoms_copy.calc = uma_calculator
        E_pred = atoms_copy.get_potential_energy() * 23.06054195
        F_pred = atoms_copy.get_forces() * 23.06054195

        # Store per-structure MAEs
        energy_train_mae.append(np.abs(E_ref - E_pred))
        force_train_mae.append(np.mean(np.abs(F_ref - F_pred)))

    energy_train_mae_epoch.append(energy_train_mae)
    force_train_mae_epoch.append(force_train_mae)

    for val_filename in val_filenames:
        atoms = Trajectory(os.path.join("./val/", val_filename))[0]
        atoms.info = {"charge": 0, "spin": 1}

        E_ref = atoms.get_potential_energy() * 23.06054195
        F_ref = atoms.get_forces() * 23.06054195

        atoms_copy = atoms.copy()
        atoms_copy.calc = uma_calculator
        E_pred = atoms_copy.get_potential_energy() * 23.06054195
        F_pred = atoms_copy.get_forces() * 23.06054195

        energy_val_mae.append(np.abs(E_ref - E_pred))
        force_val_mae.append(np.mean(np.abs(F_ref - F_pred)))

    energy_val_mae_epoch.append(energy_val_mae)
    force_val_mae_epoch.append(force_val_mae)

# Convert to arrays
energy_train_mae_epoch = np.array(energy_train_mae_epoch)
force_train_mae_epoch = np.array(force_train_mae_epoch)
energy_val_mae_epoch = np.array(energy_val_mae_epoch)
force_val_mae_epoch = np.array(force_val_mae_epoch)

# Summary MAEs per epoch (mean over all structures)
mae_energy_train_epoch = np.mean(energy_train_mae_epoch, axis=1)
mae_force_train_epoch = np.mean(force_train_mae_epoch, axis=1)
mae_energy_val_epoch = np.mean(energy_val_mae_epoch, axis=1)
mae_force_val_epoch = np.mean(force_val_mae_epoch, axis=1)

print("MAE Energy (train) per epoch:", mae_energy_train_epoch)
print("MAE Forces (train) per epoch:", mae_force_train_epoch)
print("MAE Energy (val) per epoch:", mae_energy_val_epoch)
print("MAE Forces (val) per epoch:", mae_force_val_epoch)


import matplotlib.pyplot as plt
import numpy as np

# Choose the epoch index to plot
epoch_idx = 0  # Change this to plot another epoch

# Get MAE per structure for energy and force
energy_mae = energy_train_mae_epoch[epoch_idx]
force_mae = force_train_mae_epoch[epoch_idx]

# X-axis: structure index
structure_indices = np.arange(len(energy_mae))

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(structure_indices, energy_mae, marker='o', label='Energy MAE (kcal/mol)')
# plt.plot(structure_indices, force_mae, marker='s', label='Force MAE (kcal/mol/Å)')

plt.title(f"MAE per Structure - Epoch {epoch_idx}")
plt.xlabel("Structure Index")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("output_energy.png")

plt.plot(structure_indices, force_mae, marker='s', label='Force MAE (kcal/mol/Å)')
plt.title(f"MAE per Structure - Epoch {epoch_idx}")
plt.xlabel("Structure Index")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("outputforce.png")
