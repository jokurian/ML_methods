import os
import numpy as np

from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE 
from ase import io
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator, all_changes

import sys
import matplotlib.pyplot as plt
from pyscf import scf, gto, cc, grad

from ase.io import read
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from sgdml.train import GDMLTrain
from sgdml.intf.ase_calc import SGDMLCalculator
# to use an UMA model:
settings = InferenceSettings(
    tf32=True,
    activation_checkpointing=False,
    merge_mole=True,
    compile=False,
    wigner_cuda=False,
    external_graph_gen=False,
    internal_graph_gen_version=2,
)
from neb_utils import *


ntrain = 40
seed = 42
sig_diff = 23
lam_diff = 5e-8 
nvalid = 0
ncalcs = 1   #number of separate NEB calculations to perform to estimate the stochastic error


# to use an UMA model:
uma_predictor = load_predict_unit(
    path="/home/jokurian/projects/ML_umol/uma-s-1p1.pt",
    device="cpu",
    inference_settings=settings, 
)
uma_calculator = FAIRChemCalculator(
    uma_predictor,
    task_name="omol", # options: "omol", "omat", "odac", "oc20", "omc"
)


#-------------- Create ncalcs datasets with gaussian noise on it ------------------#
np.random.seed(seed)
lower_data_file = "uma_result"
higher_data_file = "afqmc_result"
base_path = "./"
os.system(f"rm -f {base_path}/{higher_data_file}_*.xyz {base_path}/{higher_data_file}_*.npz ")
os.system(f"mkdir {base_path}/structures")
for i in range(ncalcs):  
    structures = parse_extended_xyz(f"{base_path}/{higher_data_file}.xyz",error_file=f"{base_path}/error_arrays.npz",add_random_error=True)
    write_extended_xyz(structures, f"{base_path}/{higher_data_file}_{i}.xyz")
    os.system(f"python sgdml_from_xyz.py {base_path}/{higher_data_file}_{i}.xyz --r_unit Ang --e_unit kcal/mol")
os.system(f"mv {higher_data_file}_*.npz {base_path}/{higher_data_file}_*.xyz {base_path}/structures")

#----------------------------------------------------------------------------------#


#-------------- Make difference structures for delta model --------------------#
import os


os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/{lower_data_file}.xyz --r_unit Ang --e_unit kcal/mol")
os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/{higher_data_file}.xyz --r_unit Ang --e_unit kcal/mol")
os.system(f"mv {lower_data_file}.npz {higher_data_file}.npz {base_path}/structures/")

os.system("rm difference_*.npz umol_neb_traj_*.traj")

from sgdml.train import GDMLTrain
lower_data = np.load(f"{base_path}/structures/{lower_data_file}.npz")
#del gdml_train
gdml_train = GDMLTrain()
picked_pts = gdml_train.draw_strat_sample(T=lower_data['E'],n= ntrain)
print("Picked points:", picked_pts)

for n in range(ncalcs):
    higher_data_n = np.load(f"{base_path}/structures/{higher_data_file}_{n}.npz")
    lower_data = np.load(f"{base_path}/structures/{lower_data_file}.npz")
    write_extended_xyz_data(f'{base_path}/difference_selected_{ntrain}_{n}.xyz', higher_data_n['E'][picked_pts]-lower_data['E'][picked_pts], lower_data['R'][picked_pts],
            higher_data_n['F'][picked_pts] - lower_data['F'][picked_pts], atomic_number_to_symbol(lower_data['z']))
    os.system(f"python {base_path}/sgdml_from_xyz.py {base_path}/difference_selected_{ntrain}_{n}.xyz --r_unit Ang --e_unit kcal/mol")
    os.system(f"mv {base_path}/difference_selected_{ntrain}_{n}.npz {base_path}/difference_selected_{ntrain}_{n}.xyz {base_path}/structures/")

del gdml_train
#--------------------------------------------------------------------------------#


#--------------------------- Train ncalcs models --------------------------------#
#np.random.seed(seed)
os.system(f"mkdir {base_path}/models")
for n in range(ncalcs):
        dataset= np.load(f'{base_path}/structures/difference_selected_{ntrain}_{n}.npz')
        nvalid = 0
        model_path = f"{base_path}/models/model_diff_{ntrain}_{sig_diff}_{lam_diff}_{n}.npz"
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
#---------------------------------------------------------------------------------#


#------------------------- Run NEB -----------------------------------------------#
base_path = './'
result_file = f'{base_path}/output_delta_krr_umol_{ntrain}.txt'
os.system("rm neb_result*.xyz umol_neb_traj.traj neb_samples.log output_umol.txt ts_structures.xyz")
with open(result_file, 'w') as f:
    f.write('# E_TS\tE_R\tE_P\tdE\tE_DFT\tE_CCSD_T\n')

E_TS, E_R, E_P, dE, E_DFT, E_CCSD_T = [], [], [], [], [], []
restart, iter, initFile, finalFile, max_iter = False, 1, f"{base_path}/react_opt.xyz", f"{base_path}/prod_opt.xyz", 2


for n in range(ncalcs):
#    traj = Trajectory(f"umol_neb_traj_{n}.traj", 'w')
    diff_model_path = f"{base_path}/models/model_diff_{ntrain}_{sig_diff}_{lam_diff}_{n}.npz"
    try:
        if not restart:
            initial, final = initialize(initFile, finalFile)
            images = run_neb(initial, final, diff_model_path, uma_calculator, charge=0, spin=1, fmax=0.05)#,traj = traj)

        write(f"{base_path}neb_result.xyz", images)
        # Repeat NEB max_iter number of times by taking the highest point and optimizing around there
        images = refine_neb(images, diff_model_path, uma_calculator, iterations=max_iter, base_path=base_path, fmax=0.05)
        # Repeat NEB with climbing mode on
        images = final_neb_relax(images)
        write(f"{base_path}neb_result_final.xyz", images)

        es = extract_energies(f"{base_path}neb_result_final.xyz")

        initial.calc = DeltaSGDMLCalculator_krr_umol(diff_model_path, uma_calculator)
        initE = initial.get_potential_energy()[0]
        final.calc = DeltaSGDMLCalculator_krr_umol(diff_model_path, uma_calculator) 
        finalE = final.get_potential_energy()[0]
        
        e_ts = np.max(es) * 0.036749304951208
        e_r = initE * 0.03674930495
        e_p = finalE * 0.03674930495
        de = (np.max(es) - initE) * 23.060541945329
        
        E_TS.append(e_ts)
        E_R.append(e_r)
        E_P.append(e_p)
        dE.append(de)

        c2 = extract_xyz_frames(f"{base_path}neb_result_final.xyz")
        c2 = c2[np.argmax(es)]
        atomstring = c2
        e_dft, e_ccsd_t = compute_quantum_energies(atomstring)
        E_DFT.append(e_dft)
        E_CCSD_T.append(e_ccsd_t)

        write_results(result_file, e_ts, e_r, e_p, de, e_dft, e_ccsd_t)
        write_extended_xyz_noForce(atomstring, e_dft, e_ccsd_t, f"{base_path}ts_structures.xyz")
        print(f"E(TS): {e_ts}, E(R): {e_r}, E(P): {e_p}, dE: {de}")
    except Exception as e:
        print(f"Error in NEB calculation : {e}")
        E_TS.append(0)
        E_R.append(0)
        E_P.append(0)
        dE.append(0)
        E_DFT.append(0)
        E_CCSD_T.append(0)












