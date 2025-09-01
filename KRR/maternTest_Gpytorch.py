import math
import os
import gpytorch
import numpy as np
import torch
from ase.io import read
import matplotlib.pyplot as plt


#atoms = read("md.xyz", index=":")
np.random.seed(5)
def getdij(coords):
    ##make the feature which is equal to d_ij = 1/|r_i - r_j| for i>j
    diffs = coords[:, None, :] - coords[None, :, :] 
    dist = torch.norm(diffs, dim=-1)  
    dist = dist + torch.eye(dist.shape[0]) * 1e-12 
    d_ij = 1.0 / dist   

    i, j = torch.tril_indices(row=coords.shape[0], col=coords.shape[0], offset=-1)
    d_ij = d_ij[i, j]
    return d_ij

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


# structures = parse_extended_xyz("afqmc_result.xyz")
# E = [x['energy'] for x in structures]
# geom = np.array([x['positions'] for x in structures])
# force = np.array([x['forces'] for x in structures])

atoms = read("md.xyz", index=":")
E = [a.get_potential_energy()*23 for a in atoms]
geom = np.array([a.positions for a in atoms])
force = np.array([a.get_forces() for a in atoms])

##read the file to get the energies and geometries
x_train = []
y_train = E
for i in range(len(E)):

    coords = geom[i]

    ##make the feature which is equal to d_ij = 1/|r_i - r_j| for i>j
    diffs = coords[:, None, :] - coords[None, :, :] 
    dist = np.linalg.norm(diffs, axis=-1)  
    with np.errstate(divide='ignore'):
        d_ij = 1.0 / dist   

    i, j = np.tril_indices(len(coords), k=-1)
    d_ij = d_ij[i, j]
    x_train.append(d_ij)

x_bkp = 1.*np.array(x_train)
y_bkp = 1.*np.array(y_train)

##do a random split into training, validation and testing
n_data = len(y_train)
n_train, n_val = 512, 128 #int(0.8*n_data)
n_test = n_data - n_train - n_val
indices = np.random.choice(np.arange(n_data), size=n_val+n_train, replace=False)
#indices = np.sort(indices)

mask = np.ones(n_data, dtype=bool)
mask[indices] = False
x_test = torch.from_numpy(np.array(x_train)[mask])
y_test = torch.from_numpy(np.array(y_train)[mask])

mask = np.zeros(n_data, dtype=bool)
mask[indices[n_train:]] = True
x_val = torch.from_numpy(np.array(x_train)[mask])
y_val = torch.from_numpy(np.array(y_train)[mask])

mask = np.zeros(n_data, dtype=bool)
mask[indices[:n_train]] = True
x_train = torch.from_numpy(np.array(x_train)[mask])
y_train = torch.from_numpy(np.array(y_train)[mask])


# We are using exact GP inference with a zero mean and RBF kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        ##use the same kernel as in sGDML
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)  # e.g. Matérn-5/2
        )
        self.covar_module.base_kernel.lengthscale = 20.

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Create likelihood first
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Now register constraint on its noise parameter
likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-10))
# likelihood.noise = 0.03  # initial guess for noise

model = ExactGPModel(x_train, y_train, likelihood)
# model.covar_module.base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(5))

##initialize the hyperparameters to what we used in sGDML
model.covar_module.base_kernel.lengthscale = 20.   # your choice
model.covar_module.outputscale = 1.
likelihood.noise = 1.e-8



model.eval()
likelihood.eval()

##evaluate the model on the validation data
error, trueE = [], []
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for i in range(y_test.shape[0]):
        pred = likelihood(model(x_test[i:i+1]))
        print("{0:15.3f} {1:15.3f} {2:5.3f}".format(y_test[i].item(), pred.mean.item(), pred.variance.item()))
        error.append(y_test[i].item() - pred.mean.item())
        trueE.append(y_test[i].item())

meanE = np.array(trueE).mean()
plt.plot(trueE-meanE,  np.array(error), "o")
plt.show()

print(np.array(error).mean(), np.sqrt(np.array(error).var()))
exit(0)

# ##one can obtain gradients as well
# x_val.required_grad = True
# gradError = []
# #with torch.no_grad(), gpytorch.settings.fast_pred_var():
# for I in indices:

#     coords = torch.from_numpy(geom[I])
#     coords.requires_grad=True

#     d_ij = getdij(coords)

#     pred = model(d_ij.reshape(1,-1))
#     #print("{0:15.3f} {1:15.3f} ".format(y_bkp[I].item(), pred.mean.item()))
#     #error.append(y_bkp[I].item() - pred.mean.item())
#     pred.mean.backward()

#     gradError.append(np.linalg.norm(force[I] - np.array(coords.grad)))

# plt.plot(np.array(gradError), "o")
# plt.show()

