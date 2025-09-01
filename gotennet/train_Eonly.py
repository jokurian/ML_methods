import torch
from torch_geometric.loader import DataLoader
from gotennet import GotenNet
from typing import List, Tuple
from os.path import join

from gotennet import utils
log = utils.get_logger(__name__)
from lightning import Trainer
from lightning.pytorch.loggers import Logger
from ase.io import read, write

import numpy as np
import gotennet
from gotennet.models.representation.gotennet import GotenNetWrapper
from gotennet.models.components.outputs import Atomwise
import pytorch_lightning as pl
from torch import nn
from torch.autograd import grad
import matplotlib.pyplot as plt

config_dir = "/Users/sandeepsharma/Documents/ML_stuff/AFQMC_grad_fit/gotennet/configs"

from torch_geometric.data import Data
def atomToData(atom, idx, mean = 0.):
    data = Data(
        z = torch.tensor(atom.numbers, dtype=torch.long),
        pos=torch.tensor(atom.positions, dtype=torch.float),
        edge_index=torch.tensor([[i,j] for i in range(len(atom)) for j in range(len(atom))]).T,
        y=torch.tensor([atom.get_potential_energy()-mean], dtype=torch.float),
        idx = idx,
    )
    return data

from torch_geometric.data import InMemoryDataset
class ASEXYZDataset(InMemoryDataset):
    def __init__(self, root, xyz_file, transform=None, pre_transform=None):
        self.xyz_file = xyz_file
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.xyz_file]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Not needed since xyz_file is already local
        pass

    def process(self):
        atoms_list = read(self.raw_paths[0], index=":")
        mean = np.array([a.get_potential_energy() for a in atoms_list]).mean()
        data_list = [atomToData(atoms, i, mean) for i, atoms in enumerate(atoms_list)]

        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    trainData = ASEXYZDataset(root="./data", xyz_file="md.xyz")
    train_size = int(len(trainData) * 0.8)
    val_size  = len(trainData) - train_size

    train_size = 512
    val_size = 128
    test_size = len(trainData) - train_size - val_size

    print(f"test_size: {train_size}, val_size: {val_size}")
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(trainData, [train_size, val_size, test_size],  generator=generator)

    train_loader = DataLoader(train_ds, batch_size = 32, shuffle=True, num_workers=1)
    val_loader   = DataLoader(val_ds, batch_size = 32, shuffle=True, num_workers=1)
    test_loader  = DataLoader(test_ds, batch_size = 32, shuffle=False, num_workers=1)
    # train_loader = DataLoader(train_ds, batch_size = 16, shuffle=False, num_workers=1)
    # val_loader   = DataLoader(val_ds, batch_size = 16, shuffle=False, num_workers=1)

    class Goten_Model(pl.LightningModule):
        def __init__(self, lr=0.0001, lr_decay = 0.8, lr_patience=5,
                     weight_decay = 0.01):
            super().__init__()
            self.lr = lr
            self.lr_decay = lr_decay
            self.lr_patience = lr_patience
            self.weight_decay = weight_decay

            self.save_hyperparameters()
            self.model = GotenNetWrapper(
                n_atom_basis = 128,
                n_interactions = 8,
                n_rbf = 32,
                radial_basis = "expnorm",
                activation = torch.nn.functional.silu,
                attn_dropout = 0.,
                cutoff_fn=gotennet.models.components.layers.CosineCutoff(cutoff=5.)
            )
            self.outputModel =  Atomwise(
                n_in=self.model.hidden_dim,
                activation=torch.nn.functional.silu,
            )


        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            batch.to("cpu")
            self.model.to("cpu")
            self.outputModel.to("cpu")
            batch.representation, batch.vector_representation = self.model.forward(batch)
            output = self.outputModel.forward(batch)

            loss =  nn.functional.mse_loss(output['y'][:,0], batch.y) #torch.sum((batch.y - output['y'])**2)

            self.log("train_loss", loss, prog_bar=True)

            return loss

        def validation_step(self, batch, batch_idx):
            batch.to("cpu")
            self.model.to("cpu")
            self.outputModel.to("cpu")
            batch.representation, batch.vector_representation = self.model.forward(batch)
            output = self.outputModel.forward(batch)

            loss =  nn.functional.mse_loss(output['y'][:,0], batch.y) #torch.sum((batch.y - output['y'])**2)

            self.log("val_loss", loss, prog_bar=True)

        def prediction(self, batch):
            batch.to("cpu")
            self.model.to("cpu")
            self.outputModel.to("cpu")
            batch.representation, batch.vector_representation = self.model.forward(batch)
            output = self.outputModel.forward(batch)
            return output['y'][:,0]
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())
            return torch.optim.SGD(self.parameters())


    model : Goten_Model = Goten_Model()

    if False:
        ##the checkpoint file is typically in lightning_logs/version_x/checkpoints/epoch=xx-step=xxx.ckpt
        ckpt_path = "Trained_EplusF.ckpt"
        #ckpt_path = "/Users/sandeepsharma/Documents/ML_stuff/AFQMC_grad_fit/gotennet/lightning_logs/version_20/checkpoints/epoch=49-step=6850.ckpt"
        model = Goten_Model.load_from_checkpoint(ckpt_path)

    if True:
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            filename="epoch-{epoch}",   # {epoch} placeholder
            save_top_k=-1,              # keep ALL checkpoints (no deletion)
            every_n_epochs=50           # save every 50 epochs
        )

        trainer = pl.Trainer(max_epochs=500, accelerator="auto", 
            num_sanity_val_steps = 0, gradient_clip_val = 5., callbacks=[checkpoint_callback])
        trainer.fit(model, train_loader, val_loader)
    

    model.eval()
    error, trueE = [], []
    with torch.no_grad():
        for batch in test_loader:
            output = model.prediction(batch)
            error += list((output - batch.y).detach().numpy())
            trueE += list(batch.y.detach().numpy())

    error = np.array(error).flatten()
    plt.plot(np.array(trueE).flatten()*23, error*23, 'o')
    plt.show()
    print(np.array(error*23).mean(), np.sqrt(np.array(error*23).var()))


if __name__ == "__main__":
    main()
