import torch
import torch.nn as nn
from torch.optim import RAdam
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PeptideDataset(Dataset):
    def __init__(self, npz_file):
        # Load the .npz file
        data = np.load(npz_file)

        # Extract arrays
        self.descriptors = torch.tensor(
            data["peptide_descriptor"], dtype=torch.float32)
        self.fps = torch.tensor(data["fps"], dtype=torch.float32)
        self.labels = torch.tensor(data["y"], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        desc = self.descriptors[idx].clone().detach().float()
        fps = self.fps[idx].clone().detach().float()

        x = torch.cat([desc, fps], dim=0)
        # Concatenate descriptor and fingerprint
        # x = torch.cat([self.descriptors[idx], self.fps[idx]], dim=0)
        # print(
        #     f"Index {idx} | desc shape: {desc.shape} | fps shape: {
        #         fps.shape
        #     } | x shape: {x.shape}"
        # )
        y = self.labels[idx]
        y = torch.tensor([y], dtype=torch.float32)
        return x, y


# Define the MLP model
class MyMLP(nn.Module):
    def __init__(
        self,
        dim_input=128,
        dim_linear=128,
        dim_out=16,
        activation_name="ReLU",
        MLP_num_mlp=1,
        MLP_dim_mlp=128,
        MLP_dropout_rate=0.25,
    ):
        super(MyMLP, self).__init__()

        # Activation function
        activation_dict = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "GELU": nn.GELU(),
            "ELU": nn.ELU(),
        }
        self.activation = activation_dict.get(activation_name, nn.ReLU())

        # Build MLP layers
        layers = []
        in_dim = dim_input
        for _ in range(MLP_num_mlp):
            layers.append(nn.Linear(in_dim, MLP_dim_mlp))
            layers.append(self.activation)
            layers.append(nn.Dropout(MLP_dropout_rate))
            in_dim = MLP_dim_mlp

        # Final output layer
        layers.append(nn.Linear(in_dim, dim_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# Path to the dataset
dataset_path = "model/input/MLP/60/peptide_60_Test.npz"

# Create dataset and loader
dataset = PeptideDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# Example usage
for batch_x, batch_y in dataloader:
    print(batch_x.shape)  # [batch_size, 16 + 2048]
    print(batch_y.shape)  # [batch_size]
    break


# Example usage
batch_size = 128
dim_input = 128  # Must match input feature dimension

model = MyMLP(
    dim_input=dim_input,
    dim_linear=128,
    dim_out=16,
    activation_name="ReLU",
    MLP_num_mlp=1,
    MLP_dim_mlp=128,
    MLP_dropout_rate=0.25,
)

# Define optimizer
optimizer = RAdam(model.parameters(), lr=1e-3, weight_decay=0.0005)

# Example dummy input
x = torch.randn(batch_size, dim_input)
output = model(x)
print(output.shape)  # Should print: torch.Size([128, 16])
