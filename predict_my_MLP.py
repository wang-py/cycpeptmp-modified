from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
from MLP import MyMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# Your model definition
test_npz = "model/input/MLP/60/peptide_60_RRCK.npz"
# 2. Load data
test_data = np.load(test_npz)
desc = torch.tensor(test_data["peptide_descriptor"], dtype=torch.float32)
fps = torch.tensor(test_data["fps"], dtype=torch.float32)
x_test = fps  # torch.cat([desc, fps], dim=1)

test_loader = DataLoader(TensorDataset(x_test), batch_size=128)

model_preds = []

for fold_idx in range(3):
    model = MyMLP(
        dim_input=2048,
        dim_linear=128,
        dim_out=1,
        activation_name="ReLU",
        MLP_num_mlp=1,
        MLP_dim_mlp=128,
        MLP_dropout_rate=0.25,
    ).to(device)

    model.load_state_dict(torch.load(
        f"my_MLP_model/model_fps_fold_{fold_idx}.pt"))
    model.eval()

    preds = []

    with torch.no_grad():
        for batch in test_loader:
            y_pred = model(batch[0].to(device))
            preds.append(y_pred)

    model_preds.append(torch.cat(preds, dim=0))

# Average predictions across folds
y_preds = torch.stack(model_preds, dim=0).mean(
    dim=0).cpu().numpy()  # shape [N, 1]

y_true = test_data["y"]
mse = mean_squared_error(y_true, y_preds)
r2 = r2_score(y_true, y_preds)

print(f"Test MSE: {mse:.4f}, R²: {r2:.4f}")
