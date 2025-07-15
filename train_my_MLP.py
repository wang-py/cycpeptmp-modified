import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import RAdam
from MLP import PeptideDataset, MyMLP
from tqdm import tqdm

# Assumes you have this class already
# from your_code import PeptideDataset, SimpleMLP

# === Early Stopping Setup ===
best_val_loss = float("inf")
patience = 5
counter = 0
best_model_state = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

folder = "model/input/MLP/60/"
# List of your folds
fold_paths = [
    folder + "peptide_60_Train_cv0.npz",
    folder + "peptide_60_Train_cv1.npz",
    folder + "peptide_60_Train_cv2.npz",
]
val_losses = []


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch)


for fold_idx in range(3):
    best_val_loss = float("inf")
    print(f"\n🔁 Fold {fold_idx + 1}")

    # Load validation dataset
    val_dataset = PeptideDataset(fold_paths[fold_idx])

    # Load training datasets (the other two folds)
    train_indices = [i for i in range(3) if i != fold_idx]
    train_datasets = [PeptideDataset(fold_paths[i]) for i in train_indices]
    train_dataset = ConcatDataset(train_datasets)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, collate_fn=safe_collate
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Model and optimizer
    model = MyMLP(
        dim_input=16 + 2048,
        dim_linear=128,
        dim_out=1,
        activation_name="ReLU",
        MLP_num_mlp=1,
        MLP_dim_mlp=128,
        MLP_dropout_rate=0.25,
    ).to(device)

    optimizer = RAdam(model.parameters(), lr=1e-3, weight_decay=0.0005)
    criterion = nn.MSELoss()

    # === Training loop ===
    for epoch in tqdm(range(20)):
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # === Validation loop ===
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device).view(-1, 1)

                y_pred = model(x_val)
                loss = criterion(y_pred, y_val)
                total_val_loss += loss.item() * x_val.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {
                avg_val_loss:.4f
            }"
        )
        # === Early Stopping Check ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(
                    f"⏹️ Early stopping at epoch {epoch} (no improvement for {
                        patience
                    } epochs)"
                )
                break

    model.load_state_dict(best_model_state)
    val_losses.append(best_val_loss)
    torch.save(model.state_dict(), f"my_MLP_model/model_fold{fold_idx}.pt")


# Final result
print(f"\n📊 Average Validation Loss over 3 folds: {sum(val_losses) / 3:.4f}")
