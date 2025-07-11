from model import schedular
from model import peptide_model
from model import model_utils
import torch.nn as nn
import torch
import optuna
import time
import json
import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Optuna version: {optuna.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"Device: {DEVICE}")

# Use auxiliary loss during training
USE_AUXILIARY = False


def create_model(trial):
    """
    Hyperparameters search for the Fusion model.
    """
    activation_name = trial.suggest_categorical(
        "activation_name", ["ReLU", "LeakyReLU", "SiLU", "GELU"]
    )
    dim_linear = trial.suggest_categorical("dim_linear", [64, 128, 256, 512])
    dim_out = trial.suggest_categorical("dim_out", [16, 32, 64])
    # MLP
    MLP_num_mlp = trial.suggest_int("MLP_num_mlp", 1, 6, 1)
    MLP_dim_mlp = trial.suggest_categorical("MLP_dim_mlp", [64, 128, 256, 512])
    MLP_dim_linear = dim_linear
    MLP_activation_name = activation_name
    MLP_dropout_rate = trial.suggest_float(
        "MLP_dropout_rate", 0.0, 0.3, step=0.05
    )
    MLP_dim_out = dim_out
    # concat
    # Fusion_num_concat = trial.suggest_int("Fusion_num_concat", 1, 3, 1)
    # Fusion_concat_units = [dim_linear] * Fusion_num_concat

    model = peptide_model.MultiLayerPerceptron(
        device=DEVICE,
        use_auxiliary=USE_AUXILIARY,
        # MLP
        num_mlp=MLP_num_mlp,
        dim_mlp=MLP_dim_mlp,
        activation_name=MLP_activation_name,
        dropout_rate=MLP_dropout_rate,
        dim_linear=MLP_dim_linear,
        dim_out=MLP_dim_out,
        # Fusion
    )

    return model


def create_optimizer(trial, model):
    """
    Hyperparameters search for the optimizer.
    """
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", ["AdamW", "NAdam", "RAdam"]
    )
    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    )

    # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), weight_decay=weight_decay
        )
    # lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, *, foreach=None, differentiable=False
    elif optimizer_name == "NAdam":
        optimizer = torch.optim.NAdam(
            model.parameters(), weight_decay=weight_decay
        )
    # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, *, foreach=None, differentiable=False
    elif optimizer_name == "RAdam":
        optimizer = torch.optim.RAdam(
            model.parameters(), weight_decay=weight_decay
        )

    return optimizer


MODEL_TYPE = "MLP"
REPLICA_NUM = 60  # Augmentation times

EPOCH_NUM = 50
PATIENCE = 5  # Stop early when validation loss does not decrease for five consecutive epochs

CV = 3

gamma_layer = 0.05  # Weight of auxiliary layer loss
gamma_subout = 0.10  # Weight of auxiliary sub-model loss

# OPTIMIZE
# seed = 2024
# model_utils.set_seed(seed)


def objective(trial):
    time_start_trial = time.time()

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    loss_trial = 0

    for cv in range(CV):
        folder_path = "model/input/"
        dataset_train = model_utils.load_dataset(
            folder_path, MODEL_TYPE, REPLICA_NUM, f"Train_cv{cv}"
        )
        dataset_valid = model_utils.load_dataset(
            folder_path, MODEL_TYPE, REPLICA_NUM, f"Valid_cv{cv}"
        )

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=batch_size, shuffle=False
        )

        model = create_model(trial)
        model = nn.DataParallel(model)
        model.to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = create_optimizer(trial, model)
        # OPTIMIZE: Adjusting the learning rate
        init_lr = 0.0001
        scheduler = schedular.NoamLR(
            optimizer=optimizer,
            warmup_epochs=[0.2 * EPOCH_NUM],
            total_epochs=[EPOCH_NUM],
            steps_per_epoch=len(dataset_train) // batch_size,
            init_lr=[init_lr],
            max_lr=[init_lr * 10],
            final_lr=[init_lr / 10],
        )

        model_path = f"weight/{MODEL_TYPE}_optuna/{MODEL_TYPE}-{REPLICA_NUM}_t{
            trial.number
        }_cv{cv}.cpt"

        loss_train_list, loss_valid_list = model_utils.train_loop(
            model_path,
            DEVICE,
            PATIENCE,
            EPOCH_NUM,
            dataloader_train,
            dataloader_valid,
            model,
            criterion,
            optimizer,
            scheduler,
            verbose=True,
            use_auxiliary=USE_AUXILIARY,
            gamma_layer=gamma_layer,
            gamma_subout=gamma_subout,
        )
        # Save complete loss after early stopping
        if DEVICE == "cuda":
            checkpoint = torch.load(model_path, weights_only=False)
        else:
            checkpoint = torch.load(
                model_path, map_location=torch.device("cpu")
            )
        checkpoint["loss_train_list"] = loss_train_list
        checkpoint["loss_valid_list"] = loss_valid_list
        torch.save(checkpoint, model_path)

        loss_trial += min(loss_valid_list)

    time_end_trial = time.time()
    print(
        f"Execution time of trial {trial.number:03d}: {
            (time_end_trial - time_start_trial):.0f
        }"
    )
    print(
        "------------------------------------------------------------------------"
    )

    return loss_trial / CV


study = optuna.create_study(
    direction="minimize",
    study_name=f"{MODEL_TYPE}-{REPLICA_NUM}",
    load_if_exists=True,
    storage=f"sqlite:///weight/{MODEL_TYPE}_optuna/{MODEL_TYPE}-{
        REPLICA_NUM
    }.db",
)
study.optimize(objective, 10)
study.trials_dataframe().to_csv(
    f"weight/{MODEL_TYPE}_optuna/study_history_{MODEL_TYPE}-{REPLICA_NUM}.csv"
)
