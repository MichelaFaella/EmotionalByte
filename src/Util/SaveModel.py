import csv
import os
import torch


def save_hyperparams(dirpath, run_name, **kwargs):
    csv_dir = os.path.join(dirpath, run_name)
    hyperparams_path = os.path.join(csv_dir, "hyperparams.csv")
    os.makedirs(csv_dir, exist_ok=True)
    write_header = not os.path.exists(hyperparams_path)

    model_dimension = kwargs.get("Model dimension", 32)
    dropout = kwargs.get("Dropout", 0.01)
    lr = kwargs.get("Lr", 0.001)
    weight_decay = kwargs.get("Weight Decay", 0.0)
    batch_size = kwargs.get("Batch Size", 16)
    n_head = kwargs.get("n_head", 8)
    n_epochs = kwargs.get("n_epochs", 70)
    temp = kwargs.get("Temp", 1.0)
    gamma_1 = kwargs.get("Gamma1", 0.1)
    gamma_2 = kwargs.get("Gamma2", 0.1)
    gamma_3 = kwargs.get("Gamma3", 0.1)
    modality = kwargs.get("Modality", "multi")
    bios = kwargs.get("Bios", False)


    with open(hyperparams_path, mode='a', newline='') as csvfile:
        writer_hp = csv.writer(csvfile)
        if write_header:
            writer_hp.writerow([
                "run_name", "model_dim", "dropout", "lr", "weight_decay", "batch_size",
                "n_head", "n_epochs", "temp", "gamma_1", "gamma_2", "gamma_3", "modality", "bios"
            ])
        writer_hp.writerow([
            run_name, model_dimension, dropout, lr, weight_decay, batch_size,
            n_head, n_epochs, temp, gamma_1, gamma_2, gamma_3, modality, bios
        ])


def save_results(dirpath, run_name, epoch, losses, train_loss, train_acc, train_fscore, eval_loss, eval_acc, eval_fscore):
    csv_dir = os.path.join(dirpath, run_name)
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{run_name}_log.csv")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as csv_file:
        writer_csv = csv.writer(csv_file)

        if write_header:
            header = [
                "epoch",
                "train_loss", "train_acc", "train_fscore",
                "eval_loss", "eval_acc", "eval_fscore",
                "loss_task", "loss_ce_t", "loss_ce_a", "loss_kl_t", "loss_kl_a", "total_loss"
            ]
            writer_csv.writerow(header)

        # Estrai le singole loss (alcune potrebbero non essere presenti a seconda della modalità)
        def get_loss_value(key):
            return losses[key].item() if key in losses else 0.0

        writer_csv.writerow([
            epoch + 1,
            train_loss,
            train_acc,
            train_fscore,
            eval_loss,
            eval_acc,
            eval_fscore,
            get_loss_value("loss_task"),
            get_loss_value("loss_ce_t"),
            get_loss_value("loss_ce_a"),
            get_loss_value("loss_kl_t"),
            get_loss_value("loss_kl_a"),
            get_loss_value("loss")
        ])

def read_from_csv(dirpath, run_name):
    log_path = os.path.join(dirpath, run_name, f"{run_name}_log.csv")
    hyperparams_path = os.path.join(dirpath, run_name, "hyperparams.csv")

    logs_train = {
        'loss_task': [],
        'loss_ce_t': [],
        'loss_ce_a': [],
        'loss_kl_t': [],
        'loss_kl_a': [],
        'loss': []
    }

    logs_eval = {
        'acc': [],
        'fscore': [],
        'loss': []
    }

    n_epochs = 0

    # --- Caricamento log ---
    with open(log_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            n_epochs += 1
            # Train losses
            logs_train['loss_task'].append(torch.tensor(float(row['loss_task'])))
            logs_train['loss_ce_t'].append(torch.tensor(float(row['loss_ce_t'])))
            logs_train['loss_ce_a'].append(torch.tensor(float(row['loss_ce_a'])))
            logs_train['loss_kl_t'].append(torch.tensor(float(row['loss_kl_t'])))
            logs_train['loss_kl_a'].append(torch.tensor(float(row['loss_kl_a'])))
            logs_train['loss'].append(torch.tensor(float(row['total_loss'])))

            # Eval metrics
            logs_eval['acc'].append(float(row['eval_acc']))
            logs_eval['fscore'].append(float(row['eval_fscore']))
            logs_eval['loss'].append(float(row['eval_loss']))

    # --- Caricamento hyperparams ---
    selected_keys = [
        "dropout", "lr", "weight_decay", "model_dimension", "batch_size", "modality"
    ]

    hyperparams = None
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['run_name'] == run_name:
                    hyperparams = {k: row[k] for k in selected_keys if k in row}
                    break

    return logs_train, logs_eval, n_epochs, hyperparams

