from Train.Train import TrainSDT
from Train.Gridsearch import grid_search
from Util.SaveModel import *
from Util.Plot import plotLosses, plotTotalLoss, plotEval, confusionMatrix


def runs(model_selection, run_name, dirpath, data):
    if model_selection:

        param_grid = {
            "temp": [2.0, 2.5],
            "gamma_1": [1.0],
            "gamma_2": [1.0],
            "gamma_3": [1.0], # 0 if without self-distillation
            "lr": [0.0001, 0.0005],
            "dropout": [0.0001, 0.005],
            "weight_decay": [0.0001, 0.005],
            "model_dimension": [16],
        }

        fixed_params = {
            "n_epochs": 100,
            "n_head": 8,
            "tensorboard": True,
            "batch_size": 32,
            "modality": 'text', # multi, text, audio, text_sd, audio_sd
            "bios": False
        }

        best_config, best_score, all_results = grid_search(param_grid, fixed_params, dirpath, data)

        print(f"\n Best configuration: {best_config}")
        print(f"Best validation F1: {best_score}")

    else:
        best_config = {
            "temp": 0.7,
            "gamma_1": 1,
            "gamma_2": 1,
            "gamma_3": 1,
            "lr": 0.0005,
            "dropout": 0.01,
            "weight_decay": 0.01
        }
        fixed_params = {
            "n_epochs": 150,
            "model_dimension": 144,
            "n_head": 18,
            "tensorboard": True,
            "batch_size": 256,
            "modality": 'multi', # multi, text, audio, text_sd, audio_sd
            "bios": False
        }

    TrainSDT(**best_config, **fixed_params, run_name=run_name, return_val_score=False, dirpath=dirpath, data=data)

def plot_all_metrics(result_directory, run_name):
    logs_train, logs_eval, n_epochs = load_results(dirpath=result_directory, run_name=run_name)
    hyperparams = load_hyperparams(dirpath=result_directory, run_name=run_name)
    labels_preds = load_label_pred(dirpath=result_directory, run_name=run_name)
    #labels_preds = None

    plotLosses(
        logs=logs_train, epochs=n_epochs, save_dir=result_directory,
        save_path=run_name, hyperparams=hyperparams
    )
    plotEval(
        logs=logs_eval, epochs=n_epochs, phase="TEST", save_dir=result_directory,
        save_path=run_name, hyperparams=hyperparams
    )
    plotTotalLoss(
        logs_train=logs_train, logs_test=logs_eval, epochs=n_epochs,
        save_dir=result_directory, save_path=run_name, hyperparams=hyperparams
    )

    if labels_preds is not None:
        confusionMatrix(
            labels_preds['test']['true'],
            labels_preds['test']['pred'],
            dir=result_directory,
            save_path=run_name,
            title_suffix="(Test)"
        )
        confusionMatrix(
            labels_preds['train']['true'],
            labels_preds['train']['pred'],
            dir=result_directory,
            save_path=run_name,
            title_suffix="(Training)"
        )


if __name__ == '__main__':


    model_selection = False
    load_model = False
    run_name = ("Multimodal_2")

    results = {
        0 : "Emoberta_eGeMAPSv02",
        1 : "Emoberta",
        2 : "eGeMAPSv02"
    }

    result_directory = f"./results/{results[0]}"
    pickle = "./data/iemocap_multimodal_features_6_labels_emoberta-tae898_eGeMAPSv02.pkl"


    if load_model:
        plot_all_metrics(result_directory, run_name)
    else:
        runs(model_selection, run_name, dirpath=result_directory, data=pickle)
        plot_all_metrics(result_directory, run_name)
