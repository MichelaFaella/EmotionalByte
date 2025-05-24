from Train.Train import TrainSDT
from Train.Gridsearch import grid_search
from Util.SaveModel import read_from_csv
from Util.Plot import plotLosses, plotTotalLoss, plotEval


def runs(model_selection, run_name, dirpath, data):
    if model_selection:

        param_grid = {
            "temp": [1.5, 2.0],
            "gamma_1": [1.0],
            "gamma_2": [1.0],
            "gamma_3": [1.0],
            "lr": [0.001, 0.005],
            "dropout": [0.001, 0.005],
            "weight_decay": [0.001, 0.005],
        }

        fixed_params = {
            "n_epochs": 2,
            "model_dimension": 32,
            "n-head": 8,
            "tensorboard": True,
            "batch_size": 16,
            "modality": 'multi'
        }

        best_config, best_score, all_results = grid_search(param_grid, fixed_params, dirpath, data)

        print(f"\n Best configuration: {best_config}")
        print(f"Best validation F1: {best_score}")

    else:
        best_config = {
            "temp": 3.0,
            "gamma_1": 1,
            "gamma_2": 1,
            "gamma_3": 1,
            "lr": 0.005,
            "dropout": 0.001,
            "weight_decay": 0.001,
        }
        fixed_params = {
            "n_epochs": 150,
            "model_dimension": 32,
            "n-head": 8,
            "tensorboard": True,
            "batch_size": 16,
            "modality": 'multi'
        }

    TrainSDT(**best_config, **fixed_params, run_name=run_name, return_val_score=False, dirpath=dirpath, data=data)

if __name__ == '__main__':


    model_selection = True
    load_model = False
    run_name = "Test2"

    results = {
        0 : "Emoberta_eGeMAPSv02",
        1 : "Emoberta_emobase",
        2 : "RobertaBase_eGeMAPSv02",
        3 : "RobertaBase_emobase"
    }

    datasets = {
        0: "./data/iemocap_multimodal_features_6_labels_emoberta-tae898_eGeMAPSv02.pkl",
        1: "./data/iemocap_multimodal_features_6_labels_emoberta-tae898_emobase.pkl",
        2: "./data/iemocap_multimodal_features_6_labels_roberta-base_eGeMAPSv02.pkl",
        3: "./data/iemocap_multimodal_features_6_labels_roberta-base_emobase.pkl"
    }

    result_directory = f"./results/{results[0]}/"
    pickle = datasets[0]

    if load_model:
        logs_train, logs_eval, n_epochs, hyperparams = read_from_csv(run_name)
        plotLosses(logs=logs_train, epochs=n_epochs, save_path=run_name, hyperparams=hyperparams)
        plotEval(logs=logs_eval, epochs=n_epochs, phase="TEST", save_path=run_name, hyperparams=hyperparams)
        plotTotalLoss(logs_train=logs_train, logs_test=logs_eval, epochs=n_epochs, save_path=run_name, hyperparams=hyperparams)
    else:
        runs(model_selection, run_name, dirpath=result_directory, data=pickle)



