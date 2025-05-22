from Train.Train import TrainSDT
from Train.Gridsearch import grid_search


if __name__ == '__main__':

    model_selection = False

    if model_selection:

        param_grid = {
            "temp": [1.0, 1.2],
            "gamma_1": [0.1, 0.3],
            "gamma_2": [0.1, 0.3],
            "gamma_3": [0.1, 0.3]
        }

        fixed_params = {
            "n_epochs": 200,
            "model_dimension": 32,
            "n-head": 8,
            "tensorboard": True,
            "lr": 0.0005,
            "dropout": 0.5,
            "weight_decay": 0.0001,
            "batch_size": 16
        }


        best_config, best_score, all_results = grid_search(param_grid, fixed_params)

        print(f"\n Best configuration: {best_config}")
        print(f"Best validation F1: {best_score}")

    else:
        best_config = {
            "temp": 1.0,
            "gamma_1": 1,
            "gamma_2": 0.5,
            "gamma_3": 0.5
        }

        fixed_params = {
            "n_epochs": 2,
            "model_dimension": 32,
            "n-head": 8,
            "tensorboard": True,
            "lr": 1e-3,
            "dropout": 0.0,
            "weight_decay": 0.03,
            "batch_size": 16,
            "modality": 'audio'
        }

    TrainSDT(**best_config, **fixed_params, run_name="best_model_test", return_val_score=False)