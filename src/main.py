from src.Train.Train import TrainSDT
from src.Train.Gridsearch import grid_search


if __name__ == '__main__':

<<<<<<< HEAD
    model_selection = True
=======
    model_selection = False
>>>>>>> 1cae55d (Grid search and model selection)

    if model_selection:

        param_grid = {
<<<<<<< HEAD
            "temp": [0.5, 0.8, 1.0, 1.2, 1.4],
            "gamma_1": [0.1, 0.3, 0.5, 0.8, 1.0],
            "gamma_2": [0.1, 0.3, 0.5, 0.8, 1.0],
            "gamma_3": [0.1, 0.3, 0.5, 0.8, 1.0]
        }

        fixed_params = {
            "n_epochs": 70,
=======
            "temp": [1.0, 1.2],
            "gamma_1": [0.1, 0.3],
            "gamma_2": [0.1, 0.3],
            "gamma_3": [0.1, 0.3]
        }

        fixed_params = {
            "n_epochs": 200,
<<<<<<< HEAD
>>>>>>> 1cae55d (Grid search and model selection)
=======
            "model_dimension": 32,
            "n-head": 8,
>>>>>>> b93c7d6 (remove dataset)
            "tensorboard": True,
            "lr": 0.0005,
            "dropout": 0.5,
            "weight_decay": 0.0001,
            "batch_size": 16
        }


        best_config, best_score, all_results = grid_search(param_grid, fixed_params)

<<<<<<< HEAD
        print(f"\n✅ Best configuration: {best_config}")
        print(f"🏆 Best validation F1: {best_score}")
=======
        print(f"\n Best configuration: {best_config}")
        print(f"Best validation F1: {best_score}")
>>>>>>> 1cae55d (Grid search and model selection)

    else:
        best_config = {
            "temp": 1.0,
<<<<<<< HEAD
            "gamma_1": 0.3,
            "gamma_2": 0.5,
            "gamma_3": 0.3
        }

        fixed_params = {
            "n_epochs": 2,
=======
            "gamma_1": 1,
            "gamma_2": 0.5,
            "gamma_3": 0.5
        }

        fixed_params = {
<<<<<<< HEAD
            "n_epochs": 50,
>>>>>>> 1cae55d (Grid search and model selection)
=======
            "n_epochs": 2,
            "model_dimension": 32,
            "n-head": 8,
>>>>>>> b93c7d6 (remove dataset)
            "tensorboard": True,
            "lr": 1e-3,
            "dropout": 0.0,
            "weight_decay": 0.03,
            "batch_size": 16
        }

<<<<<<< HEAD
        TrainSDT(**best_config, **fixed_params, run_name="best_model_train", return_val_score=False)

<<<<<<< HEAD
    print("\n🚀 Testing best model on test set...")
=======
    print("\nTesting best model on test set...")
>>>>>>> 1cae55d (Grid search and model selection)
=======
>>>>>>> b93c7d6 (remove dataset)
    TrainSDT(**best_config, **fixed_params, run_name="best_model_test", return_val_score=False)