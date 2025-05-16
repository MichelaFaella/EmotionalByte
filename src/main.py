from train import grid_search, TrainSDT




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
>>>>>>> 1cae55d (Grid search and model selection)
            "tensorboard": True,
            "lr": 0.0001,
            "dropout": 0.5
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
            "gamma_2": 1,
            "gamma_3": 1
        }

        fixed_params = {
            "n_epochs": 50,
>>>>>>> 1cae55d (Grid search and model selection)
            "tensorboard": True,
            "lr": 0.0001,
            "dropout": 0.5
        }

        TrainSDT(**best_config, **fixed_params, run_name="best_model_train", return_val_score=False)

<<<<<<< HEAD
    print("\n🚀 Testing best model on test set...")
=======
    print("\nTesting best model on test set...")
>>>>>>> 1cae55d (Grid search and model selection)
    TrainSDT(**best_config, **fixed_params, run_name="best_model_test", return_val_score=False)