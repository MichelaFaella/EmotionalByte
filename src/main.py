from train import grid_search, TrainSDT




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
            "tensorboard": True,
            "lr": 0.0001,
            "dropout": 0.5
        }


        best_config, best_score, all_results = grid_search(param_grid, fixed_params)

        print(f"\n Best configuration: {best_config}")
        print(f"Best validation F1: {best_score}")

    else:
        best_config = {
            "temp": 1.0,
            "gamma_1": 1,
            "gamma_2": 1,
            "gamma_3": 1
        }

        fixed_params = {
            "n_epochs": 50,
            "tensorboard": True,
            "lr": 0.0001,
            "dropout": 0.5
        }

        TrainSDT(**best_config, **fixed_params, run_name="best_model_train", return_val_score=False)

    print("\nTesting best model on test set...")
    TrainSDT(**best_config, **fixed_params, run_name="best_model_test", return_val_score=False)