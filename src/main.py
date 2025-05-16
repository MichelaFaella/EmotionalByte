from train import grid_search, TrainSDT




if __name__ == '__main__':

    model_selection = True

    if model_selection:

        param_grid = {
            "temp": [0.5, 0.8, 1.0, 1.2, 1.4],
            "gamma_1": [0.1, 0.3, 0.5, 0.8, 1.0],
            "gamma_2": [0.1, 0.3, 0.5, 0.8, 1.0],
            "gamma_3": [0.1, 0.3, 0.5, 0.8, 1.0]
        }

        fixed_params = {
            "n_epochs": 70,
            "tensorboard": True,
            "lr": 0.0001,
            "dropout": 0.5
        }


        best_config, best_score, all_results = grid_search(param_grid, fixed_params)

        print(f"\n✅ Best configuration: {best_config}")
        print(f"🏆 Best validation F1: {best_score}")

    else:
        best_config = {
            "temp": 1.0,
            "gamma_1": 0.3,
            "gamma_2": 0.5,
            "gamma_3": 0.3
        }

        fixed_params = {
            "n_epochs": 2,
            "tensorboard": True,
            "lr": 0.0001,
            "dropout": 0.5
        }

        TrainSDT(**best_config, **fixed_params, run_name="best_model_train", return_val_score=False)

    print("\n🚀 Testing best model on test set...")
    TrainSDT(**best_config, **fixed_params, run_name="best_model_test", return_val_score=False)