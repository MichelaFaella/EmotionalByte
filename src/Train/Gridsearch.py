from itertools import product
from src.Train.Train import TrainSDT


def grid_search(param_grid, fixed_params=None):
    """
    param_grid: dict of hyperparameters to grid search (values must be lists)
    fixed_params: dict of additional fixed parameters passed to TrainSDT
    """
    if fixed_params is None:
        fixed_params = {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_score = -1
    best_config = None
    all_results = []
    num_config = len(list(product(*values)))

    for i, combo in enumerate(product(*values)):
        config = dict(zip(keys, combo))
        full_config = {**fixed_params, **config}

        run_name = "grid_" + "_".join([f"{k}{v}" for k, v in config.items()])
        print(f"\n Running config {i+1}/{num_config}: {config}")

        score = TrainSDT(**full_config, run_name=run_name, return_val_score=True)
        print(f"Validation F1-score: {score}")

        all_results.append((config, score))

        if score > best_score:
            best_score = score
            best_config = config

    print(f"\nBest Config: {best_config}")
    print(f"Best Validation F1: {best_score}")
    return best_config, best_score, all_results

