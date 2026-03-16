import json
import logging
import os
from run_experiment import run_experiment

RESULTS_FILE = "results/experiments.json"
LOGGER = logging.getLogger(__name__)


def load_results():
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def run_config(config):
    LOGGER.info("Loading prior results.")
    results = load_results()

    LOGGER.info("Config: %s", config)
    LOGGER.info("Running experiment.")
    output = run_experiment(config)

    record = {
        "config": config,
        "log_loss": output["log_loss"]
    }

    results.append(record)

    LOGGER.info("Saving results.")
    save_results(results)

    LOGGER.info("Experiment complete. Log loss: %s", output["log_loss"])

    return record


if __name__ == "__main__":

    config = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "use_shooting": True,
        "use_free_throws": True,
        "use_rebounding": True,
        "use_turnovers": True,
        "use_defense": True,
        "use_playmaking": True,
        "use_win_history": True,
    }

    run_config(config)