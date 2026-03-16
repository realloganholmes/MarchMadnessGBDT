from run_experiment import run_experiment

def objective(config):

    result = run_experiment(config)

    return result["log_loss"]