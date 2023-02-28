from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def spearman(outputs: list[float], targets: list[float]) -> float:
    """
    Returns the Spearman Ranked Correlation between a list of model outputs and a list of targets.
    :param outputs: A list of floats of values outputted by the model.
    :param targets: A lis of floats of targets of the model's training.
    :return: The Spearman Ranked Correlation between the outputs and targets.
    """
    return spearmanr(outputs, targets)[0]


def kendall(outputs: list[float], targets: list[float]) -> float:
    """
    Returns the Kendall's Tau Ranked Correlation between a list of model outputs and a list of targets.
    :param outputs: A list of floats of values outputted by the model.
    :param targets: A lis of floats of targets of the model's training.
    :return: Kendall's Tau Ranked Correlation between the outputs and targets.
    """
    return kendalltau(outputs, targets)[0]


def pearson(outputs: list[float], targets: list[float]) -> float:
    """
    Returns the pearson correlation between a list of model outputs and a list of targets.
    :param outputs: A list of floats of values outputted by the model.
    :param targets: A lis of floats of targets of the model's training.
    :return: Kendall's Tau Ranked Correlation between the outputs and targets.
    """
    try:
        return pearsonr([o[0] for o in outputs], targets)[0]
    except Exception:
        return pearsonr(outputs, targets)[0]


def mse(outputs: list[float], targets: list[float]) -> float:
    """
    Returns the mean squared error between a list of model outputs and a list of targets.
    :param outputs: A list of floats of values outputted by the model.
    :param targets: A lis of floats of targets of the model's training.
    :return: The mean squared error between the outputs and targets.
    """
    return mean_squared_error(targets, outputs)


def rmse(outputs: list[float], targets: list[float]) -> float:
    """
    Returns the root mean squared error between a list of model outputs and a list of targets.
    :param outputs: A list of floats of values outputted by the model.
    :param targets: A lis of floats of targets of the model's training.
    :return: The mean squared error between the outputs and targets.
    """
    # Temporary set to mean_squared_error to keep sweep data consistent
    return mean_squared_error(targets, outputs, squared=False)


def mae(outputs: list[float], targets: list[float]) -> float:
    """
    Returns the mean absolute error between a list of model outputs and a list of targets.
    :param outputs: A list of floats of values outputted by the model.
    :param targets: A lis of floats of targets of the model's training.
    :return: The mean absolute error between the outputs and targets.
    """
    # Temporary set to mean_squared_error to keep sweep data consistent
    return mean_absolute_error(targets, outputs)


def all_metrics():
    metrics = {
        "spearman": spearman,
        "kendall": kendall,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
    }

    return metrics
