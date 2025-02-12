import numpy as np
from scipy.stats import t


def credible_interval_for_slope(mean: float, std_squared: float, n: int, alpha: float = .05):
    t_alpha_half = -t.ppf(alpha / 2, n-2)
    lower_bound = mean - t_alpha_half * std_squared ** .5
    upper_bound = mean + t_alpha_half * std_squared ** .5
    return lower_bound, upper_bound