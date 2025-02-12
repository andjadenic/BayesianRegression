import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t


def posterior_for_normal_prior(x: np.ndarray, y: np.ndarray,
                   m_alpha_x: float, s_alpha_x: float,
                   m_beta: float, s_beta: float):
    """
    Returns distribution parameters of slope beta and x_bar intersect alpha_x for simple linear regression model:
    y = beta * (x - x_bar) + alpha_x
    given the data (x - predictor, y - response variable) and assuming that priors for parameters are normal with distributions:
    beta ~ (m_beta, s_beta^2) and alpha_x ~ (m_alpha_x, s_alpha_x^2),
    Posterior distributions are normal as well with distribution:
    beta ~ (m_beta_prim, s_beta_prim_squared) and alpha_x ~ (m_alpha_x_prim, s_alpha_x_prim_squared)
    """
    # Calculating intermediate results
    x_mean = np.mean(x)
    SSx = np.sum((x - x_mean) ** 2)
    n = len(y)

    B = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x ** 2) - (np.mean(x) ** 2))
    y_intercept = np.mean(y) - B * np.mean(x)
    SSR = np.sum((y - B * x - y_intercept) ** 2)  # Sum of Squared Residuals
    sigma_squared = SSR / (n - 2)
    Ax_bar = np.mean(y)

    # Calculating output
    s_beta_prim_squared = 1 / (1 / (s_beta ** 2) + SSx / sigma_squared)
    m_beta_prim = (s_beta_prim_squared / (s_beta ** 2)) * m_beta + \
                  ((SSx * s_beta_prim_squared) / sigma_squared) * B
    s_alpha_x_prim_squared = 1 / ((1 / s_alpha_x ** 2) + (n / sigma_squared))
    m_alpha_x_prim = (s_alpha_x_prim_squared / (s_alpha_x ** 2)) * m_alpha_x + \
                     ((n * s_alpha_x_prim_squared) / sigma_squared) * Ax_bar
    return m_alpha_x_prim, s_alpha_x_prim_squared, m_beta_prim, s_beta_prim_squared, sigma_squared




