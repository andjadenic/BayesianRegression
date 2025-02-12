import numpy as np


def compute_predictive_distributions(x: np.ndarray, x_bar: float,
                                     m_alpha_x_prim: float, sigma_alpha_x_prim_squared: float,
                                     m_beta_prim: float, sigma_beta_prim_squared: float,
                                     sigma_squared: float):
    """
    For given numpy array of new observations x, method compute parameters (mean and std) of predictive distribution
    for each observation. Method returns two numpy arrays where the first one collects means and the second one
    collects squared standard deviations.
    """
    m_y_prim = m_alpha_x_prim + (x - x_bar) * m_beta_prim
    s_y_prim_squared = sigma_squared + sigma_alpha_x_prim_squared + sigma_beta_prim_squared * (x - x_bar) * (x - x_bar)
    return m_y_prim, s_y_prim_squared