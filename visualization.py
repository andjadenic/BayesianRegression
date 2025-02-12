import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_prior_and_posterior(mean_prior: float, std_prior: float,
                             mean_posterior: float, std_posterior_squared: float,
                             label_of_fig: str, name_of_fig:str):
    """
    Method is used for visualising prior and posterior distribution
    if prior is normal(mean_prior, std_prior^2)
    and posterior is normal(mean_posterior, std_posterior_squared).
    Figure will be saved in plots folder.
    """
    left, right = min(mean_prior, mean_posterior), max(mean_prior, mean_posterior)
    max_std = max(std_prior, std_posterior_squared ** .5)
    start, finish = left - 3*max_std, right + 3*max_std
    x_plot = np.linspace(start, finish, 500)
    y_prior = norm.pdf(x_plot, mean_prior, std_prior)
    y_posterior = norm.pdf(x_plot, mean_posterior, std_posterior_squared ** .5)

    plt.figure(figsize=(8, 6))
    plt.plot(x_plot, y_prior, linestyle='--', label='prior distribution', color='deeppink')
    plt.plot(x_plot, y_posterior, label='posterior distribution', color='deeppink')
    plt.xlabel(r'value of a parameter')
    plt.ylabel('probability density')
    plt.title(label_of_fig)
    plt.legend()

    plt.grid(True)
    plt.savefig('plots/' + name_of_fig + '.png')
    #plt.show()

