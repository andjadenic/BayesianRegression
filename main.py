import pandas as pd
from sklearn.model_selection import train_test_split
from compute_posterior import *
from visualization import *
from credible_intervals import *
from predictions import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Preprocessing is done in preprocessing.ipynb

# Choosing the prior for beta and alpha_x based on 2018. results is done in choose_prior.py
mean_prior_beta = 1.8535035966136237
std_prior_beta = 1/3
mean_prior_alpha_x = 11.239965559563169
std_prior_alpha_x = 0.6666666666666666

# Read csv file with results from 2019. Ironman
df = pd.read_csv('ironman_data\data_2019_lin_reg.csv')
x, y = df['run_2019'].values, df['overall_2019'].values

# Split the data from 2019. on train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Compute posterior distributions for beta and alpha_x
mean_posterior_alpha_x, std_posterior_alpha_x_squared, mean_posterior_beta, std_posterior_beta_squared, std_squared = \
    posterior_for_normal_prior(x_train, y_train, mean_prior_alpha_x,
                               std_prior_alpha_x, mean_prior_beta, std_prior_beta)

# Print prior and posterior values
print(mean_prior_alpha_x, ' ', mean_posterior_alpha_x)
print(std_prior_alpha_x, ' ', std_posterior_alpha_x_squared*.5)
print(mean_prior_beta, ' ', mean_posterior_beta)
print(std_prior_beta, ' ', std_posterior_beta_squared**.5)
print('std_squared = ', std_squared)
print('std = ', std_squared ** .5)

# Visualize prior and posterior distributions for both parameters
"""
plot_prior_and_posterior(mean_prior_alpha_x, std_prior_alpha_x,
                         mean_posterior_alpha_x, std_posterior_alpha_x_squared,
                         r'Prior and posterior distriburions of $\alpha_{\bar{x}}$',
                         'prior_and_posterior_alpha_x')
plot_prior_and_posterior(mean_prior_beta, std_prior_beta,
                         mean_posterior_beta, std_posterior_beta_squared,
                         r'Prior and posterior distributions of $\beta$',
                         'prior_and_posterior_beta')
                         """

# Visualize prior and posterior for alpha_x
plot_prior_and_posterior(mean_prior = mean_prior_alpha_x, std_prior = std_prior_alpha_x,
                         mean_posterior = mean_posterior_alpha_x, std_posterior_squared = std_posterior_alpha_x_squared,
                         label_of_fig = 'prior_and_posterior_alpha_x',
                         name_of_fig = 'Prior and posterior probability distributions of alpha_x')

# Visualize prior and posterior for beta
plot_prior_and_posterior(mean_prior = mean_prior_beta, std_prior = std_prior_beta,
                         mean_posterior = mean_posterior_beta, std_posterior_squared = std_posterior_beta_squared,
                         label_of_fig = 'prior_and_posterior_beta',
                         name_of_fig = 'Prior and posterior probability distributions of beta')

# Bayesian Credible Interval for Slope
n = len(y_train)
alpha = .05
beta_lower, beta_upper = credible_interval_for_slope(mean_posterior_beta, std_squared, n, alpha)
print(f'{100*(1-alpha)}% Bayesian Credible Interval for Slope is: [{beta_lower}, {beta_upper}]')

# Compute parameters for normal predictive distribution for future observations
x_mean = np.mean(x)
mean_pred, std_pred_squared = compute_predictive_distributions(x_test, x_mean,
                                                               mean_posterior_alpha_x, std_posterior_alpha_x_squared,
                                                               mean_posterior_beta, std_posterior_beta_squared,
                                                               std_squared)
y_sample_of_predictions = np.random.normal(mean_pred, std_pred_squared ** .5)

# Evaluation
mae = np.mean(np.abs(y_sample_of_predictions - y_test))  # mean absolute error
rmse = (np.mean((y_sample_of_predictions - y_test)**2)) ** .5 # root mean sqared error
print(f'Bayesian MAE = {mae}')
print(f'Bayesian RMSE = {rmse}')

# Visualize true and predicted values
plt.figure(figsize=(8,6))
plt.scatter(x_test, y_test, color='purple', s=2, label='true values')
plt.scatter(x_test, y_sample_of_predictions, color='deeppink', s=2, label='predicted values')
plt.legend()
#plt.show()

# Comparison to the classical statistics
model = LinearRegression()
model.fit(x_train.reshape(-1, 1), y_train)
y_pred = model.predict(x_test.reshape(-1, 1))

rmse_cl = np.sqrt(mean_squared_error(y_test, y_pred))
mae_cl = mean_absolute_error(y_test, y_pred)

print(f"Classical RMSE: {rmse_cl}")
print(f"Classical MAE: {mae_cl}")