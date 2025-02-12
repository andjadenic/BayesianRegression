# Bayesian Inference in Linear Regression: A Case Study using Ironman Triathlon World Championship Data

* The project explores the usage of Bayesian inference in linear regression to predict the overall finish time of participants in the Ironman Triathlon World Championship.
* The study utilizes datasets from the 2018 and 2019 championships, with run time as the predictor variable and overall finish time as the response variable.
* 2018 dataset to inform the prior distribution, capturing underlying relationships between run time and finish time.
* The 2019 dataset is then split into training and testing sets, enabling the estimation of posterior distributions for model parameters using Bayesian techniques.
* The predictive performance of the Bayesian model is assessed on the test set, and the results are compared to those obtained through classical statistical methods using MAE and RMSE as metrics.

### Dataset
This project uses datasets from Ironman, the most popular ultra-triathlon, consisting of 3.9 km of swimming, 181 km of cycling, and 42.2 km of running.
* The dataset consists of running and overall finish times for both 2018. and 2019. for athletes that finished the races that can be downloaded at [obstri.com/races](obstri.com/races. ).

### Model
We use Bayesian linear regression running time as a predictor $x$ and overall time as a response $y$:
$Y = \beta X + \alpha$
* We made an assumption that $\alpha$ and $\beta$ are normally distributed with prior distribution captured in 2018. dataset (for more details read Jupiter Notebook ['choosing_prior.ipynb'](https://github.com/andjadenic/BayesianRegression/blob/main/choosing_prior.ipynb)

### Learning Parameters
* We split the 2019. dataset into train and test subsets.
* Posteriors for $\alpha$ and $\beta$ are normal distributions learned from the training subset from 2019.

### Evaluation
We use MAE and RMSE for model evaluation on testing dataset.

### Used Packages
* `Pandas` for data manipulation
* `NumPy` for working with matrices
* `SciPy` for statistical computations
* `Matplotlib` for visualizations

### Supplementary PDF
For more details read [this PDF file](https://github.com/andjadenic/BayesianRegression/blob/main/Bayesian_Inference_in_Linear_Regression.pdf).
