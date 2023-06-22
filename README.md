# Project7-OpenClassrooms
Implement a scoring model

## Business case
The company "Prêt à dépenser" offers consumer credit for people with little or no loan history.

The company wants to implement a “credit scoring” tool to calculate the probability that a customer will repay his credit, then classify the request as granted or refused credit. The company thus wishes to develop a classification algorithm based on various data sources (behavioral data, data from other financial institutions, etc.).

The data used for this project is available on Kaggle at https://www.kaggle.com/competitions/home-credit-default-risk

NB: A first preprocessing was done using the Kaggle kernel available at https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script

## Objectives 

Build a scoring model, taking into account the class imbalance, that will predict a client's default probability.



## Main results

### Data cleaning and analysis

A first processing and feature engineering was carried out using a Kaggle kernel. The dataset after this preprocessing included 799 variables and 356251 customers. The number of customers having repaid the loan is 11 times the number of customers having defaulted. Following the replacement of missing values, potentially interesting variables for modeling were selected using the 3 methods:
-	removal of variables with too little variance
-	removal of variables according to their distribution in relation to the target
 -  removal of highly correlated variables.

Outliers have also been replaced.

### Modeling

  - class imbalance was managed using 4 methods: oversampling, undersampling, SMOTETomek and class weights
 - a reduced dataset of 32 variables was prepared using Recursive Feature Elimination
 - business metric, the “credit score”, was crated
   - several types of models were tested (Logistic Regression, Random Forest, XGBoost and LightGBM)
  - for each type of model and imbalance treatment method, the most promising model was selected by comparing ROC-AUC, accuracy, recall, credit score and run time
   - for each type of model, GridSearchCV/RandomizedSearchCV was used to optimize the hyperparameters
   - the models were compared after optimization 

The best model is LightGBM, which was further optimized using Bayesian optimization. The classification threshold of 0.5 minimizes financial losses, while ensuring sufficient accuracy and recall.

## Acquired Skills

-	Define and implement a model performance monitoring strategy
-	Define the strategy for developing a supervised learning model
-	Evaluate the performance of supervised learning models





