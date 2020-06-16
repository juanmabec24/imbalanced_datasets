%%time
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.metrics import confusion_matrix

''' Define model
    Note: here I take advantage of the "tree_method" argument to train on GPU, which really speeds things up'''
model = XGBClassifier(
    tree_method = "gpu_hist", 
    eval_metric=["error","aucpr"],
    verbosity = 3)
    
# Define the grid of hyperparameters we will use in cross-validation to find the best combination of them
weights = [50, 110, 125, 250]
param_grid = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(40, 240, 20),
    'learning_rate':  [0.1, 0.01, 0.05, 0.005],
    'scale_pos_weight': weights
}

# Define evaluation procedure 
scv = StratifiedKFold(n_splits=20)

# Define grid search (cross validation)
grid = RandomizedSearchCV(estimator=model, param_distributions= param_grid, n_jobs=-1, cv=scv, scoring='f1')

# Execute the grid search
grid_result = grid.fit(X_train, y_train)

# Print the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
   
y_pred_xg = grid.best_estimator_.predict(X_test)
f1_score_xg = f1_score(y_pred_xg, y_test)
print("F1 score : {}".format(f1_score_xg))
print("")

# Compute confusion matrix
cnf_matrix_xg = confusion_matrix(y_test, y_pred_xg)

# Plot non-normalized confusion matrix
plt.figure(figsize=(6, 6))
plot_confusion_matrix(cnf_matrix_xg, classes=["Non-defalut", "Default"])
plt.show()
print()
