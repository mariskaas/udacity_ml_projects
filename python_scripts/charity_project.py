# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, fbeta_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
# Read in dataset to a dataframe
data = pd.read_csv("/Users/mariskawillemsen/Documents/other/nanodegree_machinelearning_2020/DSND_Term1-master/projects/p1_charityml/census.csv")

# Data exploration
n_records = len(data)
n_greater_50k = len(data.loc[data['income'] == '>50K'])
n_at_most_50k = len(data.loc[data['income'] == '<=50K'])
greater_percent = (n_greater_50k/n_records) * 100

income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
# Initialize a scaler, then apply it to the numerical features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Apply hot encoding to features and replace outcome with zeroes and ones
features_final = pd.get_dummies(features_log_minmax_transform)
income = income_raw.replace({"<=50K": 0, ">50K":1 })

# Split set in training and testing
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size = 0.2,
                                                    random_state = 0)

TP = np.sum(income) # Counting the ones as this is the naive case.
FP = income.count() - TP # Specific to the naive case
TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case

# Accuracy, recall, precision and F-score calculations
accuracy = (TP+TN)/(TP+FP+FN+TN)
precision = (TP)/(TP+FP)
recall = (TP)/(TP+FN)
fscore = (1+(0.5**2)) *(precision*recall)/((0.5**2) * precision + recall)
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))



def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # Fit the model and get training time
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    results['train_time'] = end - start

    # Predict on train & test sets, calculate prediction times
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    results['pred_time'] = end - start

    # Calculate accuracy on train predictions
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # Calculate accuracy on test predictions
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # Calculate fbeta score on train predictions
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)

    # Calculate fbeta on test predictions
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    #print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results


#Initialze three models to test
clf_A = DecisionTreeClassifier(random_state=42)
clf_B = SVC(random_state=42)
clf_C = RandomForestClassifier(random_state=42)

# Get sample percentages
samples_100 = int(len(y_train))
samples_10 = int(samples_100/10)
samples_1 = int(samples_100/100)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Tuning hyperparameters with a gridsearch

random_seed = np.random.RandomState(seed=None)
clf = RandomForestClassifier(random_state=random_seed)

# Select parameters to tune
parameters = {'min_samples_split':range(2,5), 'min_samples_leaf':range(1,5), 'criterion':['gini', 'entropy']}

# Initiate a scorer
scorer = make_scorer(fbeta_score, beta=0.5)

# Perform grid search and get best parameters
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit with best parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the best estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
