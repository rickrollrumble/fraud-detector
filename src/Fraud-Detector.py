#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

# In[3]:

data = pd.read_csv('data/data.csv')

# In[4]:


data.head()

# Purely intuitively, most frauds are committed because of large amounts of money being deducted at the source and
# added at the destination account. So we create two additional fields, `diffOrig` and `diffDest` which is the
# difference in amount of money occured in each transaction.

# In[5]:


data['diffOrig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
data['diffDest'] = data['oldbalanceDest'] - data['newbalanceDest']

# There are three categorical variables - `type`,`nameOrig`,`nameDest`. The `nameOrig` and `nameDest` have too many
# categories with none of them holding a significant majority. So we shall drop them. We will keep `type` because of
# the limited number of categories.

# In[6]:


x = data['nameDest'].value_counts(normalize=True) * 100

# In[7]:


sns.distplot(x)

# Will probably end up discarded sourceDest and nameDest because the frequencies of individual names are too low as
# seen in the distribution plot. The plot normalized the frequency of each `nameDest` and plotted the number of
# occurences for a specific range of frequencies. It shows that a large number of destination accounts appear only a
# few times.<br> We move on to the payment types.

# In[8]:


data['type'].unique()

# For each transaction type, we obtain how many were flagged as frauds, and how many weren't.

# In[9]:


pd.crosstab(data['type'], data['isFraud'])

# Checking for the number of NA values.

# In[10]:


print('NA values\n', data.isna().sum())
print('Null values\n', data.isnull().sum())

# This tells us that no `NULL` data or `NaN` data is present in the data, so we don't need to deal with them here.

# Except `type`, each categorical variable is only a tiny percentage. We attempt to build a model without the other
# two categorical variables and one-hot encode the `type` variable. The `nameOrig` and `nameDest` columns are dropped.

# In[11]:


data_no_names = data.drop(['nameOrig', 'nameDest'], axis=1)

# Since there's a limited number of levels and limited variation in the `step` variable, it is normalized.

# In[12]:


data['step'].value_counts()

# So we need to use three preprocessors: <li> Normalizer - to convert the step variable to a 0-1 scale because of
# limited range and levels. <li> StandardScaler - to convert the amount, oldbalanceOrg, newbalanceOrig,
# oldbalanceDest, newbalanceDest, diffOrig, and diffDest to their z-scores $\frac{x-\mu}{\sigma}$. <li> One-Hot
# Encoding - done using pandas.get_dummies to convert type to dummy variables.


# In[14]:


sc = StandardScaler()
norm = Normalizer()
one_hot = OneHotEncoder()

# One-hot encoding all categorical data. We already removed the categorical data that didn't seem useful.

# In[15]:


data_no_names = pd.get_dummies(data_no_names)

# Normalizing steps column and making the changes to the column.

# In[16]:


data_no_names['step'] = norm.fit_transform(data_no_names['step'].values.reshape(1, -1)).reshape(-1, 1)

# Creating test and validate data by extracting `isFraud` and `isFlaggedFraud` columns, and dropping them from main
# dataset.

# In[17]:


frauds = data_no_names['isFraud']
validate = data_no_names['isFlaggedFraud']
data_no_names = data_no_names.drop(['isFraud', 'isFlaggedFraud'], axis=1)

# Using standard scaler on all other numerical data that isn't normalized or categorical.

# In[18]:


for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'diffOrig', 'diffDest']:
    data_no_names[col] = sc.fit_transform(np.array(data_no_names[col]).reshape(-1, 1))

# We will now verify if our preprocessing was done properly.

# In[19]:


data_no_names.head()

# The steps are normalized, all other values are standardized, and the type has been converted to dummy variables.

# Let us now count the number of 1s and 0s in the `isFraud` field.

# In[20]:


print(frauds.value_counts() / frauds.value_counts().sum())

# Here, we can see that the 1's or frauds are only 0.1% of the total transactions. This is extremely unbalanced,
# and performing random splitting and sampling results runs the risk this bias not being represented in the training
# and testing sets. So we use stratified sampling to get past this challenge.

# ### What and Why is Stratified Sampling? Small no. of 1s, and random_split may not represent properly. [Resource](
# https://medium.com/@411.codebrain/train-test-split-vs-stratifiedshufflesplit-374c3dbdcc36)

# Stratified Sampling is a **sampling technique** that is best used when a statistical population can easily be
# broken down into distinctive sub-groups. Then samples are taken from each sub-groups based on the ratio of the sub
# groups size to the total population. We can see that the 1's are a tiny fraction of the output class.  Using
# Stratified Sampling technique ensures that there will be selection from each sub-group (0 or 1) and prevents the
# chance of omitting one sub-group leading to sampling bias thus making the dog population happy!

# In[21]:

strat_shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=6)
for train_index, test_index in strat_shuffle_split.split(data_no_names, frauds):
    X_train, X_test = data_no_names.iloc[train_index], data_no_names.iloc[test_index]
    Y_train, Y_test = frauds.iloc[train_index], frauds.iloc[test_index]

# Importing the RandomForestClassifier

# In[22]:


# In K-Fold cross validation, the dataset is split into _k_ parts. One part is held back and the other parts are used
# for testing. Stratified K-Fold is a variation of KFold that uses stratified folds. The folds are made by preserving
# the percentage of samples for each class.

# In[23]:


skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=15, random_state=42)
i = 0
scores = []

for train_index, test_index in skf.split(X_train, Y_train):
    X_train_skf, X_test_skf = X_train.iloc[train_index], X_train.iloc[test_index]
    Y_train_skf, Y_test_skf = Y_train.iloc[train_index], Y_train.iloc[test_index]
    model = rf.fit(X_train_skf, Y_train_skf)
    scores.append(model.score(X_test_skf, Y_test_skf))
print('Forest trained')

# In[24]:


print(rf)

# Now that our model is trained, let us examine which features turned out to be the most important.<br>

# In[25]:


feature_imp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
ax = feature_imp.plot(kind='bar')

# <p>It is observed that `diffOrig`, the feature that we created from the difference in balance at the origin,
# and `diffDest`, the feature that we created from the difference in balance at the destination are the most
# important features.</p> <p>We had also observed previously using crosstables that all the frauds happened when the
# transaction type was TRANSFER or CASH_OUT. While this seems to be holding true for transfers, it does not work too
# well for cash outs. This has to be improved using model tuning.

# In[26]:


res = model.predict(X_test)

# <p>As we saw, this was an imbalanced classification problem due to the small fraction of 1's in the data. So
# accuracy on it's own is not a good measure of whether our model did well. We need to focus on two metrics,
# precision and recall.</p> <p>Precision measures how accurately the model measures predicted positives.
# $$precision=\frac{true\ positive}{true\ positive\ +\ false\ positive}$$ </p> <p>Recall actually calculates how many
# of the Actual Positives our model capture through labeling it as Positive. $$precision=\frac{true\ positive}{true\
# positive\ +\ false\ negative}$$ </p> <p> These metrics are combined to calculate the f1-score. </p>

# More resources:<br>
# https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9<br>
# https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c

# In[27]:

print('********** CLASSIFICATION REPORT **********')
print(classification_report(res, Y_test))
print('\n********** CONFUSION MATRIX **********')
print(confusion_matrix(res, Y_test))
print('\n********** F1 SCORE **********')
print(f1_score(res, Y_test))

# ### Tuning the model Personally, 87% is a good start. But we can always try to improve the accuracy. So we turn to
# Grid Search. Grid Search fits the model for different combinations of hyperparameters and returns the combination
# with the best scores.

# In[28]:

n_estimators = [10, 20]
max_depth = [5, 8]
min_samples_split = [1, 2, 5]
min_samples_leaf = [1, 2]

hyperF = dict(n_estimators=n_estimators, max_depth=max_depth,
              min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf)

# We will now fit the model for both kinds of scores; precision and recall.

# In[29]:


scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        RandomForestClassifier(), hyperF, scoring='%s_macro' % score,
        cv=3, verbose=1, n_jobs=-1
    )
    clf.fit(X_train, Y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(f1_score(y_true, y_pred))
    print()

# The max f1-score obtained is less than our original method without any tuning! This didn't work as well as we
# thought it would. Maybe we should try a different model. Let us try out a support vector machine classifier now.

# In[30]:


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=6)
svc = LinearSVC(dual=False, random_state=42)
i = 0
scores = []

for train_index, test_index in skf.split(X_train, Y_train):
    X_train_skf, X_test_skf = X_train.iloc[train_index], X_train.iloc[test_index]
    Y_train_skf, Y_test_skf = Y_train.iloc[train_index], Y_train.iloc[test_index]
    svc_model = svc.fit(X_train_skf, Y_train_skf)
    scores.append(svc_model.score(X_test_skf, Y_test_skf))
    i += 1
print('SVC trained')

# In[31]:


res = svc_model.predict(X_test)

# We will find the metrics for the Support Vector Classifier.

# In[32]:

print('********** CLASSIFICATION REPORT **********')
print(classification_report(res, Y_test))
print('\n********** CONFUSION MATRIX **********')
print(confusion_matrix(res, Y_test))
print('\n********** F1 SCORE **********')
print(f1_score(res, Y_test))

# ### Exporting the Models
# The models can be found in the `models` directory.

# In[36]:

filename = 'random_forest.sav'
joblib.dump(model, filename)

filename = 'random_forest_grid_search.sav'
joblib.dump(clf, filename)

filename = 'svc.sav'
joblib.dump(svc, filename)

# ### Conclusions <p>Out of the box, the random forest classifier works the best. The different combinations of trees
# detects fraudulent transactions the best based on the f1 score. Hyperparameter tuning made it worse,
# though it should be noted that a grid search to find an ideal set of parameters required way more computing power
# than my laptop can manage.</p> <p>The SVC does not compare at all in performance in this case, though it is also
# the quickest model to train.</p> <p> Increasing the number of folds and estimators only results in marginal
# increase in the f1-score but increases the computation time significantly.</p> <p>The most useful features in
# classification turn out to be the features that were created from existing features, `diffOrig` and `diffDest`.
