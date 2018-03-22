#### William Bullock - IDS Assignment 2

### Preliminaries

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing

#reading in the data, as specified in question paper:

data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
data_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

#Splitting the input variables and labels, as specified in question paper:

x_train = data_train[:, :-1]
y_train = data_train[:, -1]

x_test = data_test[:, :-1]
y_test = data_test[:, -1]

### Exercise 1:

## Solution 1: scikit-learn method

knc = KNeighborsClassifier(n_neighbors=1, weights='uniform') # invoke sklearn function for a nearest neighbour classifier, to determine classes based on the single nearest otehr data vaue (k=1)

knc.fit(x_train, y_train) # fit the algorithm, training it by associating the 13 rotation and translation invariables with classes (x and y from the traning dataset).

predictions = knc.predict(x_test) # genereate predictions on the classes for the rotation and translation invariables in the test data.

acc_test = accuracy_score(y_test, predictions) #compare the observed(determined) classes with the expected(given) classes and generate an accuracy score.

print acc_test # returns 0.945993031359

### Execise 2:

## Solution 1: scikit-learn method

k_list = [i for i in range(1, 26, 2)] #forms a lsit of potential k values (all odd)

mean_loss = [] # empty list - will be filled with mean classification error from cross validation later.

for k in k_list: # for each value of k
    loss_list = [] # empty list

    knc = KNeighborsClassifier(k, weights='uniform') # invoke same sklearn as in exe1 using k as then number of nearest neighbours.
    cv = KFold(n_splits=5) #initialise cross validation to use 5 folds

    for train, test in cv.split(x_train): # for each train data set (4/5), and test data set (1/5).

        xtraincv, xtestcv, ytraincv, ytestcv = x_train[train], x_train[test], y_train[train], y_train[test]
        knc.fit(xtraincv, ytraincv) # train the function on (4/5) of the data.
        loss = (float(1) - float(accuracy_score(ytestcv, knc.predict(xtestcv)))) # test on the remaining 1/5 of the data, generate an accuracy score, convert into a classification error.
        loss_list.append(loss) # append the new classification error to 'loss_list'

        if len(loss_list) == 5: #when 'loss_list' has 5 entries...
            mean = np.mean(loss_list) #take the mean.
            mean_loss.append(mean) # append the mean classification error to 'mean_loss'

        else:
            continue

k_loss_tuples = zip(k_list, mean_loss) # marry the values in k_list to their respective mean classification error.

k_loss_sorted = sorted(k_loss_tuples, key=lambda x: x[1]) # sort this list by ascending classification error.

k_best = k_loss_sorted[0][0] # take the k value with the lowest classification error (highest accuracy) - to be 'k best'

print k_best  # returns 3

### Execise 3:

## Solution 1: scikit-learn method

#exactly the same as exercise 1, but with 'k best' (3) as the k hyperparameter for k-NN.

knc = KNeighborsClassifier(n_neighbors=k_best, weights='uniform')

knc.fit(x_train, y_train) # fit the algorithm, training it by associating the 13 rotation and translation invariables with classes (x and y from the traning dataset).

predictions = knc.predict(x_test) # genereate predictions on the classes for the rotation and translation invariables in the test data.

acc_test = accuracy_score(y_test, predictions) #compare the observed(determined) classes with the expected(given) classes and generate an accuracy score.

print acc_test # returns 0.949477351916

### Exercise 4:

## Solution 1: scikit-learn method

# Step 1: centering and normalizing the input data

scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(x_train)  #trains the standard scaler on the training data.

x_train_standardized = scaler.transform(x_train) # centres and normalizes  the training data.

x_test_standardized = scaler.transform(x_test)# centres and normalizes the testing data.

# Step 2: Cross validation for a new k best

k_list = [i for i in range(1, 26, 2)] #forms a lsit of potential k values (all odd)

mean_loss = [] # empty list - will be filled with mean classification error from cross validation later.

for k in k_list: # for each value of k
    loss_list = [] # empty list

    knc = KNeighborsClassifier(k, weights='uniform') # invoke same sklearn as in exe1 using k as then number of nearest neighbours.
    cv = KFold(n_splits=5) #initialise cross validation to use 5 folds

    for train, test in cv.split(x_train_standardized): # for each train data set (4/5), and test data set (1/5).

        xtraincv, xtestcv, ytraincv, ytestcv = x_train[train], x_train[test], y_train[train], y_train[test]
        knc.fit(xtraincv, ytraincv) # train the function on (4/5) of the data.
        loss = (float(1) - float(accuracy_score(ytestcv, knc.predict(xtestcv)))) # test on the remaining 1/5 of the data, generate an accuracy score, convert into a classification error.
        loss_list.append(loss)# append the new classification error to 'loss_list'

        if len(loss_list) == 5: #when 'loss_list' has 5 entries...
            mean = np.mean(loss_list) #take the mean.
            mean_loss.append(mean) # append the mean classification error to 'mean_loss'

        else:
            continue

k_loss_tuples = zip(k_list, mean_loss) # marry the values in k_list to their respective mean classification error.

k_loss_sorted = sorted(k_loss_tuples, key=lambda x: x[1])# sort this list by ascending classification error.

k_best = k_loss_sorted[0][0] # take the k value with the lowest classification error (highest accuracy) - to be 'k best'

print k_best # returns 3

# Step 3: generalization and accuracy score

# Exactly the same as exercise 3, again uses 'k best' as the k hyperparameter for k-NN. The training and test data used is this time standardized (centered and normalized).

knc = KNeighborsClassifier(n_neighbors=k_best, weights='uniform')

knc.fit(x_train_standardized, y_train) # fit the algorithm, training it by associating the 13 rotation and translation invariables with classes (x and y from the traning dataset).

predictions = knc.predict(x_test_standardized) # genereate predictions on the classes for the rotation and translation invariables in the standardized test data.

acc_test = accuracy_score(y_test, predictions) #compare the observed(determined) classes with the expected(given) classes and generate an accuracy score.

print acc_test # returns 0.959930313589