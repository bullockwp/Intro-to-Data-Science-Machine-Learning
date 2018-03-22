#### William Bullock - IDS Assignment 5

### Preliminaries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

### Exercise 1

#loading data
wine_train = np.loadtxt('redwine_training.txt')
wine_test = np.loadtxt('redwine_testing.txt')

# splitting the data into input variable matrix and output vectors

x_wine_train = wine_train[:, :-1]
y_wine_train = wine_train[:, -1]

x_wine_test = wine_test[:, :-1]
y_wine_test = wine_test[:, -1]

## a)

def multivarlinreg(xdata, ydata): #Adapted from lecture 11 iPython notebook:
    """ Accepts matrix of input variables and same length list of output vectors
    and returns a list of regression coefficients """
    # creates a matrix with one column f 1's the same length as the input data
    onevec = np.ones((len(ydata), 1))
    # adds this matrix to the input variables matrix
    X = np.concatenate((onevec, xdata), axis=1)
    # computes the regression coefficients
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), ydata)
    return w

## b)

#take only the fixed acidity variables from the train dataset
acid_only_train = x_wine_train[:, :1]

#make them an array
acid_array_train = np.array(acid_only_train)

#find the weights for these
acid_theta = multivarlinreg(acid_array_train, y_wine_train)

print acid_theta

## c)

#find the weights for all 11 variables
all_thetas = multivarlinreg(x_wine_train, y_wine_train)

print all_thetas


### Exercise 2

## a)

def rmse(f, t):
    """Accepts and array of predicted values, and an array of ground truth values of same length,
     and predicts root mean squared error"""
    #initializes sum_error
    sum_error = 0.0
    #iterates over all pairs at between f and t at each position
    for i in range(len(f)):
        #calculates the absolute difference between the members of the pair
        error = np.linalg.norm(f[i]) - np.linalg.norm(t[i])
        #takes the square of this error and adds it to the sum_error value
        sum_error += (error ** 2)
    #takes the mean of the sum of all errors
    mean_error = sum_error / float(len(t))
    #returns the square root of the mean.
    return np.sqrt(mean_error)

## b)

def smartlinreg(xtrain, ytrain, xtest):
    """
    Fits to training data input variables and outputs, then performs linear regression on test data to predict outputs.
    note that the training and testing data should use hte same number of variables.
    :param xtrain: training data input
    :param ytrain: training data output
    :param xtest: testing data input
    :return: testing data output

    """
    #call outside function to calculate the w parameters in the training data.
    thetas = multivarlinreg(xtrain, ytrain)
    # creates a matrix with one column f 1's the same length as the input data
    onevec = np.ones((len(xtest), 1))
    # adds this matrix to the input variables matrix
    X = np.concatenate((onevec, xtest), axis=1)
    #finds the dot product of the parameters multiplied by the x values to give the t value
    t = np.dot(X, thetas)
    return t

#take only the fixed acidity variables from the test dataset
acid_only_test = x_wine_test[:, :1]

#make them an array
acid_array_test = np.array(acid_only_test)

#find the weights for these
acid_t = smartlinreg(acid_array_train, y_wine_train, acid_only_test)

#perform root mean squared error computation
acid_rmse = rmse(acid_t, y_wine_test)

print acid_rmse

## c)

#find the weights for all 11 variables
all_t = smartlinreg(x_wine_train, y_wine_train, x_wine_test)

#perform root mean squared error computation
all_rmse = rmse(all_t, y_wine_test)

print all_rmse


### Exercise 3

#Theoretical only

### Exercise 4

data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
data_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

# Separating the datasets by rotation & translation (x) and class (y)
x_train = data_train[:, :-1]
y_train = data_train[:, -1]

x_test = data_test[:, :-1]
y_test = data_test[:, -1]

# Invoke sklearn function for a random forest classifier using 50 trees
RFC = RandomForestClassifier(n_estimators=50)

# fit the algorithm, training it by associating each of the 13 rotation and translation invariables with classes (x and y from the traning dataset).
RFC.fit(x_train, y_train)

# Using the fit, predict the class values (y) for the rotation and translation invariables (x) for teh test dataset.
predicitons = RFC.predict(x_test)

# Compare the observed(determined) classes with the expected(given) classes and generate an accuracy score.
acc_test = accuracy_score(y_test, predicitons)

print acc_test

### Exercise 5

def function(x):
    """ returns y for x in the function given in exercise 5"""
    return np.exp(-x/2)+10*x**2

def deriv(x):
    """returns y for x in the derivative of the function given for exercise 5"""
    return 20*x-(np.exp(-x/2)/2)

# adapted from Ipython Jupyter notebook in lecture 12
def linreg_graddesc(x, learningrate, max_iter, tolerance):
    """
    Performs gradient descent using linear regression

    :param x: starting point of x (recommended x=1)
    :param learningrate: learning rate (recommended 0.1 to 0.0001)
    :param max_iter: maximum number of iterations
    :param tolerance: minimum movement step size. (recommended 0.001)
    :return: current value of x when algorithm terminates, the number of iterations completed, list of all observed values of x
    """

    # Initialize a vector of observed values of the error function we are minimizing; for evaluation purposes
    values = []
    # initialize the starting point for parameters
    cur_val = x
    num_iter = 0
    convergence = 0
    # while convergence is equal to 0
    while convergence == 0:
        # append the current value to the vector
        values.append(cur_val)
        # calculate the gradient by finding the derivative of the current value
        grad = deriv(cur_val)
        # calculate step size
        move = learningrate*grad
        # calculate new value as the old value after a step is taken
        value = cur_val - move

        # add to number of iterations
        num_iter = num_iter + 1
        # calculate difference (for tolerance)
        diff = abs(value - cur_val)

        # escape while loop if difference is below the tolerance threshold
        if diff < tolerance:
            convergence = 1
        # or if the number of iterations becomes too high
        elif num_iter > max_iter:
            convergence = 1

        # Update the value
        cur_val = value

    return cur_val, num_iter, values


def plotdlinreg(values, title):
    """Plots all the x values along a linear regression."""
    # import cool colour spectrum
    cool = plt.get_cmap('cool')
    gradient = np.linspace(0.0, 1.0, len(values))
    # separate cool spectrum into appropriate stages.
    colours = [cool(c) for c in gradient]

    # plotting the function
    gd_test = np.linspace(-2, 2, 100)
    plt.plot(gd_test, function(gd_test))

    # plotting the points on the function

    ci = 0
    for i in values:
        plt.plot(i, function(i), "o", color=colours[ci])

        ci += 1

    # plotting the tangent, only if the number of iterations is 3
    if len(values) == 4:
        for x in values:

            #finding the gradient, and intercepts
            m = deriv(x)
            b = function(x) - m * x
            z1 = -2
            z2 = 2

            tan1 = m * z1 + b
            tan2 = m * z2 + b

            #making the plot
            plt.plot([z1, z2], [tan1, tan2])


    plt.title(title, fontweight='bold')
    plt.axis([-3, 3, -5, 30])
    plt.xlabel('X')
    plt.ylabel('Y') #visual tweaks


#3 iterations
#0.1

lingd_test1_1 = linreg_graddesc(1, 0.1, 3, 0.001)
print lingd_test1_1

plotdlinreg(lingd_test1_1[2], "3 interations - learning rate of 0.1")

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/3.0.1.png', bbox_inches='tight', dpi=500)
plt.close()

#0.01

lingd_test1_2 = linreg_graddesc(1, 0.01, 3, 0.001)
print lingd_test1_2

plotdlinreg(lingd_test1_2[2], "3 interations - learning rate of 0.01")

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/3.0.01.png', bbox_inches='tight', dpi=500)
plt.close()

#0.001

lingd_test1_3 = linreg_graddesc(1, 0.001, 3, 0.001)
print lingd_test1_3

plotdlinreg(lingd_test1_3[2], "3 interations - learning rate of 0.001")

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/3.0.001.png', bbox_inches='tight', dpi=500)
plt.close()

#0.0001

lingd_test1_4 = linreg_graddesc(1, 0.0001, 3, 0.001)
print lingd_test1_4

plotdlinreg(lingd_test1_4[2], "3 interations - learning rate of 0.0001")

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/3.0.0001.png', bbox_inches='tight', dpi=500)
plt.close()


#10 iterations
#0.1

lingd_test2_1 = linreg_graddesc(1, 0.1, 10, 0.001)
print lingd_test2_1

plotdlinreg(lingd_test2_1[2], "10 interations - learning rate of 0.1")

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/10.0.1.png', bbox_inches='tight', dpi=500)
plt.close()

#0.01

lingd_test2_2 = linreg_graddesc(1, 0.01, 10, 0.001)
print lingd_test2_2

plotdlinreg(lingd_test2_2[2], "10 interations - learning rate of 0.01")

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/10.0.01.png', bbox_inches='tight', dpi=500)
plt.close()

#0.001

lingd_test2_3 = linreg_graddesc(1, 0.001, 10, 0.001)
print lingd_test2_3

plotdlinreg(lingd_test2_3[2], "10 interations - learning rate of 0.001")

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/10.0.001.png', bbox_inches='tight', dpi=500)
plt.close()


#0.0001
lingd_test2_4 = linreg_graddesc(1, 0.0001, 10, 0.001)
print lingd_test2_4

plotdlinreg(lingd_test2_4[2], "10 interations - learning rate of 0.0001")

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/10.0.0001.png', bbox_inches='tight', dpi=500)
plt.close()

#10000 iterations
#0.1

lingd_test3_1 = linreg_graddesc(1, 0.1, 10000, 10**-10)
print lingd_test3_1

#0.01

lingd_test3_2 = linreg_graddesc(1, 0.01, 10000, 10**-10)
print lingd_test3_2

#0.001

lingd_test3_3 = linreg_graddesc(1, 0.001, 10000, 10**-10)
print lingd_test3_3

#0.0001

lingd_test3_3 = linreg_graddesc(1, 0.0001, 10000, 10**-10)
print lingd_test3_3

## Exercise 6

# Preliminaries

# importing data
Iris2D1_train = np.loadtxt('Iris2D1_train.txt')
Iris2D1_test = np.loadtxt('Iris2D1_test.txt')

Iris2D2_train = np.loadtxt('Iris2D2_train.txt')
Iris2D2_test = np.loadtxt('Iris2D2_test.txt')

# Initial plots


# assigning colour based on class
colourbyclass1 = ['b' if i==1 else 'r' for i in Iris2D1_train[:, 2]]
colourbyclass2 = ['b' if i==1 else 'r' for i in Iris2D1_test[:, 2]]

# plot Iris2D1
plt.scatter(Iris2D1_train[:, 0], Iris2D1_train[:, 1], color=colourbyclass1)
plt.scatter(Iris2D1_test[:, 0], Iris2D1_test[:, 1], color=colourbyclass2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Iris2D1 Training & Test data', fontweight='bold')

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/E6-1.png', bbox_inches='tight', dpi=500)
plt.close()


# assigning colour based on class again
colourbyclass1 = ['b' if i==1 else 'r' for i in Iris2D2_train[:, 2]]
colourbyclass2 = ['b' if i==1 else 'r' for i in Iris2D2_test[:, 2]]

# plot Iris2D2
plt.scatter(Iris2D2_train[:, 0], Iris2D2_train[:, 1], color=colourbyclass1)
plt.scatter(Iris2D2_test[:, 0], Iris2D2_test[:, 1], color=colourbyclass2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Iris2D2 Training & Test data', fontweight='bold')

# plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/E6-2.png', bbox_inches='tight', dpi=500)
plt.close()


## logistic regression

# A lot of the code for this section of the exercise was adapted from the Jupyter iPython notebook from Lecture 13.

def logistic(input):
    """ returns y, the logistic function of x"""
    out = np.exp(input)/(1 + np.exp(input))
    return out

def logistic_insample(X, y, w):
    """ computes in-sample logistic loglikelihood"""
    # Extract the number of values in the matrix
    N, num_feat = X.shape
    #initialize logistic loglikelihood as 0
    E = 0
    #for each value, calculate log loglikelihood
    for n in range(N):
        E += (1.0/N)*np.log(1 + np.exp(np.dot(-y[n], np.dot(w,X[n,:]))))
    return E

def logistic_gradient(X, y, w):
    """ finds the gradient of a logistic function """
    # Extract the number of values in the matrix
    N, _ = X.shape
    g = 0*w
    #for each value, calculate gradient
    for n in range(N):
        g += ((-1/N)*y[n]*X[n,:])*logistic(-y[n]*np.dot(w, X[n,:]))
    return g

def log_reg(Xorig, y, max_iter, grad_thr):
    """
    computes weights of each variable in a logistic regression function

    :param Xorig: training variables matrix (x)
    :param y: training class values matrix (y)
    :param max_iter: Maximum iterations
    :param grad_thr: gradient threshold
    :return: weights for each input variable, list of predicted values
    """

    #extracts the number of columns and rows from training variables matrix
    num_pts, num_feat = Xorig.shape
    #creates a vector of 1s, equal to the number of columns in the training variables matrix
    onevec = np.ones((num_pts, 1))
    #appends this vector to the front of the input matrix
    X = np.concatenate((onevec, Xorig), axis=1)
    dplus1 = num_feat + 1

    # y is a N by 1 matrix of target values -1 and 1
    y = np.array((y - .5) * 2.0)

    # Initialize learning rate for gradient descent
    learningrate = 0.01

    # Initialize weights at time step 0
    w = 0.1 * np.random.randn(dplus1)

    # Compute value of logistic log likelihood
    value = logistic_insample(X, y, w)

    # initialize number of iterations and convergence
    num_iter = 0
    convergence = 0

    # Keep track of function values
    E_in = []

    while convergence == 0:
        # add to the count of the number of iterations
        num_iter = num_iter + 1

        # Compute gradient at current w by calling gradient function
        g = logistic_gradient(X, y, w)

        # Set direction to move
        v = -g

        # Update weights
        w_new = w + learningrate * v

        # Compute in-sample logistic log likelihood for new w, by calling logistic log likelihood function
        cur_value = logistic_insample(X, y, w_new)

        #re calculate current value, weights and learning rate (move step size)
        if cur_value < value:
            w = w_new
            value = cur_value
            E_in.append(value)
            learningrate *= 1.1
        else:
            learningrate *= 0.9

        # Determine whether we have converged?
        g_norm = np.linalg.norm(g)
        if g_norm < grad_thr:
            convergence = 1

        # threshold, and have we reached max_iter?
        elif num_iter > max_iter:
            convergence = 1

    #return weights
    return w, E_in

def log_pred(Xorig, w):
    """ given input variable matrix (x) and  weights, will predict class variables (y) using logistic regression"""

    # extracts the number of columns and rows from training variables matrix
    num_pts, num_feat = Xorig.shape
    # creates a vector of 1s, equal to the number of columns in the training variables matrix
    onevec = np.ones((num_pts, 1))
    # appends this vector to the front of the input matrix
    X = np.concatenate((onevec, Xorig), axis=1)

    #creates an 'empty' matrix of zeros ready to receive class values
    N, _ = X.shape
    P = np.zeros(N)

    #for each value, predicts class
    for n in range(N):
        P[n] = 1/(1+np.exp(-np.dot(w, X[n,:])))

    # 0/1 class labels
    Pthresh = np.round(P)
    return Pthresh

#separating variables and class Iris 1
iris1_train_matrix = Iris2D1_train[:,[0,1]]
iris1_train_labels = Iris2D1_train[:,2]

iris1_test_matrix = Iris2D1_test[:,[0,1]]
iris1_test_labels = Iris2D1_test[:,2]

#separating variables and class Iris 2
iris2_train_matrix = Iris2D2_train[:,[0,1]]
iris2_train_labels = Iris2D2_train[:,2]

iris2_test_matrix = Iris2D2_test[:,[0,1]]
iris2_test_labels = Iris2D2_test[:,2]

def my_logistic_regression(train_x, train_y, test_x):
    """
    given training matrix of x values, training matrix of y values, and test set of x values,
    will predict the y values (class) for the test set, using logistical regression.
    Returns predicted class values and weights used for each input variable """

    # calls logistic regression function
    w, E = log_reg(train_x, train_y, 100000, 0.001)
    print w

    #calls prediction function
    P = log_pred(test_x, w)

    return P, w

#iris1 data

#Calling function on training - train set
iris1_test_predictions, iris1_w_train = my_logistic_regression(iris1_train_matrix, iris1_train_labels, iris1_train_matrix)

#calculate 0-1 loss
iris1_test_score1 = accuracy_score(iris1_train_labels, iris1_test_predictions)
print (1 - iris1_test_score1)

#Calling function on training - test set
iris1_test_predictions, iris1_w_test = my_logistic_regression(iris1_train_matrix, iris1_train_labels, iris1_test_matrix)

#calculate 0-1 loss
iris1_test_score2 = accuracy_score(iris1_test_labels, iris1_test_predictions)
print (1 - iris1_test_score2)


#iris2 data

#Calling function on training - train set
iris2_test_predictions, iris2_w_train = my_logistic_regression(iris2_train_matrix, iris2_train_labels, iris2_train_matrix)

#calculate 0-1 loss
iris2_test_score1 = accuracy_score(iris2_train_labels, iris2_test_predictions)
print (1 - iris2_test_score1)


#Calling function on training - test set
iris2_test_predictions, iris2_w_test = my_logistic_regression(iris2_train_matrix, iris2_train_labels, iris2_test_matrix)

#calculate 0-1 loss
iris2_test_score2 = accuracy_score(iris2_test_labels, iris2_test_predictions)
print (1 - iris2_test_score2)

## plots

def line_seg(w,x):
    """ Calculate line segment from parameters """
    y = -x*(w[1]/w[2]) - w[0]/w[2]

    return y

## plots with line segments
# assigning colour based on class
colourbyclass1 = ['b' if i==1 else 'r' for i in Iris2D1_train[:, 2]]
colourbyclass2 = ['b' if i==1 else 'r' for i in Iris2D1_test[:, 2]]

# plot Iris2D1
plt.scatter(Iris2D1_train[:, 0], Iris2D1_train[:, 1], color=colourbyclass1)
plt.scatter(Iris2D1_test[:, 0], Iris2D1_test[:, 1], color=colourbyclass2)
plt.plot([4,8], [line_seg(iris1_w_train, 4), line_seg(iris1_w_train, 8)])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Iris2D1 Training & Test data', fontweight='bold')
#plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/E6-3.png', bbox_inches='tight', dpi=500)
plt.close()

# assigning colour based on class afor 2nd set
colourbyclass1 = ['b' if i==1 else 'r' for i in Iris2D2_train[:, 2]]
colourbyclass2 = ['b' if i==1 else 'r' for i in Iris2D2_test[:, 2]]

# plot Iris2D2
plt.scatter(Iris2D2_train[:, 0], Iris2D2_train[:, 1], color=colourbyclass1)
plt.scatter(Iris2D2_test[:, 0], Iris2D2_test[:, 1], color=colourbyclass2)
plt.plot([4,8], [line_seg(iris2_w_train, 4), line_seg(iris2_w_train, 8)])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Iris2D2 Training & Test data', fontweight='bold')
#plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 5/E6-4.png', bbox_inches='tight', dpi=500)
plt.close()


### Exercise 7

#Theoretical