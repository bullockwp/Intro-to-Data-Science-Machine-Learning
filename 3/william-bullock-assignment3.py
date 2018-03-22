#### William Bullock - IDS Assignment 3

### Preliminaries
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

#reading in the datasets, as specified in question paper:

data = np.loadtxt('murderdata2d.txt')

data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
data_test = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

### Exercise 1:

## 1a) PCA Function

def pca(data):
    """ Performs Principle Component Analysis on a dataset, after centering.
     Returns a list list of eigenvectors and their eigenvalues. The list is sorted by eigenvalue so that it is monotonically decreasing."""
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=False).fit(data)
    data_center = scaler.transform(data)  # Centers the data
    data_trans = np.ndarray.transpose(data_center)  # imports and transposes data
    sigma = np.cov(data_trans)  # forms a covariance matrix for the data
    evals, evecs = np.linalg.eig(sigma)  # collects eigenvalues and eigenvectors for the covariance matrix
    PCs = zip([i for i in np.ndarray.tolist(evecs)], evals)  # Assigns each eigenvalue, its corresponding eigenvectors
    PCs.sort(key=lambda x: abs(x[1]), reverse=True)  #Sorts list of PCs and eigen vectors by the magnitude of the absolute value of the eigenvalue in descending order.
    return PCs

## 1b) Plot 1

murder_data = np.loadtxt('murderdata2d.txt')
murder_pca = pca(murder_data) # calling the above function

scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=False).fit(murder_data)
murder_data_center = scaler.transform(murder_data)  # Centers the data
#murder_data_trans = np.ndarray.transpose(murder_data_center)

pc_unemployment = [i[0] for i in murder_data_center]  # extracting all percentages of unemployment in order into a list

mpapm = [i[1]for i in murder_data_center]  # extracting all murders per annum per 100,000,000 value in oder into a list

murder_evecs = [i[0]for i in murder_pca]  # extracting the previously found eigenvectors

murder_evals = [i[1] for i in murder_pca]  # extracting the previously found eigenvalues

sd0 = np.sqrt(murder_evals[0])
sd1 = np.sqrt(murder_evals[1])  # finding the standard deviations

eigen1 = tuple(murder_evecs[0])
eigen2 = tuple(murder_evecs[1])  # separating the co-ordinates of the individual eigenvectors

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)  #begin the plot

ax1.plot(pc_unemployment, mpapm, "o", color='b')
ax1.plot(0, 0, "x", color='r', mew='2') #plotting the mean

ax1.plot([0, sd1*eigen1[1]], [0, sd1*eigen1[0]], 'r')
ax1.plot([0, sd0*eigen2[1]], [0, sd0*eigen2[0]], 'r') #plotting the PCs... the eigenvectors scaled by sd.

ax1.set_title('Centered Unemployment in Relation to Murder Rate', fontweight='bold')
ax1.set_xlabel('Centered Unemployment (%)')
ax1.set_ylabel('Murders Per Annum Per 1,000,000 (People)')
ax1.set_xlim(-20, 25) # Labels and presentation tweaks

plt.show()
#fig1.savefig('/home/will/PycharmProjects/IDS/Assignment 3/Scatter.png', bbox_inches='tight', dpi=1000)
plt.close()

## 1c) Plots 2 & 3

pest_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

pest_pca = pca(pest_train) # running PCA on pest data

pest_evals = [i[1] for i in pest_pca]  # extracting eigenvalues into a list

# plot 2

fig2 = plt.figure() # Variance vs PC index plot
ax2 = fig2.add_subplot(111)
ax2.plot(pest_evals)

ax2.set_title('Variance vs PC index plot', fontweight='bold')
ax2.set_xlabel('PCs index')
ax2.set_ylabel('Projected Variance')
ax2.set_xlim(1, 14) # Labels and presentation tweaks
ax2.set_xticks(np.arange(1, 15, 1)) # Labels and presentation tweaks

plt.show()
#fig2.savefig('/home/will/PycharmProjects/IDS/Assignment 3/VarVsPC.png', bbox_inches='tight', dpi=1000)
plt.close()

# plot 3

c_var = np.cumsum(pest_evals/np.sum(pest_evals))  #Calculating cumulative normalized variance

fig3 = plt.figure() # Variance vs PC index plot
ax3 = fig3.add_subplot(111)
ax3.plot(c_var)

ax3.set_title('Cumulative Variance vs PC index plot', fontweight='bold')
ax3.set_xlabel('PC index')
ax3.set_ylabel('Cumulative Variance')
ax3.set_xlim(1, 14)
ax3.set_xticks(np.arange(1, 15, 1)) # Labels and presentation tweaks

plt.show()
#fig3.savefig('/home/will/PycharmProjects/IDS/Assignment 3/CuVarVsPC.png', bbox_inches='tight', dpi=1000)
plt.close()

### Exercise 2: MDS

pest_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')


def mds(data, d):
    """ Uses principle component analysis to perform multidimensional scaling,
      data = input data set
      d = desired number of dimensions"""
    datapca = pca(data)  # running PCA on pest data
    pc_dbest = []
    for i in range(0, d, 1):
        pc_dbest.append(datapca[i]) #finds the d eignevectors with the highest eigenvalues

    evecs = [i[0] for i in pc_dbest] # collects the eigenvectors from the above list

    # evals = [i[1] for i in pc_dbest]
    #sd0 = np.sqrt(evals[0])  # finding the standard deviations
    #sd1 = np.sqrt(evals[1])
    #eigen1 = [sd1 * i for i in evecs[0]]
    #eigen2 = [sd0 * i for i in evecs[1]]  # here i thought about scaling by Sd, but decided not to, because the lecture slides didn't mention scaling.

    eigen_matrix = np.array([i[::-1] for i in evecs])  # forms a matrix from the lists of eigenvectors
    eigen_trans = np.ndarray.transpose(eigen_matrix) # transposes eigenvector array

    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=False).fit(data)
    data_center = scaler.transform(data)  # Centers the data
    d_dimension_matrix = np.dot(data_center, eigen_trans) #finds the dot product of the input data and its transposed d best eigenvectors

    return d_dimension_matrix

pest_mds = mds(pest_train, 2) # calling the mds function on the IDSWeedCropTrain data, to scale down to 2 dimensions

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)  #begin the plot

x = list(pest_mds[:, 0])
y = list(pest_mds[:, 1])  # separating data in the matrix into co-ordinates

ax4.plot(x, y, "o", color='b') #plotting the 2D-scaled dataset

ax4.plot(0, 0, "x", color='r', mew='2') #plotting the mean

ax4.set_title('IDSWeedCropTrain as scaled to 2 dimensions, from 14', fontweight='bold')
ax4.set_xlabel('Co-ordinate 1')
ax4.set_ylabel('Co-ordinate 2') # Labels and presentation tweaks

plt.show()
#fig4.savefig('/home/will/PycharmProjects/IDS/Assignment 3/MDSPest.png', bbox_inches='tight', dpi=1000)

### Exercise 3:

#Splitting the input variables and labels, as specified in question paper:

x_train = data_train[:, :-1]
y_train = data_train[:, -1]

x_test = data_test[:, :-1]
y_test = data_test[:, -1]

## Solution 1: scikit-learn method

startingPoint = np.vstack((x_train[0, ], x_train[1, ]))  # Initializing from the first two values in the dataset(as directed but not advised)
kmeans = KMeans(n_clusters=2, n_init=1, init=startingPoint, algorithm='full').fit(x_train) #uses scikit-learn to cluster the data to two centroids.

print(kmeans.cluster_centers_)
