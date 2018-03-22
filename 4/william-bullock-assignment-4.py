#### William Bullock - IDS Assignment 4

### Preliminaries

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans


### Exercise 1:

diatoms = np.loadtxt('diatoms.txt')

##First Plot

x_diatom1 = diatoms[0][[i for i in range(180) if i%2 == 0]]
y_diatom1 = diatoms[0][[i for i in range(180) if i%2 != 0]] #forming separate lists for the X and Y coordinates for the first diatom of the dataset

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)  #begin the plot

ax1.plot(x_diatom1, y_diatom1, "o", color='b') # plotting the lists above

ax1.axis('equal')
ax1.set_title('Landmark points from the first diatom of the dataset', fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y') #visual tweaks

#fig1.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 4/diatomsolo.png', bbox_inches='tight', dpi=1000)
plt.close(fig1)

##Second Plot:

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)  #begin the plot

for diatom in diatoms:
    x_diatom = diatom[[i for i in range(180) if i % 2 == 0]]
    y_diatom = diatom[[i for i in range(180) if i % 2 != 0]] #forming separate lists for the X and Y coordinates for all diatoms of the dataset

    ax2.plot(x_diatom, y_diatom, "o", color='b')  # plotting the lists above overlapping, one by one.

ax2.axis('equal')
ax2.set_title('Landmark points for all diatoms of the dataset', fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')  # visual tweaks

#fig2.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 4/diatomsall.png', bbox_inches='tight', dpi=1000)
plt.close(fig2)

## Exercise 2:


def plotdiatom(diatomdata, title):
    """Plots all the diatoms within a dataset."""
    blues = plt.get_cmap('Blues') #import blue colour spectrum
    colours = [blues(0.0), blues(0.2), blues(0.4), blues(0.6), blues(0.8), blues(1.0)] #separate blue spectrum into 5 points.
    colourindex = -1 #initialize colour index at -1
    for diatom in diatomdata:
        colourindex += 1 #add 1 to colour index

        x_diatom = diatom[[i for i in range(180) if i % 2 == 0]]
        y_diatom = diatom[[i for i in range(180) if i % 2 != 0]]  # forming separate lists for the X and Y coordinates for all diatoms of the dataset
        plt.plot(x_diatom, y_diatom, "o", color=colours[colourindex])  # plotting the lists above overlapping, one by one, usingthe colour inex to choose the current place on the blue spectrum.

    plt.axis('equal')
    plt.title(title, fontweight='bold')
    plt.xlabel('X')
    plt.ylabel('Y') #visual tweaks


diatoms_mean = np.mean(diatoms, 0) #find the landmark points for the mean diatom

Sigma = np.cov(diatoms.T) #transposes the data and forms a covariance matrix
evals, evecs = np.linalg.eig(Sigma) #calculates eigenvalues and eigenvectors

##plot for Principle Component 1

e1 = evecs[:, 0]  #taking the first eigenvector
lambda1 = evals[0] #calculating lambda
std1 = np.sqrt(lambda1) #calculating standard deviation

diatoms_along_pc1 = np.zeros((5, 180))# create an empty matrix of 5 cells with space for 180 landmark points each.

for i in range(5):
    diatoms_along_pc1[i, :] = diatoms_mean + (i - 2)*std1*e1 #fill the matrix with the calculated diatom cells representing the PC.- uses equations given in assignment

plotdiatom(diatoms_along_pc1, 'Movement along Principle component 1') # call my plot function on the matrix for the PC.

#plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 4/DFpca1.png', bbox_inches='tight', dpi=1000)
plt.close()

##plot for Principle Component 2

e2 = evecs[:, 1]  #taking the first eigenvector
lambda2 = evals[1] #calculating lambda
std2 = np.sqrt(lambda2) #calculating standard deviation

diatoms_along_pc2 = np.zeros((5, 180))# create an empty matrix of 5 cells with space for 180 landmark points each.

for i in range(5):
    diatoms_along_pc2[i, :] = diatoms_mean + (i - 2)*std2*e2 #fill the matrix with the calculated diatom cells representing the PC.- uses equations given in assignment

plotdiatom(diatoms_along_pc2, 'Movement along Principle component 2') # call my plot function on the matrix for the PC.

#plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 4/DFpca2.png', bbox_inches='tight', dpi=1000)
plt.close()

##plot for Principle Component 3

e3 = evecs[:, 2]  #taking the first eigenvector
lambda3 = evals[2] #calculating lambda
std3 = np.sqrt(lambda3) #calculating standard deviation

diatoms_along_pc3 = np.zeros((5, 180))# create an empty matrix of 5 cells with space for 180 landmark points each.

for i in range(5):
    diatoms_along_pc3[i, :] = diatoms_mean + (i - 2)*std3*e3 #fill the matrix with the calculated diatom cells representing the PC.- uses equations given in assignment

plotdiatom(diatoms_along_pc3, 'Movement along Principle component 3') # call my plot function on the matrix for the PC.

#plt.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 4/DFpca3.png', bbox_inches='tight', dpi=1000)
plt.close()


### Exercise 3:
# I did implement PCA fom scratch in assignemnt 3, but I was docked marks for it being imperfect, due to this I have opted to use scikit - learn here.

toydata = np.loadtxt('pca_toydata.txt')


pca = decomposition.PCA(n_components=2) # Primes PCA to downscale the data to two dimensions
toypca = pca.fit_transform(toydata)
#Centers and stnadardizes the data, forms a covariance matrix, calculates eigenvectors and eigenvalues and returns the n (here 2) PCs that elicit the most variance in the data

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)  #begin the plot

ax3.scatter(toypca[:, 0], toypca[:, 1], color='r') #plots the output from the PCA


ax3.set_title('First and Second Principle Components for Toydata', fontweight='bold')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2') #visual presentation tweaks


#fig3.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 4/TOYpca1.png', bbox_inches='tight', dpi=1000)
plt.close(fig3)



toydata_less = toydata[:100] #removing the final two data points


pca = decomposition.PCA(n_components=2) # Primes PCA to downscale the data to two dimensions
toy_pca_less = pca.fit_transform(toydata_less)
#Centers and stnadardizes the data, forms a covariance matrix, calculates eigenvectors and eigenvalues and returns the n (here 2) PCs that elicit the most variance in the data

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)  #begin the plot

ax4.scatter(toy_pca_less[:, 0], toy_pca_less[:, 1], color='r') #plots the output from the PCA


ax4.set_title('1st and 2d PCs for Toydata, with final two data points removed.', fontweight='bold')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2') #visual presentation tweaks


#fig4.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 4/TOYpca2.png', bbox_inches='tight', dpi=1000)
plt.close(fig4)

### Exercise 4:


data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',') #Loading data

data_train_no_class = [i[:13] for i in data_train] #removes class data from IDSWeedCropTrain
data_train_class_only = data_train[:, -1] #Splitting the input variables and labels

data_train_array = np.vstack(data_train_no_class) #reshape nested lists of lists into arrays

pca = decomposition.PCA(n_components=2) # primes PCA to downscale the data to two dimensions

data_train_pca = pca.fit_transform(data_train_array)
#Centers and stnadardizes the data, forms a covariance matrix, calculates eigenvectors and eigenvalues and returns the n (here 2) PCs that elicit the most variance in the data

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)  #begin the plot

colorbyclass = ['b' if i==1 else 'r' for i in data_train_class_only] # ordered list of 'blue' / 'red' based on crop / weed class

plt.scatter(data_train_pca[:, 0], data_train_pca[:, 1], color=colorbyclass)
#plots the output from the PCA, assigns colour based on aformentioned list


## K-clustering

x_train = data_train[:, :-1] # separate the data into a list of lists

startingPoint = np.vstack((x_train[0, ], x_train[1, ]))  # Initializing from the first two values in the dataset(as directed but not advised)
kmeans = KMeans(n_clusters=2, n_init=1, init=startingPoint, algorithm='full').fit(x_train) # uses scikit-learn to cluster the data to two centroids.


train_kmeans = kmeans.cluster_centers_ # assign the clusters to a vector
train_kmeans_pca = pca.transform(train_kmeans) # uses the formely extablished pca fit to apply the same transformation to the clusters.

plt.scatter(train_kmeans_pca[1][0], train_kmeans_pca[1][1], marker='x', s=50, color='r', label='Weed') #adding each cluster to the plot.
plt.scatter(train_kmeans_pca[0][0], train_kmeans_pca[0][1], marker='x', s=50, color='b', label='Crop')


ax5.axis('equal')
ax5.set_title('1st and 2d PCs for weeds and crops from dataset', fontweight='bold')
ax5.set_xlabel('PC1')
ax5.set_ylabel('PC2')
ax5.legend(loc='lower right') #visual presentation tweaks

#fig5.savefig('/home/will/GIT/pycharm projects/IDS/Assignment 4/WeedCroppca.png', bbox_inches='tight', dpi=1000)
plt.close(fig5)
