#### William Bullock - IDS Assignment 1

###preliminaries and imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

dataset = np.loadtxt("/home/will/PycharmProjects/IDS/Assignment 1/smoking.txt")

###Exercise 1

def meanFEV1(data):
    '''returns the FEV1 scores for non-smokers and smokers in a dataset.'''

    non_smokers = []
    smokers = [] # create empty lists
    for line in data: # Iterate over the data set, entry by entry...
        if line[4] == 0: # if the 5th value in a line is 0 (indicative of a nonsmoker)
            non_smokers.append(line) # append the whole line from the dataset to the non_smokers list.
        else:
            smokers.append(line) # otherwise append the entry to the smokers list

    nsfev1 = [i[1] for i in non_smokers]
    sfev1= [i[1] for i in smokers] # use list comprehension to extract the FEV1 value from each finished list

    return nsfev1, sfev1 # return the means

nosfev1, yessfev1 = meanFEV1(dataset) # call the function

nosmean = np.mean(nosfev1)
yessmean = np.mean(yessfev1) # use numpy to find the mean FEV1 for each list

print nosmean
print yessmean

## ALTERNATIVELY - Its possible to find the same results using np.where and np.take followed by np.mean

###Exercise 2

allfev1s = [nosfev1, yessfev1] # Collecting the datafor the plot as a tuple of two lists.

fig = plt.figure()# beginninghte figure

ax = fig.add_subplot(111)
ax.boxplot(allfev1s) #plotting the list of tuples (as a boxplot)
ax.set_title('FEV1 Values for Non-Smokers and Smokers', fontweight='bold')
ax.set_xlabel('Smoking? 0 = no, 1 = yes')
ax.set_xticklabels([0, 1])
ax.set_ylabel('FEV1 (L)') # Labels and presentation tweaks

#plt.show()

#fig.savefig('/home/will/PycharmProjects/IDS/Assignment 1/BoxPlots.png', bbox_inches='tight', dpi=1000)

###Exercise 3

def hyptest(data):
    ''' Performs a two-sided T test on two independant ; comparing the mean FEV1 values for smokers or non-smokers in a dataset,
    with null hypothesis that the two means are equal.
    Returns true or or false, rejecting or accepting the null hypothesis respectively.'''

    non_smokers = []
    smokers = []  # create empty lists
    for line in data:  # Iterate over the data set, entry by entry...
        if line[4] == 0:  # if the 5th value in a line is 0 (indicative of a nonsmoker)
            non_smokers.append(line)  # append the whole line from the dataset to the non_smokers list.
        else:
            smokers.append(line)  # otherwise append the entry to the smokers list

    nsfev1 = [i[1] for i in non_smokers]
    sfev1 = [i[1] for i in smokers]  # use list comprehension to extract the FEV1 value from each finished list

    t_value, p_value = scipy.stats.ttest_ind(nsfev1, sfev1, equal_var=False) #use scipy to perform Welch's t-test giving a t_value and P_value, the t_value is ignored.

    if p_value < 0.05: # returns true or false, whether or not the p_value is less than the significance value of 0.05
        return True
    else:
        return False

nullhyp = hyptest(dataset) # calling the function

print nullhyp

###Exercise 4

#Creating vectors

agelist = [i[0] for i in dataset] # extracting all ages in order into a list

fevlist = [i[1]for i in dataset] # extracting all FEV1 value in oder into a list

#The Scatter Plot

fig2 = plt.figure() # beginning the figure

trend = np.polyfit(agelist, fevlist, 1)
trendline = np.poly1d(trend) # calculating a trendline

ax2 = fig2.add_subplot(111)

ax2.plot(agelist, fevlist, "o", agelist, trendline(agelist), "r") #plotting the scatter plot and the trendline

ax2.set_title('Age Plot Against FEV1 Values', fontweight='bold')
ax2.set_xlabel('Age (years)')
ax2.set_ylabel('FEV1 (L)') # Labels and presentation tweaks

#plt.show()

#fig2.savefig('/home/will/PycharmProjects/IDS/Assignment 1/AgeFEV1Scatter.png', bbox_inches='tight', dpi=1000)

#The Correlation

def corr(x,y):
    '''Finds the Spearman's rank correlation coefficient between two vectors of equal length'''

    correlation, Pvalue = scipy.stats.spearmanr(x,y) # uses predefined scipy fucntion for spearmansrank correlation coefficient,
    # generates a correlation value and a P value, the P value is ignored.

    return correlation

print corr(agelist,fevlist)

## ALTERNATIVELY : one could define their own spearnmans rank function using the formula:
# correlation = p(r (x), r (y ))
#in that case one would need to calculate the r transformation variable and use it to generate the r(x) and r(y) rank vectors.

###Exercise 5

#Creating the vectors

non_smokers_age = []
smokers_age = []  # create empty lists
for line in dataset:  # Iterate over the data set, entry by entry...
    if line[4] == 0:  # if the 5th value in a line is 0 (indicative of a nonsmoker)
        non_smokers_age.append(line[0])  # append the whole line from the dataset to the non_smokers list.
    else:
        smokers_age.append(line[0])  # otherwise append the entry to the smokers list


#The Histograms


fig3 = plt.figure()
bins = np.linspace(0, 20, 20) # setting x axis limits and bin size

plt.hist(non_smokers_age, bins, alpha=0.5, label='Non-Smokers')
plt.hist(smokers_age, bins, alpha=0.5, label='Smokers') #plotting the histograms

plt.legend(loc='upper left')

plt.title('Histograms of Subject Ages', fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Frequency') # Labels and presentation tweaks

plt.show()

#fig3.savefig('/home/will/PycharmProjects/IDS/Assignment 1/Histograms.png', bbox_inches='tight', dpi=1000)

