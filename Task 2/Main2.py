import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import numpy as np

# calculates euclidean distance between vectors 1 and 2 which can have any number of dimmensions (as long as they both have same number)
def compute_euclidean_distance(vec_1, vec_2):
    distance = np.square(np.sum((vec_1 - vec_2) ** 2))
    return distance

# creates k points with coordinates in the range of the data set passed
def initialise_centroids(dataset, k=2):

    centroid_min =  math.floor(np.amin(dataset))
    centroid_max = math.ceil(np.amax(dataset))
    centroids = []
    centroids = np.array([random.randint(centroid_min ,  centroid_max), random.randint(
        centroid_min,  centroid_max), random.randint(centroid_min , centroid_max), random.randint(centroid_min ,  centroid_max)])
    for i in range(k-1):
        centroids = np.vstack([centroids, np.array([random.randint(centroid_min ,  centroid_max), random.randint(
            centroid_min,centroid_max), random.randint(centroid_min ,  centroid_max), random.randint(centroid_min,
                                                                                    centroid_max)])])  # create centroid as a 4D vector, with random values for each in the range established above and append to matrix
        return centroids
# once points are clustered, centroids need to be calculated as the center of that cluster
def reCalcCentroids(dataset, assignments, numOfCentroids):
    tally = np.zeros(numOfCentroids) # an array that tracks the number of points assigned to each centroid - needed to find mean
    avg = np.zeros((numOfCentroids, 4)) # create a 4xNumCentroids matrix
    # iterates through each data point
    for i in range(dataset.shape[0]):
        tally[assignments[i]] = tally[assignments[i]] + 1 # increase tally for whichever centroid data point i is assigned to

        # sums up the total for each of the 4 dimensions - this is later divided by number of points to find the mean
        for j in range(4):
            avg[assignments[i]][j] = avg[assignments[i]][j] + dataset[i][j]
    for i in range(avg.shape[0]):
        if tally[i] != 0: # if the tally is zero then we don't want to perform the division because we'd have to divide by 0 which = problems
            # divide each dimensions total by the number of points to get the mean
            for j in range(4):
                avg[i][j] = avg[i][j] / tally[i]
    return avg

# for a given point, returns the index in centroids array of the centroid it is closest to
def assignPoint(point, centroids):
    values = np.array([compute_euclidean_distance(point, centroids[0])]) # create new array values that will store distances to all centroids - calculates distance to centroid 0
    for i in range(centroids.shape[0] - 1): # for the remaining centroids
        values = np.vstack(
            [values, compute_euclidean_distance(point, centroids[i + 1])]) # calculate the euclidean distance between the point and centroid i(+ 1) and then append that distance to the values array
    index_min = np.argmin(values) # the index of the centroid that is closest is the index of the lowest number in the values array
    return index_min

def kmeans(dataset, k):
    sumDistances = [] # keeps track of the objective and is appended each cycle with the current sum distances from centroids
    centroids = initialise_centroids(dataset, k) # creates k centroids in space ocupied by the dataset
    assOld = np.empty(dataset.shape[0], dtype=int) # initialise array
    Exit = False # while loop will continue to loop until this flag is true
    while Exit == False:
        ##
        ass = np.empty(dataset.shape[0], dtype=int)
        for i in range(dataset.shape[0]):
            ass[i] = assignPoint(dataset[i], centroids)
        assNew = ass # runs a cycle and assigns all of the points to a centroid
        ##
        if np.array_equal(assOld, assNew): # if no points have been assigned to a different centroid then the clustering is complete and the program can exit
            Exit = True
        else:
            assOld = assNew # if not, store the new assignments to check against after the next iteration
        for item in centroids: # for each centroid
           if np.count_nonzero(item) == 0: #if the centroid = [0, 0, 0, 0]
                return kmeans(dataset, k) # re-run the process because there has been an error in the initial assignments
        centroids = reCalcCentroids(dataset, assNew, k) #  re-calculate the centroids to be the mean of their cluster
        sumDistances.append(objFunction(dataset, centroids, assOld)) # append the array with the sum of the distance of all points to their assigned centroid - this is graphed later to show the progress of the clustering
    cluster_assigned = assNew # return the current assignments as the final clusters
    plotLine(sumDistances) # make a plot of the objective function over each iteration
    return [centroids, cluster_assigned]

# works out the sum of the distances from their assigned centroid
def objFunction(dataset, centroids, assignments):
    sumDistance = 0 # initialise total as 0
    for i in range(dataset.shape[0]): # for each data point
        sumDistance += compute_euclidean_distance(
            dataset[i], centroids[assignments[i]]) # calculate the data points euclidean distance from the centroid it is assigned to - found from the assignments array
    return sumDistance

# plots objective function over each iteration of k means
def plotLine(data):
    figur = plt.figure()
    # x = range from 0 to number of cycles of k means that were run
    x = list(range(0, len(data)))
    # y = the sum distances for each cycle
    y = data
    #plot data points
    plt.plot(x, y, 'o')
    #plot the line to connect these points
    plt.plot(x, y)
    #label axes
    ax1 = figur.add_subplot()
    ax1.set_ylabel('Sum Distance to Centroids')
    ax2 = figur.add_subplot()
    ax2.set_xlabel('Iteration')


# plots scatter graph of data points with each cluster's points appearing in a different colour - centroids appear as black points
def plotGraph(dataset, centroids, cluster_assigned, y_Axis):
    fig = plt.figure()
    dicts = {0: 'height', 1: 'tail length',
             2: 'leg length', 3: 'nose circumference'} # dictionary used for labelling y axis
    X = dataset[dicts[0]].to_numpy() # extract heights from dataset
    Y = dataset[dicts[y_Axis]].to_numpy() # extract y axis variable by referencing dicts (could be 'tail length', 'leg length' or 'node circumfrence')
    # each cluster should have different colour to visual distinguish them so each data point checked to see which cluster it belongs to, then the colour of that cluster is used when plotting point
    # this can handle up to 6 clusters after which point we run out of colours and any additional will just be yellow
    for i in range(X.shape[0]):
        if cluster_assigned[i] == 0:
            col = 'b'
        elif cluster_assigned[i] == 1:
            col = 'g'
        elif cluster_assigned[i] == 2:
            col = 'r'
        elif cluster_assigned[i] == 3:
            col = 'c'
        elif cluster_assigned[i] == 4:
            col = 'm'
        else:
            col = 'y'
        plt.scatter(X[i], Y[i], color=col) # plot point i with colour of its assigned cluster
    #plots centroids in black
    for i in range(centroids.shape[0]):
        plt.scatter(centroids[i][0], centroids[i][y_Axis], color='black')
    # label axes with height on x axis and y axis var found from dictionary lookup
    ax1 = fig.add_subplot()
    ax1.set_ylabel(dicts[y_Axis])
    ax2 = fig.add_subplot()
    ax2.set_xlabel(dicts[0])
    plt.show()

# actual code runs from here:
# imports the data from the csv file
dataset = pd.read_csv('Task2 - dataset - dog_breeds.csv')
dataMatrix = dataset.to_numpy()


# gets a value of k from the user any value of k works but if k > 6 not enough unique colours to plot
choice = int(input('Enter value for k\n'))
ans = kmeans(dataMatrix, choice) # returns array where 0th item is centroids of clusters and 1st item is array of assignments for each point to a cluster

# gets users choice for vaiable to be plotted on Y axis, height is always on x axis - choice is an int which is passed to plotGraph where dict lookup gets text string
choice = int(input('Enter number choice for data on Y-Axis:\n   1. Tail Length\n   2.Leg Length\n   3.Nose Circumference\n'))
plotGraph(dataset, ans[0], ans[1], choice) # plot graph of clusters