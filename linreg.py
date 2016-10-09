import numpy as np
from numpy import genfromtxt

#number of instances
M = 486
#number of attributes
N = 2
#learning rate (alpha)
alpha = .001
#number of iterations
iterations = 1000


#pull in csv (prescaled)
csv = genfromtxt('sample-file.csv', delimiter=',')

# create our data matrix (the first two rows)
data = csv[:,0:N]

# create output matrix (column 3) then send it to an array as ints
preoutput = csv[:,N:N+1]
outputs = np.squeeze(np.asarray(preoutput.astype(int)))

# initialize weights to 0
weights = [0]*(N+1)

# Prepend a column of ones (transpose array of ones and then concat to our data matrix)
ones = np.array([[1]*M]).T
datapre = np.concatenate((ones, data), axis=1)


# compute the model output (multiply our data matrix by our weights to get an output matrix of predictions)
hypothesis = datapre.dot(weights)

# gradient descent
for i in xrange(iterations):
    #get the difference for each instance
    D_i = hypothesis - outputs
    # gradient descent weight update
    for j in xrange(N+1):
        a = datapre[:,j].dot(D_i)
        sum = a.sum()
        weights[j] = weights[j] - (alpha * sum)
        #update hypothesis
        hypothesis = datapre.dot(weights)

#print our weights
print "Theta0:",weights[0],"Theta1:",weights[1],"Theta2:",weights[2]
