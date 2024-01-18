
import numpy as np
# X          - single array/vector
# y          - single array/vector
# theta      - single array/vector
# alpha      - scalar
# iterations - scarlar

def gradientDescent(X, y, theta, alpha, numIterations):
    '''
    # This function returns a tuple (theta, Cost array)
    '''
    m = len(y)
    arrCost =[];
    transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
    for iteration in range(0, numIterations):
        ################PLACEHOLDER3 #start##########################
        #: write your codes to update theta, i.e., the parameters to estimate. 
	# Replace the following variables if needed 
        f = np.dot(X,theta)
        change = alpha / m * np.dot((f-y),X)
        theta = np.subtract(theta, change)  # or theta = theta - alpha * gradient
        ################PLACEHOLDER3 #end##########################

        ################PLACEHOLDER4 #start##########################
        # calculate the current cost with the new theta; 
        atmp = np.mean(np.square(y-f)) 
        arrCost.append(atmp) 
     
        ################PLACEHOLDER4 #end##########################
    return theta, arrCost

