import numpy as np
import matplotlib.pyplot as plt
from getDataset import getDataSet
from sklearn.linear_model import LogisticRegression
 

# Starting codes

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# step 1: generate dataset that includes both positive and negative samples,
# where each sample is described with two features.
# 250 samples in total.

[X, y] = getDataSet()  # note that y contains only 1s and 0s,

# create figure for all charts to be placed on so can be viewed together
fig = plt.figure()


def func_DisplayData(dataSamplesX, dataSamplesY, chartNum, titleMessage):
    idx1 = (dataSamplesY == 0).nonzero()  # object indices for the 1st class
    idx2 = (dataSamplesY == 1).nonzero()
    ax = fig.add_subplot(1, 3, chartNum)
    # no more variables are needed
    plt.plot(dataSamplesX[idx1, 0], dataSamplesX[idx1, 1], 'r*')
    plt.plot(dataSamplesX[idx2, 0], dataSamplesX[idx2, 1], 'b*')
    # axis tight
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_title(titleMessage)


# plotting all samples
func_DisplayData(X, y, 1, 'All samples')

# number of training samples
nTrain = 120

######################PLACEHOLDER 1#start#########################
# write you own code to randomly pick up nTrain number of samples for training and use the rest for testing.
# WARNIN: do not use the scikit-learn or other third-party modules for this step

#maxIndex = len(X)
#randomTrainingSamples = np.random.choice(maxIndex, nTrain, replace=False)
trainX = X[train_shuffled_indicies, :]   #  training samples
trainY = y[train_shuffled_indicies, :]   # labels of training samples    nTrain X 1
 
testX =  X[test_shuffled_indicies, :]  # testing samples               
testY =  y[test_shuffled_indicies, :] # labels of testing samples     nTest X 1

####################PLACEHOLDER 1#end#########################

# plot the samples you have pickup for training, check to confirm that both negative
# and positive samples are included.
func_DisplayData(trainX, trainY, 2, 'training samples')
func_DisplayData(testX, testY, 3, 'testing samples')

# show all charts
plt.show()


#  step 2: train logistic regression models


######################PLACEHOLDER2 #start#########################
# in this placefolde r you will need to train a logistic model using the training data: trainX, and trainY.
# please delete these coding lines and use the sample codes provided in the folder "codeLogit"
logReg = LogisticRegression(fit_intercept=True, C=1e15) # create a model
logReg.fit(trainX, trainY)# training
coeffs = logReg.coef_ # coefficients
intercept = logReg.intercept_ # bias 
bHat = np.hstack((np.array([intercept]), coeffs))# model parameters
######################PLACEHOLDER2 #end #########################


# step 3: Use the model to get class labels of testing samples.
 
######################PLACEHOLDER3 #start#########################
# codes for making prediction, 
# with the learned model, apply the logistic model over testing samples
# hatProb is the probability of belonging to the class 1.
# y = 1/(1+exp(-Xb))
# yHat = 1./(1+exp( -[ones( size(X,1),1 ), X] * bHat )); ));
# WARNING: please DELETE THE FOLLOWING CODEING LINES and write your own codes for making predictions
xHat = np.concatenate((np.ones((testX.shape[0], 1)), testX), axis=1)  # add column of 1s to left most  ->  130 X 3
negXHat = np.negative(xHat)  # -1 multiplied by matrix -> still 130 X 3
hatProb = 1.0 / (1.0 + np.exp(negXHat * bHat))  # variant of classification   -> 130 X 3
# predict the class labels with a threshold
yHat = (hatProb >= 0.5).astype(int)  # convert bool (True/False) to int (1/0)
#PLACEHOLDER#end

######################PLACEHOLDER 3 #end #########################


# step 4: evaluation
# compare predictions yHat and and true labels testy to calculate average error and standard deviation
testYDiff = np.abs(yHat - testY)
avgErr = np.mean(testYDiff)
stdErr = np.std(testYDiff)

print('average error: {} ({})'.format(avgErr, stdErr))

######################PLACEHOLDER4 #start#########################
#Please add codes to calculate the ROC curves and AUC of the trained model over all the testing samples
def pred_function(X, theta):
    return 1/(1 + np.exp(- np.dot(X, theta)))
testp = pred_function(testX, theta)


y_predict = clf.predict(testX)
true_y = testY.ravel()

fpr, tpr, thresholds = roc_curve(true_y,y_predict)

plt.plot(fpr, tpr, label= 'ROC curve (AUC = %0.2f)' % auc(fpr,tpr))
plt.plot([0, 1], [0, 1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
######################PLACEHOLDER4 #end #########################