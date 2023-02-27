#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random
import math


# ## Bayes classifier functions to implement
#             
# The lab descriptions state what each function should do.


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0] # Rows in labels is the number of data points
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    # classes is a vector of the unique classes in labels
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    prior = np.zeros((Nclasses,1))

    how_many_in_class = np.zeros((Nclasses))    

    # TODO: compute the values of prior for each class!
    # ==========================
    """
    for class_iter in classes:
        for i in range(0,Npts):
            if labels[i] == class_iter:
                how_many_in_class[class_iter] = how_many_in_class[class_iter] + 1
            prior[class_iter] = float (how_many_in_class[class_iter]/Npts)    
    """
    # ==========================

    # ==========================
    # TODO: handle the W argument!
    for class_iter in classes:
        for data in range(0,Npts):
            if labels[data] == class_iter:
                how_many_in_class[class_iter] = how_many_in_class[class_iter] + W[data]
            prior[class_iter] = float (how_many_in_class[class_iter]/np.sum(W))
    
    # ==========================

    return prior

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):

    assert(X.shape[0]==labels.shape[0])
    # Npts is the number of rows in X (number of data points)
    # Ndims is the number of columns in X (number of dimensions)
    Npts,Ndims = np.shape(X)

    # classes is a vector of the unique classes in labels
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    occurance_in_class = np.zeros((Nclasses))


    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    # Mu is the mean of the data points in each class
    # Sigma is the covariance matrix of the data points in each class

    # Se equation (8)
    """ 
    for class_iter in classes:
        for i in range(0,Npts):
            if labels[i] == class_iter:
                mu[class_iter] = mu[class_iter] + X[i]
                occurance_in_class[class_iter] = occurance_in_class[class_iter] + 1
        mu[class_iter] = mu[class_iter]/occurance_in_class[class_iter]
    
    # Se equation (10)
    # For simiplification we assume that all of the feature dimensions are uncorrelated with no off diagonal covariance elements.

    for class_iter in classes:
        for i in range(0,Npts):
            for j in range(0,Ndims):
                if labels[i] == class_iter:
                    sigma[class_iter][j][j] = sigma[class_iter][j][j] + math.pow(X[i][j] - mu[class_iter][j],2)
        sigma[class_iter] = sigma[class_iter]/occurance_in_class[class_iter]
    # ==========================
    """

    # Here we add W parameter to the function 
    # Se equation (13)
    for class_iter in classes:
        for data_row in range(0,Npts):
            if labels[data_row] == class_iter:
                mu[class_iter] = mu[class_iter] + W[data_row]*X[data_row]
                occurance_in_class[class_iter] = occurance_in_class[class_iter] + W[data_row]
        mu[class_iter] = mu[class_iter]/occurance_in_class[class_iter]
    
    # Se equation (14)
    for class_iter in classes:
        for data_row in range(0,Npts):
            for data_col in range(0,Ndims):
                if labels[data_row] == class_iter:
                    sigma[class_iter][data_col][data_col] = sigma[class_iter][data_col][data_col] + W[data_row]*math.pow(X[data_row][data_col] - mu[class_iter][data_col],2)
        sigma[class_iter] = sigma[class_iter]/occurance_in_class[class_iter]
        
    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    
    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))


    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================

    # Se equation (11)   
    for class_iter in range(0,Nclasses):
        for data in range(Npts):
            term1 = -0.5*(np.log(np.linalg.det(sigma[class_iter])))
            term2 = -0.5*np.dot(np.dot((X[data,:]-mu[class_iter]),np.linalg.inv(sigma[class_iter])), (X[data,:]-mu[class_iter]).transpose())  
            term3 = np.log(prior[class_iter])
            logProb[class_iter][data]= term1 + term2 + term3
    # ==========================
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:
# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.


X, labels = genBlobs(centers=5)
mu, sigma = mlParams(X,labels)
#plotGaussian(X,labels,mu,sigma)


# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BayesClassifier(), dataset='iris', split=0.7)



#testClassifier(BayesClassifier(), dataset='vowel', split=0.7)



#plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    # Npts = number of data points in a class
    # Ndims = number of dimensions, the amount of classes 
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)
    weight_sum = 0

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================

        error = 0

        for i in range(len(vote)):
            wCur_curr = wCur[i]
            label_curr = labels[i]
            vote_curr = vote[i]
        
            if label_curr == vote_curr:
                delta = 1 
            else:
                delta = 0
                error = error + wCur_curr*(1-delta)
        if error == 0:
            error = 0.0001
        
        # Getting alpha and the new weights
        alpha = 0.5*(math.log(1-error) - math.log(error))
        for i in range (len(X)):
            if labels[i] == vote[i]:
                wCur[i] = wCur[i]*math.exp(-alpha)
            else:
                wCur[i] = wCur[i]*math.exp(alpha)
        weight_sum = np.sum(wCur)
        wCur = wCur/weight_sum
        alphas.append(alpha)
                
        # ==========================
        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        # See equation 15 in the lab description
        for class_iter in range (Nclasses):
            for classifier_iter in range(Ncomps):
                vote = classifiers[classifier_iter].classify(X)
                for data in range (Npts):
                    if vote[data] == class_iter:
                        delta = 1 
                    else:
                        delta = 0
                    votes[data][class_iter] = votes[data][class_iter] + alphas[classifier_iter]*delta     
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)

testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)

#plotBoundary(BoostClassifier(BayesClassifier()), dataset='vowel', split=0.7)



# Now repeat the steps with a decision tree classifier.


#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])

