
# coding: utf-8

# In[34]:

"""
         1         2         3         4         5         6         7      
1234567890123456789012345678901234567890123456789012345678901234567890123456

Cordell Newmiller
"""

import numpy
from matplotlib import pyplot
get_ipython().magic('matplotlib inline')

def tripledot(A,B,C):
    return (numpy.dot(A,numpy.dot(B,C)))

def ChiSquare(ymodel,ydata,ysigmas=0,numberofparameters=2):
    if not numpy.any(ysigmas):
        ysigmas=numpy.zeros(ydata.shape)+1.0
    chisquared=numpy.sum(((ydata-ymodel)/ysigmas)**2,0) # sum along axis 0
    reducedchisquared=chisquared/(ymodel.shape[0]-numberofparameters-1)
    return (chisquared,reducedchisquared)

def PolynomialFit(xdata=0,ydata=0,ysigmas=0,numberofparameters=2,verbose=False):
    """
    xdata is a vector of x data points
    ydata is a vector of y data points
    xsigmas is a vector of errors for y
    numberofparameters is the desired order of the fit, 
        e.g. 2 is linear, 3 is quadratic etc.
    verbose is a flag. If true, the function will print and plot the results.
    
    Returns: 
    ymodel is a vector of the y values of the fit, meant for 
        plotting with xdata.
    parameters is a vector of fitted coefficients , beginning with the lowest 
        order (constant) one. It has length(numberofparameters).
    psigmas is a vector of errors for the fitted coefficients. 
    chisquared is a scalar, the chi-squared of the fit. 
    reducedchisquared is a scalar, the reduced chi-squared of the fit. 
    
    """
    
    # Default Values
    if (not numpy.any(xdata)) and (not numpy.any(ydata)):
        # Assumes the file is in the working directory
        data=numpy.loadtxt('xy_fitting.txt')
        xdata=data[:,0]
        ydata=data[:,1]
    if not numpy.any(ysigmas):
        ysigmas=numpy.zeros(ydata.shape)+5.0

    # Making the weights into a diagonal matrix lets us do this fit with 
    # matrix operations instead of for loops. 
    # The diag function here serves to turn a vector into a diagonal matrix
    # with the values of the vector along the diagonal. 
    weights=numpy.diag(1/(ysigmas**2))


    # Generate monomial dy/da factors. dy/da0=1, dy/da1=x, dy/da2=x^2, etc.
    derivatives=numpy.zeros((xdata.size,numberofparameters))
    for i in range(0,numberofparameters):
        derivatives[:,i]=xdata**i 

    # Curvature matrix
    alpha = tripledot(derivatives.T,weights,derivatives)

    # Gradient vector
    beta = tripledot(ydata,weights,derivatives)


    covariencematrix=numpy.linalg.inv(alpha)

    # Parameters a
    parameters = numpy.dot(covariencematrix,beta.T)
    
    # The diag function here serves to turn the diagonal of a matrix
    # into a single vector, with off-diagonal elements discarded.
    psigmas = (numpy.diag(covariencematrix))**0.5

    ymodel=numpy.zeros(xdata.size)
    for i in range(0,numberofparameters):
        ymodel=ymodel+parameters[i]*xdata**i

    chisquared,reducedchisquared=ChiSquare(ymodel,ydata,ysigmas,numberofparameters)
        
    if verbose:
        # Suppress scientific notation output
        numpy.set_printoptions(suppress=True,precision=3) 
        
        #print("Curvature matrix α: \n",alpha)
        #print("Gradient vector β: \n",beta)
        #print("Covarience matrix: \n",covariencematrix)

        for i in range(0,numberofparameters):
            #print("a",i," is ",parameters[i]," ± ",psigmas[i],sep="")
            print("a%s is %2.2f ± %2.2f" % (i, parameters[i], psigmas[i]))


        figure1 = pyplot.figure(figsize=(10,10));
        pyplot.errorbar(xdata,ydata,yerr=ysigmas,fmt='ro');
        pyplot.plot(xdata,ymodel,'b');
        pyplot.title('Least Squares Fit')
        pyplot.xlabel('x')
        pyplot.ylabel('y')

        print("        χ2 is",chisquared,"\nReduced χ2 is",reducedchisquared)

        if reducedchisquared < 1:
            print("The data might be over-fitted.")
        elif reducedchisquared < 2:
            print("The fit appears to be appropriate.")
        else:
            print("The fit is not very good.")

        # Reset printoptions to default, useful when working in notebook
        numpy.set_printoptions(suppress=False,precision=8) 
    
    return (ymodel,parameters,psigmas,chisquared,reducedchisquared)

