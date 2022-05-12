
import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import glob
from PIL import Image
from scipy.stats import t




# • Use matrix operation to find SSE (error vector), 
# • MSE (error variance)
# • 𝑠.𝑒 ( 𝛽0), 𝑠.𝑒. (𝛽1), 𝑎𝑛𝑑 𝑠.𝑒. (𝛽2)
# • Coefficient of correlation between 𝑥1 and 𝑦, and between 𝑥2 and 𝑦
# • Hypothesis testing for linearity of 𝛽1, 𝑎𝑛𝑑 𝛽2 at 𝛼 = 0.01 separately, 
# i.e., test 𝐻0: 𝛽1 = 0 vs. 𝐻0: 𝛽1 ≠ 0, and test 𝐻0: 𝛽2 = 0 vs. 𝐻0: 𝛽2 ≠ 0
# # hypothesis testing for linearity of 𝛽1, 𝑎𝑛𝑑 𝛽2 



def transpose(matrix):
    # transpose a matrix and return the transpose
    newMatrix = []
    for i in range(len(matrix[0])):
        newMatrix.append([])
        for j in range(len(matrix)):
            newMatrix[i].append(matrix[j][i])
    return newMatrix
print(transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

# Function to determine coefficient of correlation between 2 data sets
def determine_r(nvalues1, nvalues2):
    n = len(nvalues1)
    values1 = []
    values2 = []
    for i in range(n):
        values1.append(int(nvalues1[i]))
        values2.append(int(nvalues2[i]))
    
    #Initialize Sums
    sumx = 0
    sumy = 0
    sumxy = 0
    sumx2 = 0
    sumy2 = 0

    #Calc sums
    for i in range(n):
        sumx += values1[i]
        sumy += values2[i]
        sumxy += (values1[i] * values2[i])
        sumx2 += (values1[i] ** 2)
        sumy2 += (values2[i] ** 2)
        
    
    #Calc and Return r
    numerator = (n * sumxy) - (sumx * sumy)
    denominator = math.sqrt(((n * sumx2) - (sumx ** 2)) * ((n * sumy2) - (sumy ** 2)))
    return numerator / denominator

def multiply(matrix1, matrix2):
    # multiply two matrices without using numpy
    matrix3 = []
    for i in range(len(matrix1)):
        matrix3.append([])
        for j in range(len(matrix2[0])):
            matrix3[i].append(0)
            for k in range(len(matrix1[0])):
                matrix3[i][j] += matrix1[i][k] * matrix2[k][j]
    return matrix3
print(multiply([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]))



def make_gif():
    # create a sorted list of all the images in the images folder
    images = glob.glob('images/*.png')
    # sort the list by name
    images.sort()
    # create a gif
    images = [Image.open(i) for i in images]
    images[0].save('./animation.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    




def main():
    df = pd.read_csv('./data.csv')
    # remove the first column of the dataframe 

    names = df['Name']
    # convert names to np array
    names = np.array(names)
    x1 = df['Budget Adjusted for Inflation']
    x1Values = np.array(x1)
    x2 = df['Rating']
    x2Values = np.array(x2)
    y = df['Box Office Adjusted for Inflation']
    yValues = np.array(y)


    # • Use matrix operations to find b0, b1, b2, find the regression line (𝛽 = (𝑋′𝑋)^−1(𝑋′𝑌))
    # find the slope and the intercept of the regression line
    
    # construct X and Y matrices
    X = []
    Y = []
    for i in range(len(x1Values)):
        X.append([1, x1Values[i], x2Values[i]])
        Y.append([yValues[i]])
    print(X, Y)
    # transpose X matrix
    XT = transpose(X)
    print("X TRANSPOSE:",XT)
    # multiply X and Y matrices
    term1 = np.array(multiply(XT, X))
    term2 = np.array(multiply(XT, Y))
    term1Inv = np.linalg.inv(term1)
    b = multiply(term1Inv, term2)
    
    # intercept
    b0 = b[0][0]
    # x1 slope
    b1 = b[1][0]
    # x2 slope
    b2 = b[2][0]
    # print each value labeled
    print("b0:", b0)
    print("b1:", b1)
    print("b2:", b2)
    # create a 3d scatter plot of the x, y, and z values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1Values, x2Values, yValues)
    # label the axes with the names of the variables
    ax.set_xlabel('Budget')
    ax.set_ylabel('Rating')
    ax.set_zlabel('Box Office')
    # # label each point with its name according to the names list
    # for i in range(0,len(x1Values), 20):
    #     ax.text(x1Values[i], x2Values[i], yValues[i], names[i])
    
    
    # plot the 3 dimensional regression line on the scatter plot using the x1 slope, x2 slope, and intercept
    
    # calculate the x y and z values for the regression line
    
    regressionX = np.linspace(min(x1Values), max(x1Values), 100)
    regressionY = np.linspace(min(x2Values), max(x2Values), 100)
    regressionX, regressionY = np.meshgrid(regressionX, regressionY)
    regressionZ = b0 + b1 * regressionX + b2 * regressionY
    ax.plot_surface(regressionX, regressionY, regressionZ)
    

    # • Use matrix operations to find SSE (error vector),
    # calculated the sum of the squared error vector
    errorVector = []
    for i in range(len(yValues)):
        # y - yHat
        errorVector.append(yValues[i] - (b0 + b1 * x1Values[i] + b2 * x2Values[i]))
    sseTranpose = np.transpose(errorVector)
    sse = np.matmul(sseTranpose, errorVector)
    
    print("SSE:", sse)
    # • MSE (error variance)
    mse = sse / len(x1Values)
    print("MSE:", mse)

    # coefficient of correlation between x1 and y
    rxy = determine_r(x1Values, yValues)
    print("rxy:", rxy)

    # coefficient of correlation between x2 and y
    rxy2 = determine_r(x2Values, yValues)
    print("rxy2:", rxy2)

    # coefficient of correlation between x1 and x2
    rxx = determine_r(x1Values, x2Values)
    print("rxx: ", rxx)

    # multiple coefficient of correlation between x1, x2, and y
    R = math.sqrt(((rxy ** 2) + (rxy2 ** 2) - (2 * rxy) * rxy2 * rxx) / (1 - (rxx ** 2)))
    print("R: ", R)

    # • Hypothesis testing for linearity of 𝛽1, 𝑎𝑛𝑑 𝛽2 at 𝛼 = 0.01 separately,
    # i.e., test 𝐻0: 𝛽1 = 0 vs. 𝐻0: 𝛽1 ≠ 0, and test 𝐻0: 𝛽2 = 0 vs. 𝐻0: 𝛽2 ≠ 0
    # hypothesis testing for linearity of 𝛽1

    
    # hypothesis testing for linearity of 𝛽2


    # save 360 degree rotation of the scatter plot
    for k in range(0,360, 10):
        ax.view_init(elev=10, azim=k)
        plt.savefig('./images/scatter_rotation_%d.png' % k)

    # use all the images in the folder to create a gif using 
    # make_gif()
    # plt.show()

    x1bar = 0
    for i in range(len(x1Values)):
        x1bar += x1Values[i]
    x1bar = x1bar / len(x1Values)
    print("x1bar:", x1bar)

    x2bar = 0
    for i in range(len(x2Values)):
        x2bar += x2Values[i]
    x2bar = x2bar / len(x2Values)
    print("x2bar:", x2bar)

    ybar = 0
    for i in range(len(yValues)):
        ybar += yValues[i]
    ybar = ybar / len(yValues)
    print("ybar:", ybar)
    

    

def hypothesis_testing(beta, b, standard_error, n, alpha):
    # degrees of freedom
    degreesOfFreedom = n - 3
    # t-value
    t1 = beta / standard_error
    # p-value
    p1 = (1 - t.cdf(x=t1, df = degreesOfFreedom)) * 2
    print("p1:", p1)
    # t-value
    t2 = b / standard_error
    # p-value
    p2 = (1 - t.cdf(x=t2, df = degreesOfFreedom)) * 2
    print("p2:", p2)

    if(p1 <= alpha):
        print("P1:", p1)
        print("Alpha:", alpha)
        print("Since the first p-value is less than or equal to alpha, we reject the null hypothesis.")
    else:
        print("P1:", p1)
        print("Alpha:", alpha)
        print("Since the first p-value is greater than alpha, we fail to reject the null hypothesis.")
    if(p2 <= alpha):
        print("P2:", p2)
        print("Alpha:", alpha)
        print("Since the second p-value is less than or equal to alpha, we reject the null hypothesis.")
    else:
        print("P2:", p2)
        print("Alpha:", alpha)
        print("Since the second p-value is greater than alpha, we fail to reject the null hypothesis.")





main()