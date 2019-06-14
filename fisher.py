#importing the libraries
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import norm

#This function solves the intersection of two normal distributions by taking mean and standard deviation of two distributions
def solveNormalCurves(mean1,mean2,stdev1,stdev2):
  a = 1/(2*stdev1**2) - 1/(2*stdev2**2)
  b = mean2/(stdev2**2) - mean1/(stdev1**2)
  c = mean1**2 /(2*stdev1**2) - mean2**2 / (2*stdev2**2) - np.log(stdev2/stdev1)
  return np.roots([a,b,c])

#reading the dataset
dataset_no = 2
x = pd.read_csv('Datasets/dataset_' + str(dataset_no) + '.csv')
print("Running On Dataset-" + str(dataset_no))
new = []
for i in zip(x['x'], x['y'], x['t']):
    new.append((i[0], i[1], i[2]))
new = np.array(new)

#sepearating the data set according to third column 
class0 = np.array([i for i in new if i[2]==0])
class1 = np.array([i for i in new if i[2]==1])

#finding the mean of the two classes
meanx, meany = np.mean(class0, axis = 0)[0:2]
mean0 = np.ndarray((2,1), buffer = np.array([meanx, meany]))

meanx, meany = np.mean(class1, axis = 0)[0:2]
mean1 = np.ndarray((2,1), buffer = np.array([meanx, meany]))

#finding the difference of the features from the mean and multiplying by the transpose of the differnce
sum1 = 0
for i in class0:
    x = np.ndarray((2,1), buffer = np.array([i[0], i[1]]))
    y = x-mean0
    sum1 += np.matmul(y,np.transpose(y))


sum2 = 0
for i in class1:
    x = np.ndarray((2,1), buffer = np.array([i[0], i[1]]))
    y = x-mean1
    sum2 += np.matmul(y,np.transpose(y))
sum1 += sum2

#inversing the sum and multiply it by the mean of the differnce
vector = np.matmul(inv(sum1), mean1-mean0)

#scattering the points of the two different classes
plt.scatter(class0[:,0], class0[:, 1], label="dots", color="blue", marker=".", s=3)
plt.scatter(class1[:,0], class1[:, 1], label="dots", color="red", marker=".", s=3)

#plotting the line that minimizes the error
w1, w2 = vector[0,0], vector[1,0]
print("w1: {}".format(w1))
print("w2: {}".format(w2))
print("slope of the line onto which points are to be projected(green line): {}".format(w2/w1))
if(dataset_no==1):
    x_lim = -0.025
    y_lim = 0.025
elif(dataset_no==2):
    x_lim = -0.08
    y_lim = 0.08
else:
    x_lim = -0.035
    y_lim = 0.035
x = np.linspace(x_lim, y_lim,1000)
y = (w2/w1)*x
plt.plot(x, y, label='the line', color = "green")

proj0 = np.array([w1*i[0]+w2*i[1] for i in class0])
proj1 = np.array([w1*i[0]+w2*i[1] for i in class1])

mu0, std0 = norm.fit(proj0)
mu1, std1 = norm.fit(proj1)

#finding the point of intersection of the normalisation curves
i = 0
if(dataset_no==1):
    i = 0
elif(dataset_no==2):
    i = 1
else:
    i = 1
result = solveNormalCurves(mu0,mu1,std0,std1)
print("Threshold: {}".format(result[i]))

w1, w2 = vector[0,0], vector[1,0]
x = np.linspace(-3, 3, 1000)
y = (-w1/w2)*x + result[1]/w2
plt.plot(x, y, label='the line', color = 'magenta')
plt.savefig('Outputs/Fisher/dataset-' + str(dataset_no) + '-lines.png')
plt.show()


if(dataset_no==1):
    x_lim = -0.03
    y_lim = 0.015
elif(dataset_no==2):
    x_lim = -0.03
    y_lim = 0.03
else:
    x_lim = -0.04
    y_lim = 0.05

#plotting the normalisation curves
x = np.linspace(x_lim, y_lim, 1000)
p = norm.pdf(x, mu0, std0)
plt.grid()
plt.plot(x, p, 'k', linewidth=1, color="blue")

x = np.linspace(x_lim, y_lim, 1000)
p = norm.pdf(x, mu1, std1)
plt.grid()
plt.plot(x, p, 'k', linewidth=1, color="red")
plt.plot(result,norm.pdf(result,mu0,std0),'o', color="green")

plt.scatter(proj0, [0]*len(proj0), label="dots", color="blue", marker=".", s=4)
plt.scatter(proj1, [0]*len(proj1), label="dots", color="red", marker=".", s=4)
plt.scatter(result[1],0, color = 'black', s = 30)
plt.xlim(x_lim, y_lim)
plt.savefig('Outputs/Fisher/dataset-' + str(dataset_no) + '-normal_curves.png')
plt.show()

plt.scatter(proj0, [0]*len(proj0), label="dots", color="blue", marker=".", s=4)
plt.scatter(proj1, [0]*len(proj1), label="dots", color="red", marker=".", s=4)
plt.scatter(result[1],0, color = 'black', s = 30)
plt.xlim(x_lim, y_lim)
plt.savefig('Outputs/Fisher/dataset-' + str(dataset_no) + '-1D_plot.png')
plt.show()

