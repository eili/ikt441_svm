import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

import warnings
warnings.simplefilter("ignore")

training = [[float(i) for i in i.split(";")] for i in open("training-red.csv").readlines() if i.strip()]


training_good = [i[:-1] for i in training if i[-1]>6]
#training_medium = [i[:-1] for i in training if i[-1]>4 and i[-1]<8]
training_bad = [i[:-1] for i in training if i[-1]<5]

class_good = np.ones((len(training_good),), dtype=np.int) + 1
#class_medium = np.ones((len(training_medium),), dtype=np.int)
class_bad = np.zeros((len(training_bad),), dtype=np.int)

print("Good", len(training_good))
#print("Medium", len(training_medium))
print("Bad", len(training_bad))

#print(len(training_good))

wine2d_good = []
wine2d_medium = []
wine2d_bad = []

def pickColumns(arr, ix, iy):
    result = []
    for d in arr:
        x = d[ix]
        y = d[iy]
        result.append([x, y])
    return result

wine2d_good = pickColumns(training_good, 5, 10)
#wine2d_medium = pickColumns(training_medium, 3, 4)
wine2d_bad = pickColumns(training_bad, 5, 10)

'''
winemulti_good = []
winemulti_bad = []
for d in training_good:
    x0 = d[0]
    x1 = d[3]
    x2 = d[10]
    winemulti_good.append([x0, x1, x2])

for d in training_bad:
    x0 = d[0]
    x1 = d[3]
    x2 = d[10]
    winemulti_bad.append([x0, x1, x2])
'''
#print("Acid Good")    
#print(acid_good)

def plotGraph(results, ylabel):
    x = range(0, len(results))
       
    plt.plot(x, results)
    plt.title("Wine")
    plt.ylabel(ylabel)
    plt.xlabel("Iteration")
    # plt.legend()
    plt.grid(True, color='g')
    plt.show()
    

#plotGraph(acid_good, "Good")
#plotGraph(acid_bad, "Bad")

'''
X = np.array(wine2d_good + wine2d_medium + wine2d_bad)
Y = np.append(class_good, class_medium)
Y = np.append(Y, class_bad)
'''
X = np.array(wine2d_good + wine2d_bad)
Y = np.append(class_good, class_bad)

#multi dimension data
#X = np.array(winemulti_good+winemulti_good)





C = 1
gamma = 0.5
svm_linear = svm.SVC(kernel='linear',C=C,gamma=gamma).fit(X,Y)
svm_rbf = svm.SVC(kernel='rbf',C=C,gamma=gamma).fit(X,Y)
svm_sigmoid = svm.SVC(kernel='sigmoid',C=C,gamma=gamma).fit(X,Y)
svm_poly = svm.SVC(kernel='poly', degree=3, C=C).fit(X,Y)


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

h = 0.2 #Mesh step
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
#Bad light blue
def plotSVM(svm,n,title):
    plt.subplot(2,2,n)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.title(title)


plotSVM(svm_linear,1,"Linear")
plotSVM(svm_rbf,2,"RBF")
plotSVM(svm_sigmoid,3,"Sigmoid")
plotSVM(svm_poly, 4, "Polynomial")

plt.show()
