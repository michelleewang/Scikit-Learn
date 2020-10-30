# python 3.7
# Scikit-learn ver. 0.23.2
import sklearn
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# matplotlib 3.3.1
from matplotlib import pyplot

wine = datasets.load_wine()

X = wine.data.reshape(len(wine.data), 13)
Y = wine.target

trainX, testX, trainY, testY = train_test_split(
    X, Y, test_size = 0.3, shuffle = True
    )

classifierUS = LogisticRegression(max_iter = 1000)
classifierUS.fit(trainX, trainY)
predsUS = classifierUS.predict(testX)

correctUS = 0
incorrectUS = 0
for pred, gt in zip(predsUS, testY):
    if pred == gt: correctUS += 1
    else: incorrectUS += 1
print(f"Correct: {correctUS}, Incorrect: {incorrectUS}, % Correct: {correctUS/(correctUS + incorrectUS): 5.2}")

metrics.plot_confusion_matrix(classifierUS, testX, testY)
pyplot.show()

#----------
#SCALED DATA

scaled = preprocessing.scale(X)

trainX, testX, trainY, testY = train_test_split(
    scaled, Y, test_size = 0.3, shuffle = True
    )

classifierS = LogisticRegression(max_iter = 1000)
classifierS.fit(trainX, trainY)
predsS = classifierS.predict(testX)

correctS = 0
incorrectS = 0
for predS, gt in zip(predsS, testY):
    if pred == gt: correctS += 1
    else: incorrectS += 1
print(f"Correct: {correctS}, Incorrect: {incorrectS}, % Correct: {correctS/(correctS + incorrectS): 5.2}")

metrics.plot_confusion_matrix(classifierS, testX, testY)
pyplot.show()
