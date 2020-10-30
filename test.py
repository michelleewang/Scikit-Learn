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

irises = datasets.load_iris()
wine = datasets.load_wine()
facePairs = datasets.fetch_lfw_pairs()
breastCancer = datasets.load_breast_cancer()
covertypes = datasets.fetch_covtype()
diabetes = datasets.load_diabetes()
olvettiFaces = datasets.fetch_olivetti_faces()

# change this to test different datasets
dataset = wine
dimensionality = 13

X = dataset.data.reshape(len(dataset.data), dimensionality)
Y = dataset.target

scaled = preprocessing.MinMaxScaler()
scaled_data = scaled.fit_transform(X)

trainX, testX, trainY, testY = train_test_split(
    X, Y, test_size = 0.3, shuffle = True
    )

classifier = LogisticRegression(max_iter = 1000)
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)

correct = 0
incorrect = 0
for pred, gt in zip(preds, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")

metrics.plot_confusion_matrix(classifier, testX, testY)
pyplot.show()
