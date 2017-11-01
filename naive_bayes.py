from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from base_classifier import BaseClassifier, SimpleTestPredictor

if __name__ == '__main__':
    model = BaseClassifier(GaussianNB())
    predictor = SimpleTestPredictor()
    for (name, prediction, expected) in predictor.predict(model):
        print(name, '---Accuracy : %s' %
              (accuracy_score(prediction, expected)))

    # work(lambda train, target: MultinomialNB().fit(train, target))
