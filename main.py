from sys import argv
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def run_knn(points):
    """
    Runs knn with given set of data
    :param points: set of data
    """
    m = KNN(5)
    m.train(points)
    print(f'predicted class: {m.predict(points[0])}')
    print(f'true class: {points[0].label}')
    cv = CrossValidation()
    cv.run_cv(points, 10, m, accuracy_score)


def run_1nn(points):
    """
    Runs 1-nn with given set of data
    :param points: set of data
    """
    a = KNN(1)
    a.train(points)
    print(f'predicted class: {a.predict(points[0])}')
    print(f'true class: {points[0].label}')
    # cv = CrossValidation()
    # cv.run_cv(points, 10, a, accuracy_score)


def run_1_to_30_knn(points):
    """
    Runs knn with k=0 to k=30 on a given set of data
    :param points: set of data
    """
    k = 0
    accuracy = 0
    num_of_points = len(points)
    for index in range(1, 31):
        a = KNN(index)
        a.train(points)
        print(f"classifier {index}:")
        print(f'predicted class: {a.predict(points[0])}')
        print(f'true class: {points[0].label}')
        cv = CrossValidation()
        temp_score = cv.run_cv(points, num_of_points, a, accuracy_score)
        if temp_score > accuracy:
            accuracy = temp_score
            k = index
        print()
    print(f"best classifier is: {k}, best accuracy is: {accuracy}")


def k_fold_cross_validation(points, k):
    """
    Runs a knn for a given k value on a set of data and each time with different fold
    :param points: set of data
    :param k: value for knn
    """
    folds = [2, 10, 20]
    print(f"K={k}")
    for fold in folds:
        a = KNN(k)
        a.train(points)
        cv = CrossValidation()
        print(f"{fold}-fold-cross-validation:")
        cv.run_cv(points, fold, a, accuracy_score, False, True)


def two_fold_cross_validation(points):
    """
    Runs two fold cross validation on specific k values and each time test another norm
    :param points: set of data
    """
    knns = [5, 7]
    norms = [DummyNormalizer, SumNormalizer, MinMaxNormalizer, ZNormalizer]
    prints = 0
    for knn in knns:
        print(f"K={knn}")
        for norm in norms:
            a = KNN(knn)
            nor = norm()
            nor.fit(points)
            temp_points = nor.transform(points)
            a.train(temp_points)
            cv = CrossValidation()
            accuracy = cv.run_cv(temp_points, 2, a, accuracy_score, True, True)
            print(f"Accuracy of {norm.__name__} is {accuracy}")
            prints += 1
            if prints != len(knns) * len(norms):
                print()


if __name__ == '__main__':
    loaded_points = load_data()
    print("Question 3:")
    k_fold_cross_validation(loaded_points, 19)
    print("Question 4:")
    two_fold_cross_validation(loaded_points)
