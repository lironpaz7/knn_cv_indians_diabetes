from point import Point
from numpy import mean, var


class DummyNormalizer:
    def fit(self, points):
        pass

    def transform(self, points):
        return points


class SumNormalizer:
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        """
        Fits the data set and updates the mean and variance values
        :param points: set of data
        """
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [abs(x[i]) for x in all_coordinates]
            self.mean_variance_list.append([sum(values), var(values, ddof=1) ** 0.5])

    def transform(self, points):
        """
        Executes normalization of sum normalizer on the given data set
        :param points: set of data
        :return: list of normalized data
        """
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i]) / self.mean_variance_list[i][0]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class MinMaxNormalizer:
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        """
        Fits the data set and updates the mean and variance values
        :param points: set of data
        """
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([min(values), max(values) - min(values)])

    def transform(self, points):
        """
        Executes normalization of min max normalizer on the given data set
        :param points: set of data
        :return: list of normalized data
        """
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class ZNormalizer:
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        """
        Fits the data set and updates the mean and variance values
        :param points: set of data
        """
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([mean(values), var(values, ddof=1) ** 0.5])

    def transform(self, points):
        """
        Executes normalization of Z normalizer on the given data set
        :param points: set of data
        :return: list of normalized data
        """
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new
