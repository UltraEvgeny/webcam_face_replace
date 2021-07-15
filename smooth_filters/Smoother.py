from utils.util import *


class Smoother:
    def __init__(self, size=10):
        self.maxsize = size
        self.values = []

    def put(self, value):
        self.values.append(value)
        if len(self.values)>self.maxsize:
            self.values.pop(0)

    def get_avg(self, key=lambda x: x):
        avg = sum(map(key, self.values))/len(self.values)
        return avg

    def get_median_value(self):
        m = self.values[round(len(self.values)/2)]
        return m


def get_x_from_data(data):
    return np.array([[t.x, t.y, t.z] for t in data.landmark])


def restore_data_from_x(data, x):
    new_data = data.__deepcopy__()
    for i in range(len(data.landmark)):
        new_data.landmark[i].x, new_data.landmark[i].y, new_data.landmark[i].z = x[i]
    return new_data


class Filter:
    def __init__(self, maxsize=5):
        """Initialize the one euro filter."""
        # The parameters.
        self.values = Smoother(size=maxsize)

    def __call__(self, data):
        """Compute the filtered signal."""
        # if first time:
        x = get_x_from_data(data)
        self.values.put(x)
        x_hat = self.values.get_avg()
        data_hat = restore_data_from_x(data, x_hat)
        return data_hat
