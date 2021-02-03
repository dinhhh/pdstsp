from config.arg_parser import parse_config
import pandas as pd
from scipy.spatial import distance
import elkai
import numpy as np
import random


class Utils:
    __utils = None

    def __init__(self):
        constants, ga_config, self.data_path = parse_config()

        self.speed = constants["speed"]
        self.data = pd.read_csv(self.data_path, header=None).to_numpy()[:-1]
        self.terminate = ga_config["terminate"]
        self.pop_size = ga_config["pop_size"]
        self.num_generation = ga_config["num_generation"]
        self.cx_pb = ga_config["cx_pb"]
        self.mut_pb = ga_config["mut_pb"]
        self.i_pot = self.data[0, 1:3]

    @classmethod
    def get_instance(cls):
        if cls.__utils is None:
            cls.__utils = Utils()
        return cls.__utils

    def cal_time2serve_by_truck(self, individual: list):
        city_served_by_truck_list = [i for i, v in enumerate(individual) if v == 0]
        cost_matrix = np.array([[distance.cityblock(self.data[i, 1:3], self.data[j, 1:3])
                                 for i in city_served_by_truck_list] for j in city_served_by_truck_list])
        route = elkai.solve_float_matrix(cost_matrix)
        return sum([distance.cityblock(self.data[route[i], 1:3], self.data[route[i + 1], 1:3])
                    for i in range(-1, len(route) - 1)]) / self.speed

    def cal_time2serve_by_drones(self, individual: list):
        return 2 / self.speed * sum([distance.euclidean((self.data[i, 1:3]), self.i_pot)
                                     for i in individual if i != 0])

    def cal_fitness(self, individual: list):
        return max(self.cal_time2serve_by_truck(individual=individual),
                   self.cal_time2serve_by_drones(individual=individual))

    @staticmethod
    def init_individual(size):
        ind = [random.randint(0, 1) for _ in range(size)]
        ind[0] = 0
        return ind

    @staticmethod
    def mutate_flip_bit(individual, ind_pb):
        for i in range(1, len(individual)):
            if random.random() < ind_pb:
                individual[i] = type(individual[i])(not individual[i])
        return individual,


if __name__ == '__main__':
    print(Utils.get_instance().cal_fitness([0, 0, 1]))
    var = Utils.get_instance()
    for _ in range(10):
        print(Utils.init_individual(5))