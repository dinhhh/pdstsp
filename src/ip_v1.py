import os

from ortools.linear_solver import pywraplp

from src.init_log import init_log
from src.utils import Utils

lib_path = os.path.abspath(os.path.join('..'))


def solve(logger):
    if logger is None:
        raise Exception("Error: logger is None!")

    solver = pywraplp.Solver('wsn',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    num_all_node = 0
    num_customer_node = 0
    num_drone = 0

    c_u = [] # phuc vu duoc boi drone

    x = {}
    y = {}

    alpha = solver.NumVar(0, solver.infinity(), "time")

    for i in range(num_all_node):
        for j in range(num_all_node):
            x[i, j] = solver.BoolVar("x[%i, %i]" % (i, j))

    for i in range(num_all_node):
        for k in c_u:
            y[i, k] = solver.BoolVar("y[%i, %i]" % (i, k))

    solver.Minimize(alpha)


if __name__ == '__main__':
    log = init_log()
    log.info("Start running IP...")

    for path in Utils.get_instance().data_files:
        Utils.get_instance().change_data(path)
        log.info("input path: %s" % path)
        solve(logger=log)