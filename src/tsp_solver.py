import timeit
import itertools


def remove_values_from_list(l, val):
    return [k for k in l if k != val]

def tsp_solve(const_matrix):
    n = len(const_matrix[0])
    cost_matrix = [[const_matrix[i][j] for i in range(n)] for j in range(n)]
    # ignore start node
    for i in range(n):
        cost_matrix[0][i] = 0
        cost_matrix[i][0] = 0
    # for i in cost_matrix:
    #     print(i)
    # 12    
    min_value = []
    for i in range(1, n):
        sorted_cost = [k for k in cost_matrix[i] if k != 0]
        # sorted_cost = remove_values_from_list(sorted_cost, 0) # ignore cost = 0
        sorted_cost.sort()
        min_value.append(sorted_cost[0] + sorted_cost[1]) 

    # 13
    # next_node = 0 ??????????
    next_node = min_value.index(min(min_value)) + 1 # add 1 because of ignore start node
    if next_node == 0:
        print("next node = 0 ???")

    # 14
    # link_node = [0]
    link_node = [next_node]

    # 15 + 16
    sorted_cost_at_nn = [i for i in cost_matrix[next_node]]
    sorted_cost_at_nn.sort()
    sorted_cost_at_nn = remove_values_from_list(sorted_cost_at_nn, 0)
    left = cost_matrix[next_node].index(sorted_cost_at_nn[0])
    right = cost_matrix[next_node].index(sorted_cost_at_nn[1])

    # 17 + 18
    link_node.append(right)
    link_node.insert(0, left)

    # 19
    prev_node = next_node
    
    # 20
    for i in range(n):
        cost_matrix[i][next_node] = 0
        cost_matrix[next_node][i] = 0
    cost_matrix[left][right] = 0
    cost_matrix[right][left] = 0

    # 21 + 22
    left_array = cost_matrix[left]
    right_array = cost_matrix[right]

    # 23
    for i in range(1, n - 6): # initial node + start node
        # left
        sorted_left_array = [j for j in left_array]
        sorted_left_array.sort()
        sorted_left_array = remove_values_from_list(sorted_left_array, 0)
        left = left_array.index(sorted_left_array[0])
        
        # right
        sorted_right_array = [j for j in right_array]
        sorted_right_array.sort()
        sorted_right_array = remove_values_from_list(sorted_right_array, 0)
        right = right_array.index(sorted_right_array[0])
        
        if(sorted_left_array[0] < sorted_right_array[0]):
            prev_node = left
            next_node = left
            left = next_node # ? left = left 
            link_node.insert(0, next_node)
        else: 
            prev_node = right
            next_node = right
            right = next_node # ? right = right
            link_node.append(next_node)
        # assign row and column of next_node to 0
        for i in range(n):
            cost_matrix[i][next_node] = 0
            cost_matrix[next_node][i] = 0
        cost_matrix[left][right] = 0
        cost_matrix[right][left] = 0

    # print(link_node)
    rest_node = [i for i in range(1, n) if not (i in link_node)]

    # check permutation
    check = []

    cost = []
    # gan cac node con lai sang 1 nhanh
    for i in itertools.permutations(rest_node):
        temp_cost = 0
        assigned_to_right_route = [0]
        for k in i:
            assigned_to_right_route.append(k)
        for k in link_node:
            assigned_to_right_route.append(k)
        # my check
        # check.append(assigned_to_right_route)
        for k in range(-1, n - 1):
            temp_cost += const_matrix[assigned_to_right_route[k]][assigned_to_right_route[k + 1]]
        cost.append(temp_cost)

        temp_cost = 0
        assigned_to_left_route = [0]
        for k in link_node:
            assigned_to_left_route.append(k)
        for k in i:
            assigned_to_left_route.append(k)
        # my check
        # check.append(assigned_to_left_route)
        for k in range(-1, n - 1):
            temp_cost += const_matrix[assigned_to_left_route[k]][assigned_to_left_route[k + 1]]    
        cost.append(temp_cost)
    
    for i in range(len(rest_node)):
        
        node_at_i = rest_node[i]
        temp_rest_node = [k for k in rest_node if k != node_at_i]
        for k in itertools.permutations(temp_rest_node):
            temp_cost = 0
            # [0, node at i, link_node, permutation]
            assigned_to_left_route = [0]
            assigned_to_left_route.append(node_at_i)
            for j in link_node:
                assigned_to_left_route.append(j)
            for j in k:
                assigned_to_left_route.append(j)
            # my check
            # check.append(assigned_to_left_route)
            for j in range(-1, n - 1):
                temp_cost += const_matrix[assigned_to_left_route[j]][assigned_to_left_route[j + 1]]
            cost.append(temp_cost)
            
            temp_cost = 0
            # [0, permutation, link_node, node at i]
            assigned_to_right_route = [0]
            for j in k:
                assigned_to_right_route.append(j)
            for j in link_node:
                assigned_to_right_route.append(j)
            assigned_to_right_route.append(node_at_i)
            # my check
            # check.append(assigned_to_right_route)
            for j in range(-1, n - 1):
                temp_cost += const_matrix[assigned_to_right_route[j]][assigned_to_right_route[j + 1]]
            cost.append(temp_cost)

    return min(cost)

if __name__ == '__main__':
    cost_matrix = [
        [0,3,7,5,4,3,8,9],
        [3,0,4,7,2,4,5,1],
        [7,4,0,5,10,7,2,3],
        [5,7,5,0,15,9,1,7],
        [4,2,10,15,0,10,8,2],
        [3,4,7,9,10,0,9,4],
        [8,5,2,1,8,9,0,5],
        [9,1,3,7,2,4,5,0]
    ]
    start_time = timeit.default_timer()
    print(tsp_solve(cost_matrix))
    print("time: " + str(timeit.default_timer() - start_time))
    # left_array = [9, 0, 3, 7, 0, 4, 5, 0]
    # sorted_left_array = [j for j in left_array]
    # sorted_left_array.sort()
    # sorted_left_array = remove_values_from_list(sorted_left_array, 0)
    # print(sorted_left_array)
    # left = left_array.index(sorted_left_array[0])
    # print(left)