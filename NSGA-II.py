import random as rn
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import patches
from matplotlib.patches import Polygon


def random_population(_nv, n, _lb, _ub):
    _pop = np.zeros((n, 2 * nv))
    for i in range(n):
        _pop[i, :] = np.random.uniform(lb, ub)
        for j in range(int(_pop[i, :].size / 2)):
            if _pop[i, j * 2] < 0:
                _pop[i, j * 2] = int(-1)
            else:
                _pop[i, j * 2] = int(1)

    return _pop


def crossover(_pop, crossover_rate):
    next_gen = np.zeros((crossover_rate, _pop.shape[1]))
    for i in range(int(crossover_rate / 2)):
        r1 = np.random.randint(0, _pop.shape[0])
        r2 = np.random.randint(0, _pop.shape[0])
        while r1 == r2:
            r1 = np.random.randint(0, _pop.shape[0])
            r2 = np.random.randint(0, _pop.shape[0])
        cutting_point = np.random.randint(1, _pop.shape[1])
        next_gen[2 * i, 0:cutting_point] = _pop[r1, 0:cutting_point]
        next_gen[2 * i, cutting_point:] = _pop[r2, cutting_point:]
        next_gen[2 * i + 1, 0:cutting_point] = _pop[r2, 0:cutting_point]
        next_gen[2 * i + 1, cutting_point:] = _pop[r1, cutting_point:]

    return next_gen


def mutation(_pop, mutation_rate):
    next_gen = np.zeros((mutation_rate, _pop.shape[1]))
    for i in range(int(mutation_rate / 2)):
        r1 = np.random.randint(0, _pop.shape[0])
        r2 = np.random.randint(0, _pop.shape[0])
        while r1 == r2:
            r1 = np.random.randint(0, _pop.shape[0])
            r2 = np.random.randint(0, _pop.shape[0])
        cutting_point = np.random.randint(0, _pop.shape[1])
        next_gen[2 * i] = _pop[r1]
        next_gen[2 * i, cutting_point] = _pop[r2, cutting_point]
        next_gen[2 * i + 1] = _pop[r2]
        next_gen[2 * i + 1, cutting_point] = _pop[r1, cutting_point]

    return next_gen


def local_search(_pop, n, _step_size):
    next_gen = np.zeros((n, _pop.shape[1]))
    for i in range(n):
        r1 = np.random.randint(0, _pop.shape[0])
        unit = _pop[r1, :]
        unit[1] += np.random.uniform(-_step_size, _step_size)
        if unit[1] < lb[1]:
            unit[1] = lb[1]
        if unit[1] > ub[1]:
            unit[1] = ub[1]
        next_gen[i, :] = unit

    return next_gen


def evaluation(_pop, x_s, y_s, alfa_s, _done):
    _fitness_values = np.zeros((_pop.shape[0], 2))
    _flipped_fitness_values = np.zeros((_pop.shape[0], 2))
    i = 0
    _trajectory = []
    V = np.zeros(nv)
    angle = np.zeros(nv)
    for individual in _pop:
        for n in range(nv):
            V[n] = individual[2 * n]
            angle[n] = individual[2 * n + 1]

        x = x_s - ds * math.cos(alfa_s)
        y = y_s - ds * math.sin(alfa_s)
        alfa_n = alfa_s

        for u in range(nv):
            if abs(angle[u]) < 0.0001:
                x_n = x + V[u] * math.cos(alfa_n)
                y_n = y + V[u] * math.sin(alfa_n)
            else:
                a = dist_between_axles / math.tan(angle[u])
                Ro = math.sqrt(dist_between_axles ** 2 / 4 + (abs(a) + car_width / 2) ** 2)
                tau = math.copysign(1, angle[u]) * alfa_n + a * math.sin(dist_between_axles / 2 * Ro)
                gama = V[u] * dt / Ro
                x_n = x + Ro * (math.sin(gama + tau) - math.sin(tau))
                y_n = y + math.copysign(1, angle[u]) * Ro * (math.cos(tau) - math.cos(gama + tau))
                alfa_n = alfa_n + math.copysign(1, angle[u]) * gama
                if abs(alfa_n) > math.pi:
                    alfa_n = alfa_n - math.copysign(1, alfa_n) * math.pi * 2
            x = x_n + ds * math.cos(alfa_n)
            y = y_n + ds * math.sin(alfa_n)

        for j in range(2):
            if j == 0:  # objective 1
                if parking_length < x < -5 or parking_width < y < -5:
                    _fitness_values[i, j] = 1000
                else:
                    _fitness_values[i, j] = math.sqrt(x ** 2 + y ** 2)

            elif j == 1:  # objective 2
                _fitness_values[i, j] = beta - alfa_n
        _flipped_fitness_values[i, 0] = 1 / _fitness_values[i, 0]
        _flipped_fitness_values[i, 1] = 1 / _fitness_values[i, 1]

        if _fitness_values[i, 0] <= 0.8 and \
                (abs(_fitness_values[i, 1]) <= 0.1745 or abs(_fitness_values[i, 1]) >= 2.9671):
            _done = True
            if final is True:
                _trajectory = np.append(_trajectory, [individual])

        i = i + 1

    return _fitness_values, _trajectory, _done, _flipped_fitness_values


def best_individuals_visualization(best, x_s, y_s, alfa_s):
    _positions_x = []
    _positions_y = []
    _car_angle = []
    i = 0
    C = nv * 2
    V = np.zeros(nv)
    angle = np.zeros(nv)
    best_units = np.array_split(best, len(best) / C)
    for individual in best_units:
        for n in range(nv):
            V[n] = individual[2 * n]
            angle[n] = individual[2 * n + 1]

        x = x_s - ds * math.cos(alfa_s)
        y = y_s - ds * math.sin(alfa_s)
        alfa_n = alfa_s

        for u in range(nv):

            if abs(angle[u]) < 0.0001:
                x_n = x + V[u] * dt * math.cos(alfa_n)
                y_n = y + V[u] * dt * math.sin(alfa_n)

            else:
                a = dist_between_axles / math.tan(angle[u])
                Ro = math.sqrt(dist_between_axles ** 2 / 4 + (abs(a) + car_width / 2) ** 2)
                tau = math.copysign(1, angle[u]) * alfa_n + a * math.sin(dist_between_axles / 2 * Ro)
                gama = V[u] * dt / Ro
                x_n = x + Ro * (math.sin(gama + tau) - math.sin(tau))
                y_n = y + math.copysign(1, angle[u]) * Ro * (math.cos(tau) - math.cos(gama + tau))
                alfa_n = alfa_n + math.copysign(1, angle[u]) * gama
                if abs(alfa_n) > math.pi:
                    alfa_n = alfa_n - math.copysign(1, alfa_n) * math.pi * 2

            x = x_n + ds * math.cos(alfa_n)
            y = y_n + ds * math.sin(alfa_n)
            _positions_x = np.append(_positions_x, [x])
            _positions_y = np.append(_positions_y, [y])
            _car_angle = np.append(_car_angle, [alfa_n])

        i = i + 1

    position_x_arr = _positions_x
    position_y_arr = _positions_y
    car_angles_arr = _car_angle

    return position_x_arr, position_y_arr, car_angles_arr


def crowding_calculation(_fitness_values):
    _pop_size = len(_fitness_values[:, 0])
    fitness_value_number = len(_fitness_values[0, :])
    matrix_for_crowding = np.zeros((_pop_size, fitness_value_number))
    normalize_fitness_values = (_fitness_values - _fitness_values.min(0)) / _fitness_values.ptp(0)  # normalize fit val
    for i in range(fitness_value_number):
        crowding_results = np.zeros(_pop_size)
        crowding_results[0] = 1  # extreme point has the max crowding distance
        crowding_results[_pop_size - 1] = 1  # extreme point has the max crowding distance
        sorting_normalize_fitness_values = np.sort(normalize_fitness_values[:, i])
        sorting_normalized_values_index = np.argsort(normalize_fitness_values[:, i])
        # crowding distance calculation
        crowding_results[1:_pop_size - 1] = (
                sorting_normalize_fitness_values[2:_pop_size] - sorting_normalize_fitness_values[0:_pop_size - 2])
        re_sorting = np.argsort(sorting_normalized_values_index)  # re_sorting to the original order
        matrix_for_crowding[:, i] = crowding_results[re_sorting]

    crowding_distance = np.sum(matrix_for_crowding, axis=1)  # crowding distance of each solution

    return crowding_distance


def remove_using_crowding(_fitness_values, number_solutions_needed):
    pop_index = np.arange(_fitness_values.shape[0])
    crowding_distance = crowding_calculation(_fitness_values)
    selected_pop_index = np.zeros(number_solutions_needed)
    selected_fitness_values = np.zeros((number_solutions_needed, len(_fitness_values[0, :])))

    for i in range(number_solutions_needed):
        _pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, _pop_size - 1)
        solution_2 = rn.randint(0, _pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = _fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, solution_1, axis=0)
            _fitness_values = np.delete(fitness_values, solution_1, axis=0)
            crowding_distance = np.delete(crowding_distance, solution_1, axis=0)
        else:
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = _fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, solution_2, axis=0)
            _fitness_values = np.delete(fitness_values, solution_2, axis=0)
            crowding_distance = np.delete(crowding_distance, solution_2, axis=0)

    selected_pop_index = np.asarray(selected_pop_index, dtype=int)

    return selected_pop_index


def pareto_front_finding(_fitness_values, pop_index):
    _pop_size = _fitness_values.shape[0]
    _pareto_front = np.ones(_pop_size, dtype=bool)

    for i in range(_pop_size):
        for j in range(_pop_size):
            if all(_fitness_values[j] <= _fitness_values[i]) and any(_fitness_values[j] < _fitness_values[i]):
                _pareto_front[i] = 0
                break

    return pop_index[_pareto_front]


def selection(_pop, _fitness_values, _pop_size):
    pop_index_0 = np.arange(pop.shape[0])
    pop_index = np.arange(pop.shape[0])
    _pareto_front_index = []
    while len(_pareto_front_index) < _pop_size:
        new_pareto_front = pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
        total_pareto_size = len(_pareto_front_index) + len(new_pareto_front)
        if total_pareto_size > _pop_size:
            number_solutions_needed = pop_size - len(_pareto_front_index)
            selected_solutions = (remove_using_crowding(_fitness_values[new_pareto_front], number_solutions_needed))
            new_pareto_front = new_pareto_front[selected_solutions]

        _pareto_front_index = np.hstack((_pareto_front_index, new_pareto_front))  # add to pareto
        remaining_index = set(pop_index) - set(_pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))
    selected_pop = _pop[_pareto_front_index.astype(int)]

    return selected_pop


def GOL(_flipped_fitness_values, _fitness_values):
    gol = []
    max_fitness_val_pos = max(_fitness_values[:, 0])
    max_fitness_val_ang = max(_fitness_values[:, 1])
    for k in range(pop_summed):
        if _flipped_fitness_values[k, 0] / max_fitness_val_pos < _flipped_fitness_values[k, 1] / max_fitness_val_ang:
            gol = np.append(gol, _flipped_fitness_values[k, 0] / max_fitness_val_pos)
        else:
            gol = np.append(gol, _flipped_fitness_values[k, 1] / max_fitness_val_ang)
    best_gol = max(gol)

    return best_gol


########################
#      Parameters      #
########################
starting_x = 50.0               # wartości od 10.0 do 55.0
starting_y = 35.0               # wartości od 10.0 do 35.0
car_rotation = -math.pi/3        # wartości od -math.pi do math.pi
number_of_controls = 60
population_size = 160
########################
#      Parameters      #
########################

stan = [starting_x, starting_y, car_rotation]
nv = number_of_controls
lb = []
ub = []
for _ in range(nv):
    lb = np.append(lb, [-1, -math.pi / 6])
    ub = np.append(ub, [1, math.pi / 6])

pop_size = population_size
rate_crossover = 30
rate_mutation = 20
rate_local_search = 30
pop_summed = int(population_size + rate_crossover + rate_mutation + rate_local_search)
step_size = 0.1
pop = random_population(nv, pop_size, lb, ub)

best_gols = []
final = False
done = False
parking_spot_length = 6.0
parking_spot_width = 3.0
beta = 0
parking_length = 60.0
parking_width = 40.0
car_width = 1.8
car_length = 4.0
front_axle = 1.2
rear_axle = 0.34
ds = (front_axle - rear_axle / 2)
dist_between_axles = car_length - front_axle - rear_axle
dt = 1

iterations = 0
while not done:
    offspring_from_crossover = crossover(pop, rate_crossover)
    offspring_from_mutation = mutation(pop, rate_mutation)
    offspring_from_local_search = local_search(pop, rate_local_search, step_size)

    pop = np.append(pop, offspring_from_crossover, axis=0)
    pop = np.append(pop, offspring_from_mutation, axis=0)
    pop = np.append(pop, offspring_from_local_search, axis=0)
    fitness_values, trajectory, done, flipped_fitness_values = evaluation(pop, stan[0], stan[1], stan[2], done)
    best_gols = np.append(best_gols, GOL(flipped_fitness_values, fitness_values))
    pop = selection(pop, fitness_values, pop_size)
    print('iteration', iterations)
    iterations = iterations + 1

final = True
fitness_values, final_trajectory, done, final_flipped_fitness_values = evaluation(pop, stan[0], stan[1], stan[2], done)

positions_x, positions_y, car_angles = best_individuals_visualization(final_trajectory, stan[0], stan[1], stan[2])
index = np.arange(pop.shape[0]).astype(int)
pareto_front_index = pareto_front_finding(fitness_values, index)
pop = pop[pareto_front_index, :]
pareto_front = fitness_values[pareto_front_index]
print("______________")
print("Kryteria optymalizacji:")
print("Odl. od miejsca | Różnica kąta wzgl.")
print("parkingowego    | miejsca parkingowego")
print(fitness_values)
plt.scatter(fitness_values[:, 0], abs(abs(fitness_values[:, 1] * (180 / math.pi)) - 180), marker='x', c='r')
plt.scatter(pareto_front[:, 0], abs(abs(pareto_front[:, 1] * (180 / math.pi)) - 180), marker='x', c='b')
blue_patch = patches.Patch(color='blue', label='Osobniki Pareto Optymalne')
red_patch = patches.Patch(color='red', label='Reszta populacji')
plt.legend(handles=[blue_patch, red_patch])
plt.xlabel('Odległość od miejsca parkingowego w linii prostej [m]')
plt.ylabel('Różnica kąta względem miejsca parkingowego [stopnie]')
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title('Trasa przejazdu optymalnego osobnika')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_xlim(-10, parking_length)
ax.set_ylim(-10, parking_width)
ax.add_patch(patches.Rectangle((0 - parking_spot_length / 2, 0 - parking_spot_width / 2), parking_spot_length,
                               parking_spot_width, edgecolor='black', fill=False))
fig.show()

for m in range(nv):
    xA = positions_x[m] - car_length / 2 * math.cos(car_angles[m]) - car_width / 2 * math.sin(car_angles[m])
    yA = positions_y[m] - car_length / 2 * math.sin(car_angles[m]) + car_width / 2 * math.cos(car_angles[m])

    xB = xA + car_width * math.sin(car_angles[m])
    yB = yA - car_width * math.cos(car_angles[m])

    xD = xA + car_length * math.cos(car_angles[m])
    yD = yA + car_length * math.sin(car_angles[m])

    xC = xB + car_length * math.cos(car_angles[m])
    yC = yB + car_length * math.sin(car_angles[m])

    points = [[xA, yA], [xB, yB], [xC, yC], [xD, yD]]
    car = Polygon(points, fill=None, edgecolor='r')
    ax.add_patch(car)
plt.show()

plot_iterations = np.arange(iterations)
plt.scatter(plot_iterations, best_gols, marker='o', c='g')
plt.title('Najlepszy parametr GOL dla każdej iteracji')
plt.xlabel('Numer iteracji')
plt.ylabel('Parametr GOL')
plt.show()
