# '''
# @file GPfSI_optimization.py
# @brief Subcode to optimize coefficients in models
# Copyright (C) 2021 Jinwoo Im
# Sonny Astani Department of Civil and Environmental Engineering, University of Southern California, Los Angeles, USA
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# 
# This is the subcode of GPfSI to optimize coefficients in models.
# '''

import numpy as np
import GPfSI_data as GPfSI_dt
import GPfSI_expressiontree as GPfSI_et
import sympy
from sympy import *
from multiprocessing import Pool
import os
import copy

# Enviroment initialization
variable_symbols = 'X, Xx, Xxx, Xt'
X, Xx, Xxx, Xt = sympy.symbols(variable_symbols)
ignore_errors = np.seterr(all='ignore')

def GA_optimization(equations, training_data, fitnesstype, datatype_weight,
                    GA_number_population, GA_tolerance_function,
                    representation_rate, crossover_rate, mutation_rate,
                    pool_opt, info):
    if isinstance(equations, list) != True:
        equations = [equations]
    optimized_equations = []
    model_number = training_data[-1]

    if info == True:
        print('GA_Optimization for', len(equations), 'equations with',
              GA_tolerance_function[0], 'is the maximum of minimum fitness changes in',
              GA_tolerance_function[1], 'consecutive generations')
    number_equations = len(equations)

    if model_number == 1:
        normalization = np.sqrt(np.sum(training_data[1]**2, axis=1)/len(training_data[1][0].flatten()))
        space = [0]
    elif model_number == 2:
        normalization = np.sqrt(np.sum(training_data[1]**2, axis=(1,2))/len(training_data[1][0].flatten()))
        space = training_data[0][0]
    else:
        sys.exit('wrong model number')

    subdata_number = int(len(training_data[0][-1])*fitnesstype[3])
    subbatch_number = fitnesstype[4]
    subbatchdata_number = int(subdata_number/subbatch_number)

    data_shape = training_data[1].shape
    subdata_limit = data_shape[1] - subbatchdata_number - 1
    if subdata_limit > 0:
        subdata_index = np.random.randint( 0, subdata_limit, subbatch_number )
    else:
        subdata_index = np.zeros(1, dtype=int)
        subbatchdata_number = data_shape[1]

    training_subdata_batch = []
    times_batch = []
    for i in range(subbatch_number):
        training_subdata_batch.append(training_data[1]
                                      [:,subdata_index[i]:(subdata_index[i]+subbatchdata_number)])
        if model_number == 1:
            times_batch.append(training_data[0]
                               [:,subdata_index[i]:(subdata_index[i]+subbatchdata_number)])
        elif model_number == 2:
            times_batch.append(training_data[0][-1]
                               [subdata_index[i]:(subdata_index[i]+subbatchdata_number)])

    if pool_opt == 1:
        if info == True:
            print('with a single process running ...')
        for i in range(number_equations):
            optimized_equations.append(GA_optimization_one(equations[i], i+1,
                                                           training_subdata_batch, space, times_batch,
                                                           fitnesstype, normalization, datatype_weight,
                                                           GA_number_population, GA_tolerance_function,
                                                           representation_rate, crossover_rate, mutation_rate, info))

    else:
        if pool_opt == None:
            pool_opt = os.cpu_count()
        if info == True:
            print('with',pool_opt,'processors running ...')
        with Pool(pool_opt) as p:
            pool_results = p.starmap_async(GA_optimization_one,
                                           zip(equations, np.arange(1,number_equations+1),
                                               [training_subdata_batch]*number_equations,
                                               [space]*number_equations,
                                               [times_batch]*number_equations,
                                               [fitnesstype]*number_equations,
                                               [normalization]*number_equations,
                                               [datatype_weight]*number_equations,
                                               [GA_number_population]*number_equations,
                                               [GA_tolerance_function]*number_equations,
                                               [representation_rate]*number_equations,
                                               [crossover_rate]*number_equations,
                                               [mutation_rate]*number_equations,
                                               [info]*number_equations), chunksize=1)

            optimized_equations = pool_results.get()
            p.close()
            p.join()
            del p

    return optimized_equations

def GA_optimization_one(GA_equation, GA_equation_number,
                        training_subdata_batch, space, times_batch,
                        fitnesstype, normalization, datatype_weight,
                        GA_number_population, GA_tolerance_function,
                        representation_rate, crossover_rate, mutation_rate, info):

    optimized_equations = []
    info_eq = GPfSI_et.interpretation(GA_equation)

    if len(info_eq[3]) != 0:
        success = True
        coefficients = random_coefficient_generator(GA_equation, GA_number_population, fitnesstype)
        coefficients, GA_fitness_info = GA_fitness_test(GA_equation, GA_equation_number, coefficients,
                                                        training_subdata_batch, space, times_batch,
                                                        fitnesstype, normalization, datatype_weight, False)

        generation = 1
        fitness_thread = [ GA_fitness_info[0] ]
        fitness_minimum = [ GA_fitness_info[0] ]
        if info == True:
            info_GA_equation = GPfSI_et.interpretation(GA_equation)
            info_GA_equation[3] = coefficients[0]
            eq = GPfSI_et.change_numbers(GA_equation, info_GA_equation)
            print('equation', GA_equation_number, 'starts with', sympify(eq.eqn_print()), round(fitness_thread[-1],6))

        while True:
            coefficients = GA_evolutionary_operators(coefficients, GA_fitness_info, GA_number_population, fitnesstype,
                                                     representation_rate, crossover_rate, mutation_rate, False)
            coefficients, GA_fitness_info = GA_fitness_test(GA_equation, GA_equation_number, coefficients,
                                                            training_subdata_batch, space, times_batch,
                                                            fitnesstype, normalization, datatype_weight, False)

            generation += 1
            fitness_thread.append(GA_fitness_info[0])
            fitness_minimum.append(np.min(fitness_thread))
            if info == True:
                info_GA_equation[3] = coefficients[0]
                eq = GPfSI_et.change_numbers(GA_equation, info_GA_equation)
#                 print('generation', generation, 'best one', sympify(eq.eqn_print()), round(fitness_thread[-1],6))

            if generation > GA_tolerance_function[1]:
                change_fitness_minimums = np.diff(fitness_minimum
                                                  [(generation-1-GA_tolerance_function[1]):(generation-1)])
                abs_change_fitness_minimums = abs(change_fitness_minimums)
                maximum_change_fitness_minimums = round(np.max(abs_change_fitness_minimums),6)
                if info == True:
                    print('maximum_change_fitness', maximum_change_fitness_minimums, 'over',
                          GA_tolerance_function[1], 'consecutive generations')
                if maximum_change_fitness_minimums <= GA_tolerance_function[0]:
                    break
                if generation > 10*GA_tolerance_function[1]:
                    success = False
                    break

        info_eq[3] = coefficients[0]
        optimized_equations.append(GPfSI_et.change_numbers(GA_equation, info_eq))
        if info == True:
            if success == True:
                print('equation', GA_equation_number, 'is done with fitness', round(fitness_thread[-1],6),
                      'at generation', generation, '\n')
            else:
                print('equation', GA_equation_number, 'fails at generation', generation, '\n')

    else:
        optimized_equations.append(GA_equation)
        if info == True:
            print('equation',GA_equation_number,'no coefficient \n')

    return optimized_equations[0]

def random_coefficient_generator(GA_equation, GA_number_population, fitnesstype):
    coefficients = []
    info = GPfSI_et.interpretation(GA_equation)
    number_numbers = len(info[3])
    coefficients.append(info[3])
    for j in range(1, GA_number_population):
        individual = []
        for i in range(number_numbers):
            if fitnesstype[1] == True:
#                 coeff = round(coefficients[0][i][1]*float(np.exp(np.random.uniform(np.log(0.05), np.log(20)))), 6)
                coeff = round(coefficients[0][i][1]*np.random.uniform(0.05, 20), 6)
            else:
                coeff = round(coefficients[0][i][1]*np.random.uniform(0.05, 20), 6)
            while abs(coeff) >= 1e3:
                coeff = round(coeff*0.1, 6)
            individual.append([coefficients[0][i][0], coeff])
        coefficients.append(individual)
    return coefficients

def GA_fitness_test(GA_equation, GA_equation_number, coefficients, training_subdata_batch, space, times_batch,
                    fitnesstype, normalization, datatype_weight, info):
    function_errors = []
    response_errors = []
    coefficients_error = []
    number_population = len(coefficients)
    tree_info = GA_equation.tree_info
    info_GA_equation = GPfSI_et.interpretation(GA_equation)
    interation_uni = 0
    while number_population-1 > interation_uni:
        index_same = []
        new_coefficients = []
        for i in range(interation_uni+1, number_population):
            if coefficients[interation_uni] == coefficients[i]:
                index_same.append(i)
        for i in range(number_population):
            if i not in index_same:
                new_coefficients.append(coefficients[i])
        coefficients = new_coefficients
        number_population = len(coefficients)
        interation_uni += 1
    tem_equations = []

    for i in coefficients:
        info_GA_equation[3] = i
        tem_equations.append(GPfSI_et.change_numbers(GA_equation, info_GA_equation))

    if fitnesstype[0] == 1:
        for i in range(number_population):
            tem_errors = GPfSI_dt.function_data_error(tem_equations[i], i+1,
                                                      training_subdata_batch, normalization,
                                                      space, times_batch,
                                                      False, info)
            function_errors.append(tem_errors)

    if fitnesstype[1] == 1:
        for i in range(number_population):
            tem_errors = GPfSI_dt.response_data_error(tem_equations[i], i+1,
                                                     training_subdata_batch, normalization,
                                                     datatype_weight, space, times_batch, False, info)
            response_errors.append(tem_errors)


    for i in range(number_population):
        multiobjective = 0
        if fitnesstype[0] == True:
            multiobjective += function_errors[i]
        if fitnesstype[1] == True:
            multiobjective += response_errors[i]
        coefficients_error.append([round(multiobjective, 6), coefficients[i]])

    coefficients_error = sorted(coefficients_error, key=lambda temp: temp[0])
    GA_fitness_info = [i[0] for i in coefficients_error]
    coefficients = [i[1] for i in coefficients_error]
    return coefficients, GA_fitness_info

def GA_evolutionary_operators(coefficients, GA_fitness_info, GA_number_population, fitnesstype,
                              representation_rate, crossover_rate, mutation_rate, info):
    number_population = len(coefficients)
    number_representation = int(np.ceil(GA_number_population*representation_rate))
    if number_population < number_representation:
        number_representation = number_population

    evolved_coefficients = []
    for i in range(number_representation):
        evolved_coefficients.append(copy.deepcopy(coefficients[i]))
    if len(coefficients[0]) > 1:
        while len(evolved_coefficients) < GA_number_population:
            parent1 = copy.deepcopy(GPfSI_et.tounament_selection(coefficients, GA_fitness_info))
            parent2 = copy.deepcopy(GPfSI_et.tounament_selection(coefficients, GA_fitness_info))
            child1 = parent1
            child2 = parent2
            if np.random.random() < crossover_rate:
                child1, child2 = GA_crossover(parent1, parent2)
            if np.random.random() < mutation_rate:
                child1 = GA_mutate(parent1, fitnesstype)
                child2 = GA_mutate(parent2, fitnesstype)
            evolved_coefficients.append(child1)
            if len(evolved_coefficients) < number_population:
                evolved_coefficients.append(child2)
    else:
        while len(evolved_coefficients) < GA_number_population:
            parent1 = copy.deepcopy(GPfSI_et.tounament_selection(coefficients, GA_fitness_info))
            child1 = parent1
            child1 = GA_mutate(parent1, fitnesstype)
            evolved_coefficients.append(child1)
    return evolved_coefficients

def GA_crossover(parent_1, parent_2):
    index = np.random.randint(len(parent_1))
    child_1 = parent_1[:index] + parent_2[index:]
    child_2 = parent_2[:index] + parent_1[index:]
    return child_1, child_2

def GA_mutate(parent_1, fitnesstype):
    mutate_index = np.random.randint(len(parent_1))
    if fitnesstype[1] == True:
#         coeff = round(parent_1[mutate_index][1]*float(np.exp(np.random.uniform(np.log(0.05), np.log(20)))), 6)
        coeff = round(parent_1[mutate_index][1]*np.random.uniform(0.05, 20), 6)
    else:
        coeff = round(parent_1[mutate_index][1]*np.random.uniform(0.05, 20), 6)
    while abs(coeff) >= 1e3:
        coeff = round(coeff*0.1, 6)
    parent_1[mutate_index][1] = coeff
    return parent_1
