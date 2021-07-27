# '''
# @file GPfSI.py
# @brief Main code to excute GPfSI
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
# This is the main code of Genetic Programing for System Identification (GPfSI).
# '''


import numpy as np
import GPfSI_data as GPfSI_dt
import GPfSI_expressiontree as GPfSI_et
import GPfSI_optimization as GPfSI_ot
import time
import os
import sys
import sympy
from sympy import *

# Enviroment initialization
variable_symbols = 'X, Xx, Xxx, Xt'
X, Xx, Xxx, Xt = sympy.symbols(variable_symbols)
ignore_errors = np.seterr(all='ignore')
if not os.path.exists('results'):
    os.makedirs('results')

def model_information(reference_model):

    model_number = reference_model[0]

    if model_number == 1:
        nt = int(1/1e-4)
        dt = 1/nt
        amplitude = 1
        series = 30

        datatype_weight = [1,0,0]
        label = r'$x$ [-]'
        rhs = 10*X
        parameters = [nt, dt, amplitude, series]

    elif model_number == 2:
        nt = int(1/1e-4)
        dt = 1/nt
        nx = 128
        dx = 1/nx
        mu = 0.5
        sigma = 0.05

        datatype_weight = [1,0,0,0]
        label = r'$c$ [-]'
        rhs = 1 / 100 * Xxx
        parameters = [nt, dt, nx, dx, mu, sigma]

    else:
        sys.exit('wrong model number')

    model_info = [model_number, rhs, parameters, label]

    return model_info, datatype_weight

def GPfSI_allinone(case_parameters, reference_model, training_parameters, expressiontree_parameters):

    # GPfSI simulation information
    case = case_parameters[0]
    info = case_parameters[1]
    core_number = case_parameters[2]
    randomseed = case_parameters[3]
    saveinterval = case_parameters[4]
    generation_continues = case_parameters[5]

    pool_fit = core_number
    pool_opt = core_number
    if info == True:
        print(f'GPfSI case {case} initializaiton with {core_number} core(s)')

    # Reference model information
    model_number = reference_model[0]
    error_level = reference_model[1]
    model_info, datatype_weight = model_information(reference_model)

    # GPfSI simulation hyperparameters
    # Fitness test
    number_session = training_parameters[0]
    fitnesstypes = training_parameters[1]
    termination_conditions = training_parameters[2]

    # Expression tree
    method_generation = expressiontree_parameters[0]
    node = expressiontree_parameters[1]
    pseudo_node = expressiontree_parameters[2]
    variables = expressiontree_parameters[3]
    tree_weight = expressiontree_parameters[4]
    tree_level = expressiontree_parameters[5]
    number_population = expressiontree_parameters[6]
    evolution_parameters = expressiontree_parameters[7]

    nodenumber_max = 2**tree_level - 1
    representation_rate = evolution_parameters[0]
    crossover_rate = evolution_parameters[1]
    mutation_rate = evolution_parameters[2]
    tree_components = [node, pseudo_node, variables]
    tree_info = [tree_components, tree_weight, tree_level]

    # Coefficient optimization
    GA_frequency_optimization = training_parameters[3]
    GA_apply_equations = training_parameters[4]
    GA_number_population = training_parameters[5]
    GA_tolerance_function = training_parameters[6]

    # GPfSI execution
    # trial setting
    np.random.seed(randomseed)
    if info == True:
        print(f'GPfSI starts with random seed {randomseed}')

    training_data = GPfSI_dt.reference_data_generation(model_info, error_level, datatype_weight, True)


    # generation setting
    if generation_continues == None:
        if info == True:
            print(f'Generation 1 initialized ...')
        equations = GPfSI_et.random_equation_generator(number_population, tree_info, method_generation, True)
        equations = GPfSI_et.remove_repetition(equations)
        equations, fitness_info = GPfSI_dt.fitness_test(equations, training_data, datatype_weight,
                                                        fitnesstypes[0], pool_fit, True, True)
        generation = 1
        generations = [ generation ]
        bestequations = [ equations[0] ]
        bestfitness = [ fitness_info[0] ]
        minimums_bestfitness = [ min(bestfitness) ]
        initial_time = time.time()
        computationaltimes = [ initial_time ]
        generation_offsets = [ 1 ]

        GPfSI_data = [generations, bestequations, bestfitness, minimums_bestfitness, [equations, fitness_info],
                     computationaltimes, generation_offsets, np.random.get_state()]

        GPfSI_configuration = [ case_parameters, reference_model, training_parameters, expressiontree_parameters ]
        np.save( f'results/case_{case}_generation_{generation}', GPfSI_data )
        np.save( f'results/initial_case_{case}_configuration', GPfSI_configuration )
        if info == True:
            print(f'Generation {generation}')
            print('Best 5 equations')
            for i in range(5):
                print(f'{sympify(equations[i].eqn_print())} / {fitness_info[i]}')
            print('')
    else:
        if info == True:
            print(f'Case {case} / Generation {generation_continues} continued ...')
        saved_data = np.load(f'results/case_{case}_generation_{generation_continues}.npy', allow_pickle = True)
        generations = saved_data[0]
        generation = generations[-1]
        bestequations = saved_data[1]
        bestfitness = saved_data[2]
        minimums_bestfitness = saved_data[3]
        equations = saved_data[4][0]
        fitness_info = saved_data[4][1]
        computationaltimes = saved_data[5]
        generation_offsets = saved_data[6]
        np.random.set_state(saved_data[7])
        initial_time = time.time() - computationaltimes[-1]
        generation_continues = None
        if info == True:
            print(f'Generation {generation} / Time laps {round(initial_time)} s')
            print('Best 5 equations')
            for i in range(5):
                print(f'{sympify(equations[i].eqn_print())} / {fitness_info[i]}')
            print('')

    # fitness session setting
    for session in range(len(generation_offsets)-1, number_session):
        termination = 0
        if info == True:
            print(f'Phase {session+1} starts at generation {generation}')
            print(f'Fitness type {fitnesstypes[session]} & termination condition {termination_conditions[session]}')
            print(f'Best equation {sympify(equations[0].eqn_print())} with fitness value {fitness_info[0]}\n')

        # evolution starts
        while True:
            if info == True:
                print(f'Generation {generation}: evolution')
            equations = GPfSI_et.evolutionary_operators(equations, fitness_info,
                                                       representation_rate, crossover_rate, mutation_rate, nodenumber_max, False)

            if (generation+1) % GA_frequency_optimization == 0:
                equations, fitness_info = GPfSI_dt.fitness_test(equations, training_data,
                                                               datatype_weight, fitnesstypes[session], pool_fit, False, False)
                if info == True:
                    tloc = time.localtime()
                    current_time = time.strftime("%H:%M:%S", tloc)
                    print('Generation', generation,'optimization', generation+1, current_time)
                equations[:GA_apply_equations] = GPfSI_ot.GA_optimization(equations[:GA_apply_equations],
                                                                         training_data, fitnesstypes[session],
                                                                         datatype_weight, GA_number_population,
                                                                         GA_tolerance_function, representation_rate,
                                                                         crossover_rate, mutation_rate,
                                                                         pool_opt, True)
            if info == True:
                print(f'Generation {generation}: fitness test')
            equations, fitness_info = GPfSI_dt.fitness_test(equations, training_data,
                                                         datatype_weight, fitnesstypes[session], pool_fit, False, False)
            generation += 1
            midtime = time.time() - initial_time

            # saving data each saving interval
            generations.append(generation)
            bestequations.append(equations[0])
            bestfitness.append(fitness_info[0])
            minimums_bestfitness.append(min(bestfitness[generation_offsets[-1]:]))
            computationaltimes.append(midtime)
            if generation % saveinterval == 0:
                GPfSI_data = [generations, bestequations, bestfitness, minimums_bestfitness, [equations, fitness_info],
                             computationaltimes, generation_offsets, np.random.get_state()]
                np.save( f'results/case_{case}_generation_{generation}', GPfSI_data )
                if info == True:
                    print(f'Generation {generation} / Time laps {round(midtime)} s')
                    print('Best 5 equations')
                    for j in range(5):
                        print(f'{sympify(equations[j].eqn_print())} / {fitness_info[j]}')
                    print('')

            # convergence check
            if generation - generation_offsets[-1] > termination_conditions[session][1]:
                change_minimums_bestfitness = np.diff(minimums_bestfitness[(generation-generation_offsets[-1]-
                                                                            termination_conditions[session][1])
                                                                           :(generation-generation_offsets[-1])])
                abs_change_minimums_bestfitness = abs(change_minimums_bestfitness)
                maximum_change_minimums_bestfitness = round(np.max(abs_change_minimums_bestfitness),6)
                if maximum_change_minimums_bestfitness <= termination_conditions[session][0]:
                    phase_end = 'success'
                    termination = 1
                if generation - generation_offsets[-1] > 10*termination_conditions[session][1]:
                    phase_end = 'failure'
                    termination = 1
            if termination != 0:
                if info == True:
                    print(f'Phase {session+1} ends with {phase_end} at generation {generation}')
                    print(f'Best equation {sympify(equations[0].eqn_print())} with fitness value {fitness_info[0]}')
                    generation_offsets.append(generation)
                break

        # data saving each session
        GPfSI_data = [generations, bestequations, bestfitness, minimums_bestfitness, [equations, fitness_info],
                     computationaltimes, generation_offsets, np.random.get_state()]
        np.save( f'results/final_case_{case}_session_{session+1}', GPfSI_data )

    return GPfSI_data
