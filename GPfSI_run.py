# '''
# @file GPfSI_run.py
# @brief Python file to configure and excute GPfSI 
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
# This is the Python file to configure and execute Genetic Programing for System Identification (GPfSI).
# '''



# Necessary Modules ("numpy", "pandas", "sympy", "time", "multiprocessing", "os", "sys")
# Please install "sympy" library

import numpy as np
import sympy
from sympy import *
import GPfSI



# Input/output data for GPfSI simulation

# If you want to use your own data, please prepare *.npy type files
# In case of symbolic regression,
# input.npy ( number of data * data dimensions ) and output.npy ( number of data * 1 )
# In case of differential equation,
# domain.npy ( space domain, time domain ),
# input.npy ( number of data * data dimensions ), and output.npy ( number of data * 1 )

# One example is here
# ...



# General information of GPfSI simulation

case_number = 2 # integer
# case_number is used for the filename

infomation = True # True or False
# If you want to see relevant information in each step, you can set it True.

core_number = 2 # interger or None (which uses all available cores)
randomseed = 1 # interger or None
saveinterval = 10 # integer
generation_continues = None # integer or None
case_parameters = [ case_number, infomation, core_number, randomseed, saveinterval, generation_continues ]





# Information for a reference system model
# Need user's prior knowledge of the system

model_number = 2 # integer 1 or 2
# If you want to make try Harmonic oscillator equation (an 1D ODE example), set model_number to 1.
# If you want to make try Heat equation (an 1D PDE example), set model_number to 2.

error_level = 0.05 # float number, e.g., 0.01, 0.05, 0.1

reference_model = [ model_number, error_level ]





# Hyperparameters of GPfSI simulation
# Important factors are compelxity penalty coefficient and basis functions

# for fitness test

number_session = 2 # integer 1 or 2
# the number of fitness test sessions

fitnesstype1 = [ False, True, 0.005, 0.1, 1]
fitnesstype2 = [ True, False, 0.005, 0.1, 1] # if number_session is 2
# fitnesstype = [ equation error, solution error, compelxity penalty coefficient,
#                 sampling portion of the data, subbatch number]

termination_condition1 = [ 0.001, 50 ]
termination_condition2 = [ 0.001, 50 ] # if number_session is 2
# termination_condition = [ minimum fitness change, consecutive generation ]

# for expression tree (genetic programming)

method_generation = 'ramped half-and-half' # 'ramped half-and-half', 'grow', or 'full'
operations = [ '+', '-', '*' ]
basis_functions = [ 'Heaviside', 'sign', 'Abs', 'exp', 'erf' , 'sin', 'cos', 'log' ]
# e.g., [ 'Heaviside', 'sign', 'Abs', 'exp', 'erf' , 'sin', 'cos', 'log' ]
# check available options (https://docs.sympy.org/latest/modules/functions/index.html)

if model_number == 1:
    variables = ['X', 'Xt']
else:
    variables = ['X', 'Xx', 'Xxx'] # input variables
# ['X', 'Xt'] if the model number is 1
# ['X', 'Xx', 'Xxx'] if the model number is 2

tree_weight = [5,1,2,2] # integers; node-seminode-variable-number
tree_level = 5 # integer; level limit = 2**tree_level - 1
number_population = 100 # integer
representation_rate = 0.1 # float
crossover_rate = 0.8 # float
mutation_rate = 0.2 # float
evolution_parameters = [ representation_rate, crossover_rate, mutation_rate ]
expressiontree_parameters = [ method_generation, operations, basis_functions, variables,
                             tree_weight, tree_level, number_population, evolution_parameters ]


# for coefficient optimization (genetic algorithm)

GA_frequency_optimization = 50
GA_apply_equations = 20
GA_number_population = 30
GA_tolerance_function = [0.01, 30]
training_parameters = [ number_session, [fitnesstype1, fitnesstype2],
                       [termination_condition1, termination_condition2],
                       GA_frequency_optimization, GA_apply_equations, GA_number_population, GA_tolerance_function]





# Excution the whold GPfSI simulation with one function

GPfSI_results = GPfSI.GPfSI_allinone(case_parameters, reference_model, training_parameters, expressiontree_parameters)





# Plot results

case = 2
session = 2

saved_data = np.load(f'results/final_case_{case}_session_{session}.npy', allow_pickle = True)
generations = saved_data[0]
bestequations = saved_data[1]
bestfitness = saved_data[2]
generation = generations[-1]
minimums_bestfitness = saved_data[3]
equations = saved_data[4][0]
fitness_info = saved_data[4][1]
computationaltimes = saved_data[5]
generation_offsets = saved_data[6]
np.random.set_state(saved_data[7])

saved_configuration = np.load(f'results/initial_case_{case}_configuration.npy', allow_pickle = True)
case_parameters = saved_configuration[0]
reference_model = saved_configuration[1]
training_parameters = saved_configuration[2]
expressiontree_parameters = saved_configuration[3]


# Loss function over time

mildstones=[]    
print('each seesion end')   
for i in generation_offsets:
    mildstones.append(bestequations[i-1])
    print(f'best at gen. {i}: {bestequations[i-1].eqn_print()} // {bestfitness[i-1]}')
    
complexity = []
for j in range(len(bestequations)):
    complexity.append(GPfSI_et.counting_components(bestequations[j]))
complexity = np.array(complexity)
complexityterm = complexity*training_parameters[1][0][2]
rest = bestfitness - complexityterm

ymin = 1e-3
ymax = 1e1
fig, ax = plt.subplots(figsize=(10,3))
plt.vlines(generation_offsets, ymin, ymax, linewidth = 3, colors='cyan', label='Phase ends')
plt.plot(generations, bestfitness, linewidth = 2, linestyle = '-', alpha = 0.8, color='k', label=r'Loss function')
plt.plot(generations, rest, linewidth = 2, linestyle = ':', alpha = 0.8, color='k', label=r'Training error')
plt.plot(generations, complexityterm, linewidth = 2, alpha = 0.8, color='b', label=r'Complexity penalty')
plt.xlim(0,len(generations))
plt.ylim(ymin, ymax)
plt.yscale('log')
plt.xlabel(r'Genration [-]', fontsize=18, fontname='Arial')
plt.ylabel(r'Value [-]', fontsize=18, fontname='Arial')
plt.xticks(fontsize=16, fontname='Arial')
plt.yticks(fontsize=16, fontname='Arial')
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.tick_params(which="major", direction="in", right=True, top=True, length=6, width=1)
ax.tick_params(which="minor", direction="in", right=True, top=True, length=4, width=1)
plt.legend(prop={'family': 'Arial', 'size': 12}, loc=1)
plt.tight_layout()
plt.savefig(f'results/lossfunction_case_{case}.pdf',dpi=200, bbox_inches='tight')
plt.show()

# Top 5 models

print('top 10 models w/ loss function values')
for i in range(10):
    print(f'{i+1}: {equations[i].eqn_print()} // {fitness_info[i]}')

# texts save to csv    
    
with open(f'results/result_outline_case_{case}.csv', 'w', newline='') as csvfile:
    result_csv = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    result_csv.writerow(['Model 1 is r(X, Xt) in "f(t) = m*Xtt + r(X, Xt)"'])
    result_csv.writerow(['Model 2 is M(X, Xx, Xxx) in "Xt = M(X, Xx, Xxx)"'])
    result_csv.writerow([' '])
    result_csv.writerow(['each seesion end'])
    for i in generation_offsets:
        result_csv.writerow([f'best at gen. {i}: {bestequations[i-1].eqn_print()} // {bestfitness[i-1]}'])
    result_csv.writerow([' '])
    result_csv.writerow(['top 10 models w/ loss function values'])
    for i in range(10):
        result_csv.writerow([f'{i+1}: {equations[i].eqn_print()} // {fitness_info[i]}'])
    csvfile.close()    
    