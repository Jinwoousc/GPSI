# '''
# @file GPfSI_data.py
# @brief Subcode to generate and compute data
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
# This is the subcode of GPfSI to generate and compute data.
# '''

import numpy as np
import GPfSI_expressiontree as GPfSI_et
import sympy
import scipy
from sympy import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import os
import sys

# Enviroment initialization
variable_symbols = 'X, Xx, Xxx, Xt'
X, Xx, Xxx, Xt = sympy.symbols(variable_symbols)
ignore_errors = np.seterr(all='ignore')

def reference_data_generation(model_info, error_level, datatype_weight, info):
    model_number = model_info[0]
    rhs = model_info[1]
    parameters = model_info[2]
    label = model_info[3]

    if model_number == 1:
        parameters = model_info[2]
        nt = parameters[0]
        dt = parameters[1]
        amplitude = parameters[2]
        series = parameters[3]
        k = np.arange(1,series+1)
        random_translation = np.random.uniform(0, 2*np.pi, series)
        t = np.linspace(0,nt*dt,nt+1)

        # excitation data generation
        excitation_raw = np.sum(np.sin(2*np.pi*k[:, None]/1*t[None, :] + random_translation[:, None]), axis=0)
        excitation_normalized = amplitude*excitation_raw/np.std(excitation_raw)
        excitation_data = np.vstack((t, excitation_normalized))
        domain = excitation_data

        if info == True:
            print('Initial condition for training data')
            fig, ax = plt.subplots(figsize=(5,4))
            plt.plot(excitation_data[0], excitation_data[1], color='black')
            plt.xlabel('t [-]', fontsize=14)
            plt.ylabel('Excitation Force [-]', fontsize=14)
            plt.xlim(0,1)
            plt.ylim(-4,4)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.04))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.4))
            plt.xticks(fontsize=14, fontname='Arial')
            plt.yticks(fontsize=14, fontname='Arial')
            ax.tick_params(which="major", direction="in", right=True, top=True, length=5, pad=7)
            ax.tick_params(which="minor", direction="in", right=True, top=True, length=3)
            plt.tight_layout()
            plt.show()

        rhs_l = lambdify([X, Xt], rhs,
                         modules=[{'Heaviside': lambda x: np.heaviside(x,0.5)},
                                  'numpy','scipy'])
        responses, success = rk4_integration_1d_ode(rhs_l, np.array([0,0]), excitation_data[1], t, info)

        if info == True:
            print(success)
            print('Reference responses')
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(t, responses[0,:], color='k', linestyle='-', linewidth=3, alpha=0.9, label='t = '+str(t[-1]))
            plt.xlabel(r'$t$ [-]', fontsize=16, fontname='Arial')
            plt.ylabel(label, fontsize=16, fontname='Arial')
            plt.xlim(0,1)
            plt.ylim(-4,4)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.04))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.4))
            plt.xticks(fontsize=14, fontname='Arial')
            plt.yticks(fontsize=14, fontname='Arial')
            ax.tick_params(which="major", direction="in", right=True, top=True, length=5, pad=7)
            ax.tick_params(which="minor", direction="in", right=True, top=True, length=3)
            plt.tight_layout()
            plt.show()

            training_data = np.zeros(responses.shape)
            data_shape = responses[0].shape
            root_mean_square = np.sqrt(np.sum(responses**2, axis=1)/len(responses[0].flatten()))

    elif model_number == 2:
        nt = parameters[0]
        dt = parameters[1]
        nx = parameters[2]
        dx = parameters[3]
        mu = parameters[4]
        sigma = parameters[5]

        t = np.linspace(0,nt*dt,nt+1)
        x = np.arange(dx/2,1 + dx/2,dx)
        domain = [x, t]
        Xv = np.zeros((len(x),len(t)))
        Xv[:,0]=1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2)/4/10*3
        Xv[:,0]/=np.max(Xv[:,0])

        if info == True:
            print('Initial condition for training data')
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(x, Xv[:,0], color='k', linestyle='-', linewidth=3, label='t = 0')
            plt.xlabel(r'$x$ [-]', fontsize=16, fontname='Arial')
            plt.ylabel(label, fontsize=16, fontname='Arial')
            plt.xlim(0,1)
            plt.ylim(0,1)
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.04))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.04))
            plt.xticks(fontsize=14, fontname='Arial')
            plt.yticks(fontsize=14, fontname='Arial')
            ax.tick_params(which="major", direction="in", right=True, top=True, length=5, pad=7)
            ax.tick_params(which="minor", direction="in", right=True, top=True, length=3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show()

        lhs = lambdify([X, Xx, Xxx], rhs,
                       modules=[{'Heaviside': lambda x: np.heaviside(x,0.5)},
                                'numpy','scipy'])
        responses, success = rk4_integration_1d(lhs, Xv[:,0], x, t, info, 'spectral')

        if info == True:
            print(success)
            print('Reference responses')
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(x, responses[0,0,:], color='lightgray', linestyle='-', linewidth=3, alpha=0.9, label='t = 0')
            ax.plot(x, responses[0,int(nt/4),:], color='darkgray', linestyle='-', linewidth=3, alpha=0.9, label='t = '+str(t[int(nt/4)]))
            ax.plot(x, responses[0,int(nt/4*2),:], color='gray', linestyle='-', linewidth=3, alpha=0.9, label='t = '+str(t[int(nt/4*2)]))
            ax.plot(x, responses[0,int(nt/4*3),:], color='dimgray', linestyle='-', linewidth=3, alpha=0.9, label='t = '+str(t[int(nt/4)*3]))
            ax.plot(x, responses[0,-1,:], color='k', linestyle='-', linewidth=3, alpha=0.9, label='t = '+str(t[-1]))
            plt.xlabel(r'$x$ [-]', fontsize=16, fontname='Arial')
            plt.ylabel(label, fontsize=16, fontname='Arial')
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xticks(fontsize=14, fontname='Arial')
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(MultipleLocator(0.04))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.04))
            plt.yticks(fontsize=14, fontname='Arial')
            ax.tick_params(which="major", direction="in", right=True, top=True, length=5, pad=7)
            ax.tick_params(which="minor", direction="in", right=True, top=True, length=3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show()

            training_data = np.zeros(responses.shape)
            data_shape = responses[0].shape
            root_mean_square = np.sqrt(np.sum(responses**2, axis=(1,2))/len(responses[0].flatten()))

    else:
        sys.exit('wrong model number')

    for i in range(len(responses)):
        training_data[i] = responses[i] + root_mean_square[i] * np.random.normal(0, error_level, data_shape)
        if info == True:
            print('Error calculation ... (', str(i+1),'/',len(responses),')')
    if info == True:
        print('Error level in training data is',
              round(calculate_error(training_data, responses, datatype_weight, root_mean_square)*100,1),'%')

    return [domain, training_data, model_number]


def rk4_integration_1d(function, initials, space, times, info, differentiation):
    success = True
    t = times
    dt = times[1] - times[0]
    nt = len(t)
    L = space[0]+space[-1]
    nx = len(space)

    iteration = 0
    processing = 0

    if differentiation == 'spectral':
        kx = (2*np.pi/L)*np.append(np.arange(0,(nx/2)), np.arange(-(nx/2), 0))
        kx2 = kx**2
        fftX = np.fft.fft(initials)
        dXdx = (np.fft.ifft(1j*kx*fftX)).real
        ddXdxdx = (np.fft.ifft(-1*kx2*fftX)).real
        responses = np.zeros((4, nt, nx))
        nextresponse = np.asarray([initials, dXdx, ddXdxdx])
        responses[0,iteration] = nextresponse[0]
        responses[1,iteration] = nextresponse[1]
        responses[2,iteration] = nextresponse[2]
        responses[-1, iteration] = function( nextresponse[0], nextresponse[1], nextresponse[2] )

    elif differentiation == 'finite-difference':
        #  boundary conditions
        left_initials = np.insert(initials,0,0)
        left_initials_right = np.insert(left_initials,len(left_initials),left_initials[-1])
        dXdx_extended = np.gradient(left_initials_right, L/nx, edge_order=1)
        ddXdxdx_extended = np.gradient(dXdx_extended, L/nx, edge_order=1)
        nextresponse = np.asarray([left_initials_right, dXdx_extended, ddXdxdx_extended])
        responses = np.zeros((4, nt, nx))
        responses[0,iteration] = nextresponse[0,1:-1]
        responses[1,iteration] = nextresponse[1,1:-1]
        responses[2,iteration] = nextresponse[2,1:-1]
        responses[-1, iteration] = function( nextresponse[0,1:-1], nextresponse[1,1:-1], nextresponse[2,1:-1] )

    while iteration < len(t) - 1:
        g1 = dt/6*np.array(function( nextresponse[0], nextresponse[1], nextresponse[2] ))
        g2 = dt/3*np.array(function( nextresponse[0] + g1*dt/2, nextresponse[1] + g1*dt/2, nextresponse[2] + g1*dt/2 ))
        g3 = dt/3*np.array(function( nextresponse[0] + g2*dt/2, nextresponse[1] + g2*dt/2, nextresponse[2] + g2*dt/2 ))
        g4 = dt/6*np.array(function( nextresponse[0] + g3*dt, nextresponse[1] + g3*dt, nextresponse[2] + g3*dt ))
        deltaresponse = g1+g2+g3+g4

        if np.isnan(deltaresponse).any() or np.isinf(deltaresponse).any() or (np.abs(deltaresponse) >= 1e10).any():
            iteration = float('inf')
            success = False
            break

        if differentiation == 'spectral':
            nextresponse[0] = nextresponse[0] + deltaresponse
            fftX = np.fft.fft(nextresponse[0])
            nextresponse[1] = (np.fft.ifft(1j*kx*fftX)).real
            nextresponse[2] = (np.fft.ifft(-1*kx2*fftX)).real
            responses[:-1, iteration+1] = nextresponse
            responses[-1, iteration+1] = function( nextresponse[0], nextresponse[1], nextresponse[2] )

        elif differentiation == 'finite-difference':
            nextresponse[0] = nextresponse[0] + deltaresponse
            #  boundary conditions
            nextresponse[0,0] = 0
            nextresponse[0,-1] = nextresponse[0,-2]
            nextresponse[1] = np.gradient(nextresponse[0], L/nx, edge_order=1)
            nextresponse[2] = np.gradient(nextresponse[1], L/nx, edge_order=1)
            responses[:-1, iteration+1] = nextresponse[:,1:-1]
            responses[-1, iteration+1] = function( nextresponse[0,1:-1], nextresponse[1,1:-1], nextresponse[2,1:-1] )

        if info == True:
            if round((iteration+1)/(len(t)-1)*100) >= processing:
                print('Processing responses: ', processing, '%')
                processing += 20
        iteration = iteration + 1

    if info == True:
        print(success)
    return responses, success

def rk4_integration_1d_ode(function_wo, initials, excitation, times, info):
    success = True
    t = times
    dt = times[1] - times[0]
    t_index_ini = t[0]*1/dt
    nt = len(t)
    def function(u, t):
        return np.asarray([ u[1] , (excitation[int(round(t/dt-t_index_ini,1))]-function_wo(u[0],u[1]))/0.001 ])

    iteration = 0
    processing = 0

    responses = np.zeros((3, nt))
    nextresponse = initials
    responses[:2,0] = initials
    responses[-1,0] = function( nextresponse, t[iteration] )[1]

    while iteration < len(t) - 1:
        g1 = dt/6*np.array(function( nextresponse, t[iteration] ))
        g2 = dt/3*np.array(function( nextresponse + g1*dt/2, t[iteration] + dt/2 ))
        g3 = dt/3*np.array(function( nextresponse + g2*dt/2, t[iteration] + dt/2 ))
        g4 = dt/6*np.array(function( nextresponse + g3*dt, t[iteration] + dt ))
        deltaresponse = g1+g2+g3+g4

        if np.isnan(deltaresponse).any() or np.isinf(deltaresponse).any() or (np.abs(deltaresponse) >= 1e10).any():
            iteration = float('inf')
            success = False
            break

        nextresponse = nextresponse + deltaresponse
        responses[:-1, iteration+1] = nextresponse
        responses[-1, iteration+1] = function( nextresponse, t[iteration+1] )[1]

        if info == True:
            if round((iteration+1)/(len(t)-1)*100) >= processing:
                print('Processing responses: ', processing, '%')
                processing += 20
        iteration = iteration + 1
    return responses, success


def fitness_test(equations, training_data, datatype_weight, fitnesstype, pool_fit, first_generation, info):
    if isinstance(equations, list) == False:
        equations = [equations]
    number_population = len(equations)
    function_errors = []
    response_errors = []
    equations_error = []
    model_number = training_data[-1]

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

    if info == True:
        print('Fitness test for ', number_population, ' equations')
        fitnesstype_name = ['function error', 'solution error', 'complexity penalty']
        print('Fitness type is', [fitnesstype_name[i] for i in range(3) if fitnesstype[i] > 0])
        print(f'{fitnesstype[4]} subsample batches having {subbatchdata_number} time subsequence each')

    if pool_fit == 1:
        if info == True:
            print('with a single process running ...')
        if fitnesstype[0] == 1:
            if info == True:
                print('Function error calculation ...')
            for i in range(number_population):
                tem_errors = function_data_error(equations[i], i+1, training_subdata_batch, normalization,
                                                 space, times_batch, first_generation, info)
                if first_generation:
                    function_errors.append(tem_errors[0])
                    equations[i] = tem_errors[1]
                else:
                    function_errors.append(tem_errors)

        if fitnesstype[1] == 1:
            if info == True:
                print('Response error calculation ...')
            for i in range(number_population):
                tem_errors = response_data_error(equations[i], i+1,
                                                 training_subdata_batch, normalization,
                                                 datatype_weight, space, times_batch,
                                                 first_generation, info)
                if first_generation:
                    response_errors.append(tem_errors[0])
                    equations[i] = tem_errors[1]
                else:
                    response_errors.append(tem_errors)
    else:
        if pool_fit == None:
            pool_fit = os.cpu_count()
        if info == True:
            print('with',pool_fit,'processors running ...')
        with Pool(pool_fit) as p:
            if fitnesstype[0] == 1:
                if info == True:
                    print('Function calculation ...')
                pool_results = p.starmap_async(function_data_error,
                                         zip(equations, np.arange(1, number_population+1),
                                             [training_subdata_batch]*number_population,
                                             [normalization]*number_population,
                                             [space]*number_population,
                                             [times_batch]*number_population,
                                             [first_generation]*number_population,
                                             [info]*number_population), chunksize=1)
                pool_results_get = pool_results.get()
                for i in range(number_population):
                    if first_generation:
                        function_errors.append(pool_results_get[i][0])
                        equations[i] = pool_results_get[i][1]
                    else:
                        function_errors.append(pool_results_get[i])

            if fitnesstype[1] == 1:
                if info == True:
                    print('Response calculation ...')
                pool_results = p.starmap_async(response_data_error,
                                         zip(equations, np.arange(1, number_population+1),
                                             [training_subdata_batch]*number_population,
                                             [normalization]*number_population,
                                             [datatype_weight]*number_population,
                                             [space]*number_population,
                                             [times_batch]*number_population,
                                             [first_generation]*number_population,
                                             [info]*number_population), chunksize=1)
                pool_results_get = pool_results.get()
                for i in range(number_population):
                    if first_generation:
                        response_errors.append(pool_results_get[i][0])
                        equations[i] = pool_results_get[i][1]
                    else:
                        response_errors.append(pool_results_get[i])
            p.close()
            p.join()
            del p

    if info == True:
        print('Fitness value calculation ...')
    for i in range(number_population):
        multiobjective = 0
        if fitnesstype[0] == True:
            multiobjective += function_errors[i]
        if fitnesstype[1] == True:
            multiobjective += response_errors[i]
        if fitnesstype[2] > 0:
            multiobjective += fitnesstype[2]*GPfSI_et.counting_components(equations[i])
        equations_error.append([round(multiobjective, 6), equations[i]])

    equations_error = sorted(equations_error, key=lambda temp: temp[0])
    fitness_info = [i[0] for i in equations_error]
    equations = [i[1] for i in equations_error]
    if info == True:
        print('Fitness test is done')
    return equations, fitness_info

def response_data_generation(equation, equation_number, initials, space, times, info):
    try:
        rhs = equation.eqn_print()
        success = True
        if len(space) == 1:
            rhs_l = lambdify([X, Xt], rhs,
                             modules=[{'Heaviside': lambda x: np.heaviside(x,0.5)},
                                      'numpy','scipy'])
            responses, success = rk4_integration_1d_ode(rhs_l, initials[:2], times[1], times[0], info)
        else:
            lhs = lambdify(sympify(equation.tree_info[0][2]), rhs,
                           modules=[{'Heaviside': lambda x: np.heaviside(x,0.5)},
                                    'numpy','scipy'])
            responses, success = rk4_integration_1d(lhs, initials[0], space, times, info, 'spectral')
    except:
        success = False

    if success == False:
        responses = float('inf')
        if info == True:
            print('Equation number ', equation_number,' is failed')
    else:
        if info == True:
            print('Equation number ', equation_number,' is done')

    return responses

def response_data_generation_f(equation, equation_number, initials, space, times, info):
    try:
        rhs = equation.eqn_print()
        success = True
        lhs = lambdify(sympify(equation.tree_info[0][2]), rhs,
                       modules=[{'Heaviside': lambda x: np.heaviside(x,0.5)},
                                'numpy','scipy'])
        responses, success = rk4_integration_1d(lhs, initials[0], space, times, info, 'finite-difference')
    except:
        success = False

    if success == False:
        responses = float('inf')
        if info == True:
            print('Equation number ', equation_number,' is failed')
    else:
        if info == True:
            print('Equation number ', equation_number,' is done')

    return responses

def calculate_error(test_data, reference_data, datatype_weight, normalization):
    if len(datatype_weight) > 1:
        rmserror = np.zeros(len(datatype_weight))
        for k in range(len(datatype_weight)):
            rmserror[k] = np.sqrt(np.sum((test_data[k]-reference_data[k])**2)/len(reference_data[k].flatten()))/normalization[k]
        rmserror_average = np.dot(rmserror, datatype_weight)/sum(datatype_weight)
    else:
        rmserror_average = np.sqrt(np.sum((test_data-reference_data)**2)/len(reference_data.flatten()))/normalization
    if np.isnan(rmserror_average) or np.isinf(rmserror_average):
        rmserror_average = float('inf')
    return rmserror_average


def response_data_error(equation, equation_number,
                        training_subdata_batch, normalization,
                        datatype_weight, space, times_batch, first_generation, info):

    errors = []
    for i in range(len(training_subdata_batch)):
        if i == 1:
            info = False
        initials = training_subdata_batch[i][:,0]
        times = times_batch[i]
        response_data = response_data_generation(equation, equation_number,
                                                 initials, space, times, info)
        reference_data = training_subdata_batch[i]

        if first_generation:
            tree_info = equation.tree_info
            rollingseed = np.random.random(equation_number)

            while np.isinf(response_data).any():
                equation = GPfSI_et.random_equation_generator(1, tree_info, 'grow', False)[0]
                sympified_equations = sympify(equation.eqn_print())
                if info == True:
                    print('Equation number', equation_number,'is',sympified_equations)
                response_data = response_data_generation(equation, equation_number,
                                                         initials, space, times, info)

        if np.isinf(response_data).any():
            error = float('inf')
        else:
            error = calculate_error(response_data, reference_data, datatype_weight, normalization)
        errors.append(error)
    avgerror = np.mean(errors)

    if first_generation:
        return avgerror, equation
    else:
        return avgerror

def function_data_error(equation, equation_number,
                        training_subdata_batch, normalization, space, times_batch, first_generation, info):
    errors = []
    for i in range(len(training_subdata_batch)):
        if i == 1:
            info = False
        input_training = training_subdata_batch[i][:-1]
        output_training = training_subdata_batch[i][-1]
        output_model = function_data_generation(equation, equation_number, input_training, space, times_batch[i][1], info)
        if first_generation:
            tree_info = equation.tree_info
            rollingseed = np.random.random(equation_number)

            while np.isinf(output_model).any():
                equation = GPfSI_et.random_equation_generator(1, tree_info, 'grow', False)[0]
                sympified_equations = sympify(equation.eqn_print())
                if info == True:
                    print('Equation number', equation_number,'is',sympified_equations)
                output_model = function_data_generation(equation, equation_number, input_training, space, times_batch[i][1], info)

        if np.isinf(output_model).any():
            error = float('inf')
        else:
            error = calculate_error(output_model, output_training, [1], normalization[-1])
        errors.append(error)
    avgerror = np.mean(errors)

    if first_generation:
        return avgerror, equation
    else:
        return avgerror

def function_data_generation(equation, equation_number, input_training, space, excitation, info):
    try:
        rhs = equation.eqn_print()
        success = True
        lhs = lambdify(sympify(equation.tree_info[0][2]), rhs,
                       modules=[{'Heaviside': lambda x: np.heaviside(x,0.5)},
                                'numpy','scipy'])

        if len(space) == 1:
            output_model = (excitation-lhs(input_training[0],input_training[1]))/0.001
        else:
            if len(equation.tree_info[0][2]) == 2:
                output_model = lhs(input_training[0],input_training[1])
            elif len(equation.tree_info[0][2]) == 3:
                output_model = lhs(input_training[0],input_training[1],input_training[2])
            elif len(equation.tree_info[0][2]) == 4:
                output_model = lhs(input_training[0],input_training[1],input_training[2],
                                   input_training[3])
            elif len(equation.tree_info[0][2]) == 5:
                output_model = lhs(input_training[0],input_training[1],input_training[2],
                                   input_training[3],input_training[4])

        if np.isnan(output_model).any() or np.isnan(output_model).any() or (np.abs(output_model) >= 1e10).any():
            success = False
    except:
        success = False

    if success == False:
        output_model = float('inf')
        if info == True:
            print('Equation number ', equation_number,' is failed')
    else:
        if info == True:
            print('Equation number ', equation_number,' is done')

    return output_model
