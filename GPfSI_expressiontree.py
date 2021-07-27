# '''
# @file GPfSI_expressiontree.py
# @brief Subcode to construct and modify models
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
# This is the subcode of GPfSI to construct and modify models (expression trees).
# '''

import numpy as np
import sympy
from sympy import *
from multiprocessing import Pool
import time
import os

# Enviroment initialization
variable_symbols = 'X, Xx, Xxx, Xt'
X, Xx, Xxx, Xt = sympy.symbols(variable_symbols)
ignore_errors = np.seterr(all='ignore')

def random_equation_generator(number_population, tree_info, method, info):
    def probability_tree(tree_info):
        probability_tree_all = []
        probability_tree_wn = []
        probability_tree_wt = []
        probability_number_tree = [0,1,2,3]
        for i in range(len(probability_number_tree)):
            tree_weight_each = tree_weight[i]
            while tree_weight_each > 0:
                probability_tree_all.append(probability_number_tree[i])
                if probability_number_tree[i] == 0 or probability_number_tree[i] == 1:
                    probability_tree_wn.append(probability_number_tree[i])
                if probability_number_tree[i] == 2 or probability_number_tree[i] == 3:
                    probability_tree_wt.append(probability_number_tree[i])
                tree_weight_each = tree_weight_each - 1
        probability_tree = [ probability_tree_all, probability_tree_wn, probability_tree_wt]
        return probability_tree

    def random_equation(tree_components, probability_tree, tree_level, method):
        onetree = expressiontree(0, tree_info)
        rooting(onetree, probability_tree, method)
        growing(onetree, probability_tree, tree_level, method)
        return onetree

    def process20(iteration, process, number):
        if (iteration/(number-1))*100 >= process:
            print(process, '% generation')
            process += 20
        return process

    if info == True:
        print('Initialization method is', method)

    equations = []
    tree_components = tree_info[0]
    tree_weight = tree_info[1]
    tree_level = tree_info[2]
    probability_tree = probability_tree(tree_weight)

    if number_population == 1:
        success = False
        while success == False:
            equation_one, success = simplify_equation(random_equation(tree_components, probability_tree, tree_level, method))
        equations.append(equation_one)
    else:
        if method == 'ramped half-and-half':
            half = number_population // 2
            rest = number_population - half
            process = 0
            for i in range(half):
                if info == True:
                    process = process20(i, process, half)
                success = False
                while success == False:
                    equation_one, success = simplify_equation(random_equation(tree_components,
                                                                              probability_tree, tree_level, 'grow'))
                equations.append(equation_one)
            process = 0
            for i in range(rest):
                if info == True:
                    process = process20(i, process, rest)
                success = False
                while success == False:
                    equation_one, success = simplify_equation(random_equation(tree_components,
                                                                              probability_tree, tree_level, 'full'))
                equations.append(equation_one)
        else:
            process = 0
            for i in range(number_population):
                if info == True:
                    process = process20(i, process, number_population)
                success = False
                while success == False:
                    equation_one, success = simplify_equation(random_equation(tree_components,
                                                                              probability_tree, tree_level, method))
                equations.append(equation_one)

    if info == True:
        if len(equations) >= 10:
            print('10 sample equations')
            for i in range(10):
                print(str(i+1),': ',equations[i].eqn_print())
    return equations

class expressiontree:
    def __init__(self, value, tree_info):
        self.node = value
        self.left_child = None
        self.right_child = None
        self.tree_info = tree_info
    def insert_left_child(self, value):
        self.left_child = expressiontree(value, self.tree_info)
    def insert_right_child(self, value):
        self.right_child = expressiontree(value, self.tree_info)

    # expression tree printing
    def eqn_print(self):
        eqn = ''
        if str(self.node) in self.tree_info[0][1]:
            eqn = eqn + str(self.node)
            eqn = eqn + '(' + self.left_child.eqn_print() + ')'
        else:
            eqn = eqn + '('
            if self.left_child:
                eqn = eqn + self.left_child.eqn_print()
            eqn = eqn + str(self.node)
            if self.right_child:
                eqn = eqn + self.right_child.eqn_print()
            eqn = eqn + ')'
        return eqn

# expression tree growing
def rooting(root, probability_tree, method):
    if root.node == 0:
        root.node = np.random.choice(root.tree_info[0][0])
        if method == 'full':
            root.insert_left_child(np.random.choice(probability_tree[1]))
            root.insert_right_child(np.random.choice(probability_tree[1]))
        else:
            root.insert_left_child(np.random.choice(probability_tree[0]))
            root.insert_right_child(np.random.choice(probability_tree[0]))
    else:
        root.node = np.random.choice(root.tree_info[0][1])
        if method == 'full':
            root.insert_left_child(np.random.choice(probability_tree[1]))
        else:
            root.insert_left_child(np.random.choice(probability_tree[0]))

def growing(root, probability_tree, tree_level, method):
    if root.left_child:
        if tree_level > 2:
            if root.left_child.node < 2:
                rooting(root.left_child, probability_tree, method)
                growing(root.left_child, probability_tree, tree_level - 1, method)
            elif root.left_child.node == 2:
                root.insert_left_child(np.random.choice(root.tree_info[0][2]))
            else:
                root.insert_left_child(round(np.random.random(), 6))
        else:
            if root.left_child.node < 2:
                root.left_child.node = np.random.choice(probability_tree[2])
            if root.left_child.node == 2:
                root.insert_left_child(np.random.choice(root.tree_info[0][2]))
            else:
                root.insert_left_child(round(np.random.random(), 6))

    if root.right_child:
        if tree_level > 2:
            if root.right_child.node < 2:
                rooting(root.right_child, probability_tree, method)
                growing(root.right_child, probability_tree, tree_level - 1, method)
            elif root.right_child.node == 2:
                root.insert_right_child(np.random.choice(root.tree_info[0][2]))
            else:
                root.insert_right_child(round(np.random.random(), 6))
        else:
            if root.right_child.node < 2:
                root.right_child.node = np.random.choice(probability_tree[2])
            if root.right_child.node == 2:
                root.insert_right_child(np.random.choice(root.tree_info[0][2]))
            else:
                root.insert_right_child(round(np.random.random(), 6))

def counting_components(tree):
    def counting(tree, number_components):
        if tree.left_child:
            number_components += 1
            if tree.left_child.left_child:
                number_components = counting(tree.left_child, number_components)
        if tree.right_child:
            number_components += 1
            if tree.right_child.left_child:
                number_components = counting(tree.right_child, number_components)
        return number_components
    number_components = counting(tree, 1)
    return number_components

def evolutionary_operators(equations, fitness_info, representation_rate, crossover_rate, mutation_rate,
                           nodenumber_max, info):
    number_population = len(equations)
    number_representation = int(np.ceil(number_population*representation_rate))
    number_evolution = number_population - number_representation
    evolved_equations = []
    if info == True:
        print(f'Evolve {number_population} equations \nRepresentation: {representation_rate} / Crossover: {crossover_rate} / Mutation: {mutation_rate} \nMaximum node number: {nodenumber_max}')

    unit_coefficients_repequations = []
    for i in equations:
        try:
            unit_coefficiented = check_structure(i)
            unit_coefficients_repequations.append(sympify(unit_coefficiented.eqn_print()))
        except:
            pass

    # representation
    rep_i = 0
    rep_structure = []
    while len(evolved_equations) <= number_representation:
        tem_equation = copy_tree(equations[rep_i])
        # node number check
        if counting_components(tem_equation) <= nodenumber_max:
            if len(evolved_equations) == 0:
                evolved_equations.append(tem_equation)
                rep_structure.append(unit_coefficients_repequations[rep_i])
            else:
                if repetition_check_one(tem_equation, rep_structure) == False:
                    evolved_equations.append(tem_equation)
                    rep_structure.append(unit_coefficients_repequations[rep_i])
        rep_i += 1


    # evolution & node number check & repetition check
    evolved_equations = evolution(evolved_equations, equations, number_population, fitness_info,
                                  crossover_rate, mutation_rate,
                                  nodenumber_max, unit_coefficients_repequations)

    return evolved_equations

def evolution(evolved_equations, equations, number_population, fitness_info, crossover_rate, mutation_rate,
              nodenumber_max, unit_coefficients_repequations):
    while len(evolved_equations) < number_population:
        parent1 = tounament_selection(equations, fitness_info)
        parent2 = tounament_selection(equations, fitness_info)
        child1 = copy_tree(parent1)
        child2 = copy_tree(parent2)

        # crossover
        if np.random.random() < crossover_rate:
            child1, child2 = crossover(parent1, parent2, nodenumber_max, False)

        # mutation
        if np.random.random() < mutation_rate:
            child1 = mutation(parent1, nodenumber_max, False)
            child2 = mutation(parent2, nodenumber_max, False)

        child1, success = simplify_equation(child1)
        if success == False:
            child1 = copy_tree(parent1)
        child2, success = simplify_equation(child2)
        if success == False:
            child2 = copy_tree(parent2)

        if counting_components(child1) <= nodenumber_max:
            if repetition_check_one(child1, unit_coefficients_repequations) == False:
                evolved_equations.append(child1)
        if len(evolved_equations) < number_population:
            if counting_components(child2) <= nodenumber_max:
                if repetition_check_one(child2, unit_coefficients_repequations) == False:
                    evolved_equations.append(child2)

    return evolved_equations

def crossover(equation_a, equation_b, nodenumber_max, info):
    # select eqns. and reconstruct to the tree structure
    tree_a = copy_tree(equation_a)
    tree_b = copy_tree(equation_b)
    tree_info = equation_a.tree_info
    if info == True:
        print('originals:\n'+tree_a.eqn_print()+'\n'+tree_b.eqn_print())

    # count the number of components
    info_a = node_infomation(tree_a)
    info_b = node_infomation(tree_b)

    # select the number of components
    max_counting_a = min( len(info_a), nodenumber_max - len(info_b) )
    if max_counting_a <= 0:
        number_random_comp_a = info_a[-1][0]
    else:
        number_random_comp_a = np.random.choice( [i[0] for i in info_a if i[1] <= max_counting_a] )
    max_counting_b = min( len(info_b), nodenumber_max - ( len(info_a) - info_a[number_random_comp_a][1] ) )
    if max_counting_b <= 0:
        number_random_comp_b = info_b[-1][0]
    else:
        number_random_comp_b = np.random.choice( [i[0] for i in info_b if i[1] <= max_counting_b] )
    if info == True:
        print(max_counting_a, max_counting_b,
              '\n', len(info_a), number_random_comp_a, '\n', len(info_b), number_random_comp_b)

    # cut components
    random_comp_a = select_random_component(tree_a, number_random_comp_a)
    random_comp_b = select_random_component(tree_b, number_random_comp_b)
    if info == True:
        print('\nrandom_nodes:\n'+random_comp_a.eqn_print()+'\n'+random_comp_b.eqn_print())

    # apply reproducing operator
    change_component(tree_a, random_comp_b, number_random_comp_a)
    change_component(tree_b, random_comp_a, number_random_comp_b)
    if info == True:
        print('\nfinal_products:\n'+tree_a.eqn_print()+'\n'+tree_b.eqn_print())
    return tree_a, tree_b

def mutation(equation, nodenumber_max, info):
    tree_a = copy_tree(equation)
    tree_info = equation.tree_info
    if info == True:
        print('originals:\n' + tree_a.eqn_print())
    # mutation one
    random_equation = random_equation_generator(1, tree_info, 'grow', False)[0]
    if info == True:
        print('elements:\n' + random_equation.eqn_print())

    info_a = node_infomation(tree_a)
    max_counting_a = nodenumber_max - len(info_a)
    if  max_counting_a <= 0:
        number_random_comp_a = info_a[-1][0]
    else:
        number_random_comp_a = np.random.choice( [i[0] for i in info_a if i[1] <= max_counting_a] )
    if info == True:
        print(max_counting_a, '\n', len(info_a), number_random_comp_a)
    change_component(tree_a, random_equation, number_random_comp_a)
    if info == True:
        print('final_products:\n' + tree_a.eqn_print())
    return tree_a

def tounament_selection(equations, fitness_info):
    tounament_eqns = []
    for i in range(2):
        random_n = np.random.choice( np.arange(len(equations)) )
        tounament_eqns.append([fitness_info[random_n], equations[random_n]])
    selected_equation = sorted(tounament_eqns, key=lambda temp: temp[0])[0][1]
    return selected_equation

def copy_tree(tree_original):
    tree_info = tree_original.tree_info
    tree_copy = expressiontree(0, tree_info)
    tree_copy.node = tree_original.node
    if tree_original.left_child:
        tree_copy.left_child = copy_tree(tree_original.left_child)
    if tree_original.right_child:
        tree_copy.right_child = copy_tree(tree_original.right_child)
    return tree_copy

def node_infomation(tree):
    node_info = []
    def counting(tree, node_number, node_info):
        node_info.append( ( node_number, counting_components(tree) ) )
        node_number += 1
        if tree.left_child:
            node_number, node_info = counting(tree.left_child, node_number, node_info)
        if tree.right_child:
            node_number, node_info = counting(tree.right_child, node_number, node_info)
        return node_number, node_info
    node_number, node_info = counting(tree, 0, node_info)
    return node_info

def select_random_component(tree, number_random_component):
    def random_selection(tree, selected_component, number_random_component):
        if number_random_component == float('inf'):
            number_random_component = float('inf')
        elif number_random_component == 0:
            selected_component.node = tree.node
            selected_component.left_child = tree.left_child
            selected_component.right_child = tree.right_child
            number_random_component = float('inf')
        else:
            number_random_component -= 1
            if tree.left_child:
                number_random_component = random_selection(tree.left_child,
                                                           selected_component, number_random_component)
            if tree.right_child:
                number_random_component = random_selection(tree.right_child,
                                                           selected_component, number_random_component)
        return number_random_component
    selected_component = expressiontree(0, tree.tree_info)
    number_random_component = random_selection(tree, selected_component, number_random_component)
    return selected_component

def change_component(tree, random_component, number_random_component):
    def change(tree, random_component, number_random_component):
        if number_random_component == float('inf'):
            number_random_component = float('inf')
        elif number_random_component == 0:
            tree.node = random_component.node
            tree.left_child = random_component.left_child
            tree.right_child = random_component.right_child
            number_random_component = float('inf')
        else:
            number_random_component -= 1
            if tree.left_child:
                number_random_component = change(tree.left_child, random_component, number_random_component)
            if tree.right_child:
                number_random_component = change(tree.right_child, random_component, number_random_component)
        return number_random_component
    number_random_component = change(tree, random_component, number_random_component)

def check_structure(equation):
    info_equation = interpretation(equation)
    for i in info_equation[3]:
        i[1] = 1.000
    unit_coefficiented = change_numbers(equation, info_equation)
    return unit_coefficiented

def interpretation(tree):
    tree_info = tree.tree_info
    info = [ [], [], [], [] ] # node-seminode-variable-number
    def extract_info(tree, info, number_node):
        if tree.node in tree_info[0][0]:
            info[0].append([number_node, tree.node])
        elif tree.node in tree_info[0][1]:
            info[1].append([number_node, tree.node])
        elif tree.node in tree_info[0][2]:
            info[2].append([number_node, tree.node])
        else:
            info[3].append([number_node, tree.node])
        number_node += 1
        if tree.left_child:
            info, number_node = extract_info(tree.left_child, info, number_node)
        if tree.right_child:
            info, number_node = extract_info(tree.right_child, info, number_node)
        return info, number_node
    info, number_node = extract_info(tree, info, 0)
    return info

def change_numbers(tree, info):
    new_tree = copy_tree(tree)
    def substitution(new_tree, info, number_node, number_index):
        if number_index < len(info[3]):
            if number_node == info[3][number_index][0]:
                new_tree.node = info[3][number_index][1]
                number_index += 1
        number_node += 1
        if new_tree.left_child:
            info, number_node, number_index = substitution(new_tree.left_child, info, number_node, number_index)
        if new_tree.right_child:
            info, number_node, number_index = substitution(new_tree.right_child, info, number_node, number_index)
        return info, number_node, number_index
    substitution(new_tree, info, 0, 0)
    return new_tree

def repetition_check_one(equation, unit_coefficients_repequations):
    try:
        same_boolean = False
        unit_coefficiented = check_structure(equation)
        unit_coefficients_equation = sympify(unit_coefficiented.eqn_print())
        for i in range(len(unit_coefficients_repequations)):
            if unit_coefficients_repequations[i] == unit_coefficients_equation:
                same_boolean = True
    except:
        same_boolean = False
    return same_boolean

def inverse_expressiontree(equation, tree_info):
    def inversing(equation, tree, function_depth, function_rep, function_exp):
        if len(equation.args) == 1:
            if ((equation.func in sympify(tree_info[0][1]))
                and (function_depth < 2)
                and (function_rep)):
                function_depth += 1
                tree.node = str(equation.func)
                if equation.func == exp:
                    if function_exp == True:
                        tree.node = '*'
                        tree.insert_right_child(1.000)
                    function_exp = True
                if (len(equation.args[0].args) == 1) and (equation.func == equation.args[0].func):
                    function_rep = False
            else:
                tree.node = '*'
                tree.insert_right_child(1.000)
                function_rep = True

            tree.insert_left_child(1.000)
            left_element = equation.args[0]
            inversing(left_element, tree.left_child, function_depth, function_rep, function_exp)

        elif equation.func == Piecewise:
            tree.node = '*'
            tree.insert_right_child(1.000)
            tree.insert_left_child(1.000)
            left_element = equation.args[0].args[0]
            inversing(left_element, tree.left_child, function_depth, function_rep, function_exp)

        elif equation.func == Pow:
            tree.node = '*'
            tree.insert_left_child(1.000)
            tree.insert_right_child(1.000)
            left_element = equation.args[0]**(equation.args[1]-1)
            right_element = equation.args[0]
            inversing(left_element, tree.left_child, function_depth, function_rep, function_exp)
            inversing(right_element, tree.right_child, function_depth, function_rep, function_exp)

        elif equation.func == Add:
            tree.node = '+'
            tree.insert_left_child(1.000)
            tree.insert_right_child(1.000)
            components = equation.args
            all_index = np.arange(len(components))
            components_number = [counting_components_eq(i) for i in components]
            right_index = np.where(components_number == np.min(components_number))[0][0]
            right_element = components[right_index]
            left_index = np.delete(all_index, np.argwhere( all_index == right_index ) )
            left_element = 0
            for i in left_index:
                left_element += components[i]
            inversing(left_element, tree.left_child, function_depth, function_rep, function_exp)
            inversing(right_element, tree.right_child, function_depth, function_rep, function_exp)

        elif equation.func == Mul:
            tree.node = '*'
            tree.insert_left_child(1.000)
            tree.insert_right_child(1.000)
            components = equation.args
            all_index = np.arange(len(components))
            components_number = [counting_components_eq(i) for i in components]
            right_index = np.where(components_number == np.min(components_number))[0][0]
            right_element = components[right_index]
            left_index = np.delete(all_index, np.argwhere( all_index == right_index ) )
            left_element = 1
            for i in left_index:
                left_element *= components[i]
            inversing(left_element, tree.left_child, function_depth, function_rep, function_exp)
            inversing(right_element, tree.right_child, function_depth, function_rep, function_exp)

        else:
            if str(equation) in tree.tree_info[0][2]:
                tree.node = str(equation)
            else:
                try:
                    number = round(float(equation), 6)
                except:
                    number = 0
                if np.isinf(number) or np.isnan(number):
                    number = 0
                else:
                    while abs(number) >= 1e3:
                        number /= float(1e1)
                tree.node = number

    inversed_expressiontree = expressiontree(0, tree_info)
    function_depth = 0
    function_rep = True
    function_exp = False
    inversing(equation, inversed_expressiontree, function_depth, function_rep, function_exp)
    return inversed_expressiontree

def counting_components_eq(equation):
    def counting(equation, number_node):
        if equation.args:
            number_node += len(equation.args)
            for i in equation.args:
                number_node += counting(i, 0)
        else:
            number_node = 0
        return number_node
    number_node = counting(equation, 1)
    return number_node

def remove_repetition(equations):
    tree_info = equations[0].tree_info
    number_population = len(equations)
    population = len(equations)
    unit_coefficients_equations = []
    new_equations = []

    for i in range(number_population):
        try:
            unit_coefficiented = check_structure(equations[i])
            unit_coefficients_equations.append(sympify(unit_coefficiented.eqn_print()))
            new_equations.append(equations[i])
        except:
            pass
    equations = new_equations
    population = len(equations)

    iteration = 0
    while population - 1 > iteration:
        index_same = []
        new_unit_coefficients_equations = []
        new_equations = []
        for j in range(iteration+1, population):
            if unit_coefficients_equations[iteration] == unit_coefficients_equations[j]:
                index_same.append(j)
        for j in range(population):
            if j not in index_same:
                new_unit_coefficients_equations.append(unit_coefficients_equations[j])
                new_equations.append(equations[j])
        unit_coefficients_equations = new_unit_coefficients_equations
        equations = new_equations
        population = len(equations)
        iteration += 1

    number_repeatation = number_population - len(equations)
    equations = equations + random_equation_generator(number_repeatation, tree_info, 'grow', False)

    return equations

def simplify_equation(equation):
    success = True
    try:
        simplified_equation = expand(equation.eqn_print())
        simplify_expressiontree = inverse_expressiontree(simplified_equation, equation.tree_info)
    except:
        try:
            sympified_equation = sympify(equation.eqn_print())
            simplify_expressiontree = inverse_expressiontree(sympified_equation, equation.tree_info)
        except:
            print('sympification fails', equation.eqn_print())
            simplify_expressiontree = equation
            success = False
    return [simplify_expressiontree, success]
