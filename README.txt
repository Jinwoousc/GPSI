
@program 
Genetic Programming for System Identification (GPfSI)

Copyright (C) 2021 Jinwoo Im
Sonny Astani Department of Civil and Environmental Engineering, University of Southern California, Los Angeles, USA

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.



@file description

GPfSI_run.ipynb - the Jupyter notebook file to configure and execute GPfSI
GPfSI_run.py - the Python file to configure and execute GPfSI

GPfSI.py - the main code of GPfSI
GPfSI_data.py - the subcode of GPfSI to generate and compute data
GPfSI_expressiontree.py - the subcode of GPfSI to construct and modify models (expression trees)
GPfSI_optimization.py - the subcode of GPfSI to optimize coefficients in models



@input/output data
Input
 - The data are automatically simulated within the code package.
 - Excitation and system responses (e.g., external force and displacement in model 1; initial concentration profile over space and its temporal evolution in model 2)
 
Output
 - The data will be saved in the results folder.
 - initial_case_#_configuration.npy: configuration information for an identification case
 - case_#_generation_#.npy: candidate models and their information over generations
 - final_case_#_session_#.npy: candidate models and their information at each session end 
 - lossfunction_case_#.pdf: figure for loss function values and their composition over generations
 - result_outline_case_#.csv: top 10 models at the last generation with their loss function values



@execution
If you use Jupyter notebook (v.6.2.0) based on Python (v.3.7.4),
1. You can modify GPfSI_run.ipynb to select an example (ODE or PDE case) and change hyperparameters.
2. You can execute GPfSI by running cell by cell in Jupyter notebook.

If you do use only Python (v.3.7.4),
1. You can modify GPfSI_run.py to select an example (ODE or PDE case) and change hyperparameters.
2. You can "python GPfSI_run.py" in your terminal or prompt to execute GPfSI.