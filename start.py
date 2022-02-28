import sys
import pandas as pd

#python start.py ordenado archivo_prob archivo_blocks archivo_sols restringido parada x_binario 

ordenado       = sys.argv[1]
archivo_prob   = sys.argv[2]
archivo_blocks = sys.argv[3]
archivo_sols   = sys.argv[4]
restringido    = sys.argv[5]
parada         = sys.argv[6]
x_binario      = sys.argv[7]

flag_full = False
x_binary  = False

if restringido == '1':
  flag_full = True

if x_binario == '1':
  x_binary = True

if ordenado == '1':
  from funciones import *
  model,info,instancia = reader(archivo_prob,archivo_blocks)
  cut_mine(model)
  model.setParam('NumericFocus',3)
  y0_array,x0_array,times_k,q_array,v_array = original_solver(model, instancia, option = 'pwl', flag_full = flag_full, x_binary = x_binary, new_model = False, previous = None , parada = parada)
  y0_array = pd.DataFrame(data = y0_array)
  sol_to_OMP(y0_array,archivo_sols)
else:
  from funciones_alt import *  
  model,info,instancia = reader(archivo_prob,archivo_blocks)
  model.setParam('NumericFocus',3)
  y0_array,x0_array,times_k,q_array,v_array = original_solver(model, instancia, option = 'pwl', flag_full = flag_full, x_binary = x_binary, previous = None, parada = parada)
  y0_array = pd.DataFrame(data = y0_array)
  sol_to_OMP(y0_array,archivo_sols)