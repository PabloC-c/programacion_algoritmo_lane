from funciones import *
import sys

prob = sys.argv[1]
blocks = sys.argv[2]
flagfull = int(sys.argv[3])
stop = int(sys.argv[4])
soluciones = sys.argv[5]

#Direcciones: pares de la forma (direccion archivo .prob, direccion archivo .blocks)
#Direcctiorio donde se corre /home/pcarrascoc/practica/Codigo/
#direcciones_pablo = [('../../Instancias/mines/kd_4phases_f0.prob','../../Instancias/incrementsBlocks/kd_inc.blocks'),('../../Instancias/mines/chaiten_4phases_f0.prob','../../Instancias/incrementsBlocks/chaiten_inc.blocks'),('../../Instancias/mines/marvinml_4phases_f0.prob','../../Instancias/incrementsBlocks/marvinml_inc.blocks'),('../../Instancias/mines/palomo25_4phases_f0.prob','../../Instancias/incrementsBlocks/palomo25_inc.blocks')]

direcciones = [(prob,blocks)]

#data = {0:["NDESTINATIONS", "NPERIODS", "DISCOUNT_RATE", "NCONSTRAINTS", "CONSTRAINT","CONSTRAINT","OBJECTIVE","OBJETIVO","INCREMENTS"],1:["2","20","0.1","2","0 4 P * L 2","1 4 P 1 L 1","0 5","1 6","2"]}
#bloques= {0:[0,1,2,3,4],1:[0,0,1,1,2],2:[1,1,1,1,2],3:[1,1,1,1,2],4:[1,1,1,1,2],5:[1,1,-2,1,2],6:[1,1,-2,1,2],7:[1,1,1,1,2],8:[1,1,1,1,2],9:[0,0,1,1,2]}
#direcciones =  [direcciones_pablo[1]] #[('../Instancias/mines/marvinml_4phases_f0.prob','../Instancias/incrementsBlocks/marvinml_inc.blocks')]
#direcciones = [(data,bloques)]

for valor in direcciones:
  model,info,instancia = reader(valor[0],valor[1])
  cut_mine(model)
  model.setParam('NumericFocus',3)
  flag = bool(flagfull)
  stop = bool(stop)
  if stop:
      stop = "cauchy"
  else:
      stop = "concava"
  #if flag:
  #  carpeta = "sols2_restr"
  #else:
  #  carpeta = "sols2_free"
  directory = valor[1]
  directory = directory[33:-6]
  directory_times  = soluciones+ directory[:-1]+'_times.txt'
  directory_v      = soluciones+ directory[:-1]+'_v_k.txt'
  directory_y      = soluciones+ directory + 'txt'
  directory_table1 = soluciones+ directory[:-1]+'_values.txt'
  directory_table2 = soluciones+ directory[:-1]+'_increments.txt'
  directory_table3 = soluciones+ directory[:-1]+'_constraints.txt'
  directory_prev = valor[0]
  directory_prev = directory_prev[23:-5]
  array = directory_prev.split('_')
  if array[0] == 'palomo25':
    array[0] = 'palomo'
  final_ip = '../../Instancias/sols/'+array[0]+'_'+directory_prev+'_default.TOPOSORT.ip.sol'
  y_integer = read_y(final_ip)
  aux_q_array,aux_v_array = create_arrays_y(y_integer,model,instancia)
  previous = None#[aux_q_array,aux_v_array]
  #last_increment(y_integer,instancia,model)
  print(soluciones,stop)
  y0_array,x0_array,times_k,q_array,v_array = original_solver(model,instancia,option = 'pwl',flag_full = flag,x_binary = False, new_model = False, previous = previous , parada = stop)
  writer_y(directory_y,y0_array)
  y = read_y(directory_y)
  feasible,output,p_increments,binary_x = check_feasibility(instancia,model,y)
  print('Feasible',feasible)
  df1,df2,df3 = Tablas(model,instancia,y)
  write_table(df1,directory_table1)
  write_table(df2.T,directory_table2)
  write_table(df3,directory_table3)
  writer_v_k(directory_v,v_array)
  writer_times(directory_times,times_k)
  
