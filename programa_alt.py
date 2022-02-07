from funciones_alt import *
import sys

indice = int(sys.argv[1])
#Direcciones: pares de la forma (direccion archivo .prob, direccion archivo .blocks)
#Direcctiorio donde se corre /home/pcarrascoc/practica/Codigo/
direcciones_pablo = [('../../Instancias/mines/kd_4phases_f0.prob','../../Instancias/incrementsBlocks/kd_inc.blocks'),('../../Instancias/mines/chaiten_4phases_f0.prob','../../Instancias/incrementsBlocks/chaiten_inc.blocks'),('../../Instancias/mines/marvinml_4phases_f0.prob','../../Instancias/incrementsBlocks/marvinml_inc.blocks'),('../../Instancias/mines/palomo25_4phases_f0.prob','../../Instancias/incrementsBlocks/palomo25_inc.blocks')]
direcciones = [direcciones_pablo[indice]]
#data = {0:["NDESTINATIONS", "NPERIODS", "DISCOUNT_RATE", "NCONSTRAINTS", "CONSTRAINT","CONSTRAINT","OBJECTIVE","OBJETIVO","INCREMENTS"],1:["2","20","0.1","2","0 4 P * L 2","1 4 P 1 L 1","0 5","1 6","2"]}
#bloques= {0:[0,1,2,3,4],1:[0,0,1,1,2],2:[1,1,1,1,2],3:[1,1,1,1,2],4:[1,1,1,1,2],5:[1,1,-2,1,2],6:[1,1,-2,1,2],7:[1,1,1,1,2],8:[1,1,1,1,2],9:[0,0,1,1,2]}
#direcciones =  [direcciones_pablo[1]] #[('../Instancias/mines/marvinml_4phases_f0.prob','../Instancias/incrementsBlocks/marvinml_inc.blocks')]
#direcciones = [(data,bloques)]

for valor in direcciones:
  model,info,instancia = reader(valor[0],valor[1])
  #cut_mine(model)
  model.setParam('NumericFocus',1)
  directory = valor[1]
  directory = directory[33:-6]
  directory_times  = '../../sols2'+ directory[:-1]+'_times.txt'
  directory_v      = '../../sols2'+ directory[:-1]+'_v_k.txt'
  directory_y      = '../../sols2'+ directory + 'txt'
  directory_table1 = '../../sols2'+ directory[:-1]+'_values.txt'
  directory_table2 = '../../sols2'+ directory[:-1]+'_increments.txt'
  directory_table3 = '../../sols2'+ directory[:-1]+'_constraints.txt'
  #directory_prev = valor[0]
  #directory_prev = directory_prev[23:-5]
  #array = directory_prev.split('_')
  #directory_prev = array[0]
  #directory_prev = directory_prev[:-2]
  #for string in array:
  #  directory_prev += '_'+string
  #final_ip = '../../Instancias/sols/'+directory_prev+'_default.TOPOSORT.ip.sol'
  #y_integer = read_y(final_ip)
  #last_increment(y_integer,instancia,model)
  valor2= valor2(model,instancia)
  print(valor2)
  yf,xf,times,q_array,v_array = original_solver(model,instancia,option = 'pwl',flag_full =True)
  writer_y(directory_y,yf)
  y = read_y(directory_y)
  feasible,output,pincrements = check_factibility(instancia,model,y)
  print('Feasible',feasible)
  df1,df2,df3 = Tablas(model,instancia,y)
  write_table(df1,directory_table1)
  write_table(df2.T,directory_table2)
  write_table(df3,directory_table3)
  writer_v_k(directory_v,v_array)
  writer_times(directory_times,times)
  print('Siguiente problema')
