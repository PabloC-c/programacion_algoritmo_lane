from funciones import *

#Direcciones: pares de la forma (direccion archivo .prob, direccion archivo .blocks)
#Direcctiorio donde se corre /home/pcarrascoc/practica/Codigo/
direcciones_pablo = [('../../Instancias/mines/kd_4phases_f0.prob','../../Instancias/incrementsBlocks/kd_inc.blocks'),('../../Instancias/mines/chaiten_4phases_f0.prob','../../Instancias/incrementsBlocks/chaiten_inc.blocks'),('../../Instancias/mines/marvinml_4phases_f0.prob','../../Instancias/incrementsBlocks/marvinml_inc.blocks'),('../../Instancias/mines/palomo25_4phases_f0.prob','../../Instancias/incrementsBlocks/palomo25_inc.blocks')]
direcciones = [direcciones_pablo[0]]
#data = {0:["NDESTINATIONS", "NPERIODS", "DISCOUNT_RATE", "NCONSTRAINTS", "CONSTRAINT","CONSTRAINT","OBJECTIVE","OBJETIVO","INCREMENTS"],1:["2","20","0.1","2","0 4 P * L 2","1 4 P 1 L 1","0 5","1 6","2"]}
#bloques= {0:[0,1,2,3,4],1:[0,0,1,1,2],2:[1,1,1,1,2],3:[1,1,1,1,2],4:[1,1,1,1,2],5:[1,1,-2,1,2],6:[1,1,-2,1,2],7:[1,1,1,1,2],8:[1,1,1,1,2],9:[0,0,1,1,2]}
#direcciones =  [direcciones_pablo[1]] #[('../Instancias/mines/marvinml_4phases_f0.prob','../Instancias/incrementsBlocks/marvinml_inc.blocks')]
#direcciones = [(data,bloques)]

for valor in direcciones:
  model,info,instancia = reader(valor[0],valor[1])
  #yf,xf,times,q_array,v_array = original_solver(model,instancia,option = 'pwl',flag_full = False)
  directory = valor[1]
  directory = directory[33:-6]
  #directory_times = '../../sols'+directory[:-1]+'_times.txt'
  #directory_v = '../../sols'+directory[:-1]+'_v_k.txt'
  directory = '../../sols' + directory + 'txt'
  #writer_y(directory,yf)
  y = read_y(directory)
  feasible,output,pincrements = check_factibility(instancia,model,y)
  print('feasible',feasible)
  #print(output)
  #print(pincrements)
  tablas=Tablas(model,instancia,y)
  print(tablas[0])
  print(tablas[1])
  print(tablas[2])

  #writer_v_k(directory_v,v_array)
  #writer_times(directory_times,times)
  print('Siguiente problema')
  
