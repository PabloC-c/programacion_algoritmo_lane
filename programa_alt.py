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
  #print(model._oincrements[0][0],model._oincrements[0][1],model._oincrements[1][0])
  print(model._nphases)
  print(model._nbenches)
  print("min", model._bench_min)
  print("min2", model._bench_max)
  #peso=grafo(model,instancia)
  #aristas=[]
  #g = Graph()
  #for i in range(model._nbenches* model._nphases):
  #  if i % model._nphases> 0:
  #    g.addEdge(i-1,i)
  #  if i >= model._nphases:
  #    g.addEdge(i- model._nphases,i)
  #visited = g.BFS(0,peso,model)
  #print(visited)
  #increment_max=0
  #norma_max=0
  #max_benches=0
  #max_phases=0
  #for i in range(len(visited)):
  #  if visited[i]== True:
  #    norma= (i%model._nphases) + (i//model._nphases)
  #    benches=(i//model._nphases)
  #    phases=(i%model._nphases) 
  #    if norma> norma_max:
  #      increment_max=i
  #      norma_max=norma
  #    if phases> max_phases:
  #      max_phases= phases
  #    if benches> max_benches:
  #     max_benches= benches
    #if visited[i]==False:
     # model._bincrements[i//model._nphases][i%model._nphases]= []
      #model._qincrements[i//model._nphases][i%model._nphases]= 0
      #model._oincrements[i//model._nphases][i%model._nphases]= 0
  #print("max",max_benches)
  #model._bincrements= model._bincrements[:max_benches+1]
  #for x in model._bincrements:
   # x= x[:max_phases+1]
  #model._nphases= max_phases +1
  #model._nbenches= max_benches +1
  #if model._bench_min ==0:
   # model._nbenches= max_benches+ 1
  #model._blocks=[]
  #for set in model._bincrements:
   # for set2 in set:
    #  model._blocks+= set2
  model.setParam('NumericFocus',3)
  directory = valor[1]
  directory = directory[33:-6]
  directory_times  = '../../sols_model2_free'+ directory[:-1]+'_times.txt'
  directory_v      = '../../sols_model2_free'+ directory[:-1]+'_v_k.txt'
  directory_y      = '../../sols_model2_free'+ directory + 'txt'
  directory_table1 = '../../sols_model2_free'+ directory[:-1]+'_values.txt'
  directory_table2 = '../../sols_model2_free'+ directory[:-1]+'_increments.txt'
  directory_table3 = '../../sols_model2_free'+ directory[:-1]+'_constraints.txt'
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
  yf,xf,times,q_array,v_array = original_solver(model,instancia,option = 'pwl',flag_full =False, x_binary = False, parada = 'cauchy')   
  writer_y(directory_y,yf)
  y = read_y(directory_y)
  feasible,output,pincrements,df,df3 = check_factibility(instancia,model,y)
  print('Feasible',feasible)
  write_table(df,directory_table1)
  write_table(pincrements,directory_table2)
  write_table(df3,directory_table3)
  writer_v_k(directory_v,v_array)
  writer_times(directory_times,times)
  print('Siguiente problema')
