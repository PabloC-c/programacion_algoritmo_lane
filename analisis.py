from funciones import *
#Direcciones: pares de la forma (direccion archivo .prob, direccion archivo .blocks)
#Direcctiorio donde se corre /home/pcarrascoc/practica/Codigo/
direcciones_pablo = [('../../Instancias/mines/chaiten_4phases_f0.prob','../../Instancias/incrementsBlocks/chaiten_inc.blocks'),('../../Instancias/mines/kd_4phases_f0.prob','../../Instancias/incrementsBlocks/kd_inc.blocks'),('../../Instancias/mines/marvinml_4phases_f0.prob','../../Instancias/incrementsBlocks/marvinml_inc.blocks'),('../../Instancias/mines/palomo25_4phases_f0.prob','../../Instancias/incrementsBlocks/palomo25_inc.blocks')]
#direcciones = direcciones_pablo
#data = {0:["NDESTINATIONS", "NPERIODS", "DISCOUNT_RATE", "NCONSTRAINTS", "CONSTRAINT","CONSTRAINT","OBJECTIVE","OBJETIVO","INCREMENTS"],1:["2","20","0.1","2","0 4 P * L 2","1 4 P 1 L 1","0 5","1 6","2"]}
#bloques= {0:[0,1,2,3,4],1:[0,0,1,1,2],2:[1,1,1,1,2],3:[1,1,1,1,2],4:[1,1,1,1,2],5:[1,1,-2,1,2],6:[1,1,-2,1,2],7:[1,1,1,1,2],8:[1,1,1,1,2],9:[0,0,1,1,2]}
direcciones =  [direcciones_pablo[3]]
#direcciones = [(data,bloques)]
#resultados           = []
#resultados_flag_full = []

prints = True
soluciones_prev = False
filtrado = False

if prints:
  for valor in direcciones:
    model,info,instancia = reader(valor[0],valor[1])
    directory = valor[1]
    directory = directory[33:-6]
    directory_times = '../../sols2'+directory[:-1]+'_times.txt'
    directory_v = '../../sols2'+directory[:-1]+'_v_k.txt'
    #directory = '../../sols2' + directory + 'txt' 
    #y = read_y(directory)
    v = read_y(directory_v)
    v = v[1:]*(1.0/1000000)
    times = read_y(directory_times)
    times = times*(1.0/60)
    #u = calculate_u(y,model,instancia)
    #V_0 = sum(u[i] for i in range(len(u)))
    print('Tiempo total: ',times[0].sum())
    #Grafico de valores V_0
    fig1 = plt.figure()
    ax1  = fig1.add_subplot()
    x = np.arange(1,len(times)+1,dtype=np.int16)
    if len(times) > 10:
      ax1.scatter(x,v, color = 'c', marker = '.',linestyle='solid')
    else:
      ax1.scatter(x,v, color = 'c', marker = 'o')
    ax1.set_title('Valor inicial estimado de la Mina ' + 'Palomo')
    ax1.set_xlabel('iteracion k')
    ax1.set_ylabel('V_0(k)')
    fig1.savefig(directory_v[:-3]+'.png')

elif soluciones_prev:
  for valor in direcciones:
    model,info,instancia = reader(valor[0],valor[1])
    directory1 = 'palomo_palomo25_4phases_f0_default.TOPOSORT.ip.sol'
    #directory2 = 'palomo_palomo25_4phases_f0_default.BZ.lp.sol'
    directory1 = '../../Instancias/sols/'+directory1 
    #directory2 = '../../Instancias/sols/'+directory2
    y_integer = read_y(directory1)
    arreglo   = []
    for b in y_integer[0]:
      i_aux = int(instancia.iloc[int(b),-1])
      if not i_aux in arreglo:
        arreglo.append(i_aux)
    arreglo_real = np.arange(0,model._nincrements,dtype = int)
    arreglo = np.array(arreglo)
    idx = np.argsort(arreglo)
    arreglo = arreglo[idx]
    n = len(arreglo)
    for i in range(n):
      if arreglo[i] == arreglo_real[i]:
        print('Problema, orlando:',arreglo[i],', Real',arreglo_real[i])  
    #y_linear  = read_y(directory2)
    u_integer = calculate_u(y_integer,model,instancia)
    #u_linear  = calculate_u(y_linear,model,instancia)
    #V_0_integer = sum(u_integer[i] for i in range(len(u_integer)))
    #V_0_linear  = sum(u_linear[i] for i in range(len(u_linear)))
    #print('Valor inicial estimado de la mina (integer): ',V_0_integer)
    #print('Valor inicial estimado de la mina (linear): ',V_0_linear)

elif filtrado:
  valor = direcciones[0]
  model,info,instancia = reader(valor[0],valor[1])
  change = cut_mine(model)
  print(instancia)