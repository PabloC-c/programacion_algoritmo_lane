from funciones_alt import *
import sys
import pandas as pd
import math 

indice = int(sys.argv[1])
#Direcciones: pares de la forma (direccion archivo .prob, direccion archivo .blocks)
#Direcctiorio donde se corre /home/pcarrascoc/practica/Codigo/
direcciones_pablo = [('../../Instancias/mines/chaiten_4phases_f0.prob','../../Instancias/incrementsBlocks/chaiten_inc.blocks'),('../../Instancias/mines/kd_4phases_f0.prob','../../Instancias/incrementsBlocks/kd_inc.blocks'),('../../Instancias/mines/marvinml_4phases_f0.prob','../../Instancias/incrementsBlocks/marvinml_inc.blocks'),('../../Instancias/mines/palomo25_4phases_f0.prob','../../Instancias/incrementsBlocks/palomo25_inc.blocks')]
#direcciones = direcciones_pablo
#data = {0:["NDESTINATIONS", "NPERIODS", "DISCOUNT_RATE", "NCONSTRAINTS", "CONSTRAINT","CONSTRAINT","OBJECTIVE","OBJETIVO","INCREMENTS"],1:["2","20","0.1","2","0 4 P * L 2","1 4 P 1 L 1","0 5","1 6","2"]}
#bloques= {0:[0,1,2,3,4],1:[0,0,1,1,2],2:[1,1,1,1,2],3:[1,1,1,1,2],4:[1,1,1,1,2],5:[1,1,-2,1,2],6:[1,1,-2,1,2],7:[1,1,1,1,2],8:[1,1,1,1,2],9:[0,0,1,1,2]}
direcciones =  [direcciones_pablo[indice]]
#direcciones = [(data,bloques)]
#resultados           = []
#resultados_flag_full = []

soluciones_Orlando = [("../../Instancias/sols/chaiten_chaiten_4phases_f0_default.BZ.lp.sol","../../Instancias/sols/chaiten_chaiten_4phases_f0_default.TOPOSORT.ip.sol"),("../../Instancias/sols/kd_kd_4phases_f0_default.BZ.lp.sol","../../Instancias/sols/kd_kd_4phases_f0_default.TOPOSORT.ip.sol"),("../../Instancias/sols/marvinml_marvinml_4phases_f0_default.BZ.lp.sol","../../Instancias/sols/marvinml_marvinml_4phases_f0_default.TOPOSORT.ip.sol"),("../../Instancias/sols/palomo25_palomo25_4phases_f0_default.BZ.lp.sol","../../Instancias/sols/palomo25_palomo25_4phases_f0_default.TOPOSORT.ip.sol")]

soluciones_OMP = [("../../solsOMP/chaiten_inc_chaiten_4phases_f0_default.BZ.lp.sol","../../solsOMP/chaiten_inc_chaiten_4phases_f0_default.TOPOSORT.ip.sol"),("../../solsOMP/kd_inc_kd_4phases_f0_default.BZ.lp.sol","../../solsOMP/kd_inc_kd_4phases_f0_default.TOPOSORT.ip.sol"),("../../solsOMP/marvinml_inc_marvinml_4phases_f0_default.BZ.lp.sol","../../solsOMP/marvinml_inc_marvinml_4phases_f0_default.TOPOSORT.ip.sol"),("../../solsOMP/palomo25_inc_palomo25_4phases_f0_default.BZ.lp.sol","../../solsOMP/palomo25_inc_palomo25_4phases_f0_default.TOPOSORT.ip.sol")]


soluciones_modelo2 = ["../../model2_sol_free/chaiten_inc.txt","../../model2_sol_free/kd_inc.txt","../../model2_sol_free/marvinml_inc.txt",""]

soluciones_OMP = soluciones_OMP[indice]
soluciones_Orlando = soluciones_Orlando[indice]
soluciones_modelo2 = soluciones_modelo2[indice]

prints = True
soluciones_prev = False
filtrado = False
random = False
diferencia = False

if prints:
  for valor in direcciones:
    nombres_minas =["palomo","kd","marvin","chaiten"]
    model,info,instancia = reader(valor[0],valor[1])
    directory = valor[1]
    directory = directory[33:-6]
    for m in nombres_minas:
      if m in directory:
        nombre_mina = m
    print(directory)
    print(nombre_mina)
    modelo = "funciones"
    flag = True
    if modelo == "funciones":
      if flag:
        carpeta = "sols2_restr"
      else:
        carpeta = "sols2_free"
      directory_times = '../../'+carpeta+directory[:-1]+'_times.txt'
      directory_v = '../../'+carpeta+directory[:-1]+'_v_k.txt'
    if modelo == "funciones_alt":
      if flag:
        carpeta = "sols_model2_restr"
      else:
        carpeta = "sols_model2_free"       
      directory_times = '../../'+carpeta+directory[:-1]+'_times.txt'
      directory_v = '../../'+carpeta+directory[:-1]+'_v_k.txt'
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
    x = np.arange(1,len(v)+1,dtype=np.int16)
    if len(times) > 10:
      ax1.plot(x,v,'-o',linestyle='solid')
    else:
      ax1.plot(x,v,'-o')
    ax1.set_title('Valor inicial estimado de la Mina ' + nombre_mina + ' [$1.000.000]')
    ax1.set_xlabel('iteracion k')
    ax1.set_ylabel('V_0(k)')
    fig1.savefig(directory_v[:-3]+'.png')
    #Grafico de tiempo
    fig1 = plt.figure()
    ax1  = fig1.add_subplot()
    x = np.arange(1,len(times)+1,dtype=np.int16)
    if len(times) > 10:
      ax1.plot(x,times, '-o',linestyle='solid')
    else:
      ax1.plot(x,times, '-o')
    ax1.set_title('Tiempo por Iteracion en la Mina ' + nombre_mina + ' [Minutos]')
    ax1.set_xlabel('iteracion k')
    ax1.set_ylabel('Tiempo')
    fig1.savefig(directory_times[:-3]+'.png')    

elif soluciones_prev:
  for valor in direcciones:
    model,info,instancia = reader(valor[0],valor[1])
    comparacion = "Orlando"
    print(comparacion)
    if comparacion == "Orlando":
      #directory1 = 'marvin_marvin_4phases_f0_default.TOPOSORT.ip.sol'
      #directory2 = 'palomo_palomo25_4phases_f0_default.BZ.lp.sol'
      #directory1 = '../../Instancias/sols/'+directory1 
      ##directory2 = '../../Instancias/sols/'+directory2
      directory1 = soluciones_Orlando[1]
      directory2 = soluciones_Orlando[0]
      mina = soluciones_Orlando[1][22:29]
      y_integer = read_y(directory1)
      #feasible,output,tabla,df,df3,x_binary= check_feasibility(instancia,model,y_integer)
      print("Mina :", mina)
      #print('Factible: ',feasible)
      #print('Violaciones: ',output)
      #print('x binario:' ,x_binary)
      y_linear  = read_y(directory2)
      u_integer = calculate_u(y_integer,model,instancia)
      u_linear  = calculate_u(y_linear,model,instancia)
      V_0_integer = sum(u_integer[i] for i in range(len(u_integer)))
      V_0_linear  = sum(u_linear[i] for i in range(len(u_linear)))
      print('Valor inicial estimado de la mina (integer): ',V_0_integer/1000000)
      print('Valor inicial estimado de la mina (linear): ',V_0_linear/1000000)
    elif comparacion == "OMP":
      directory1 = soluciones_OMP[1]
      directory2 = soluciones_OMP[0]
      mina = soluciones_OMP[1][14:21]
      y_integer = read_y(directory1)
      #feasible,output,tabla,df,df3,x_binary= check_feasibility(instancia,model,y_integer)
      print("Mina :", mina)
      #print('Factible: ',feasible)
      #print('Violaciones: ',output)
      #print('x binario:' ,x_binary)
      y_linear  = read_y(directory2)
      u_integer = calculate_u(y_integer,model,instancia)
      u_linear  = calculate_u(y_linear,model,instancia)
      V_0_integer = sum(u_integer[i] for i in range(len(u_integer)))
      V_0_linear  = sum(u_linear[i] for i in range(len(u_linear)))
      print('Valor inicial estimado de la mina (integer): ',V_0_integer/1000000)
      print('Valor inicial estimado de la mina (linear): ',V_0_linear/1000000)
      

elif filtrado:
  valor = direcciones[0]
  model,info,instancia = reader(valor[0],valor[1])
  print(instancia[model._phases].unique())
  #print(instancia)

elif random:
  valor = direcciones[0]
  model,info,instancia = reader(valor[0],valor[1])
  directory = valor[1]
  directory = directory[33:-6]
  directory = '../../sols_full' + directory + 'txt' 
  y = sol_to_OMP(directory)
elif diferencia:
  valor=direcciones[0]
  model, info, instancia = reader(valor[0],valor[1])
  solprevia = pd.read_csv(soluciones_Orlando[0], header = None,sep =" ")
  sol2 = pd.read_csv(soluciones_modelo2, header=None,sep =" ")
  M = max(instancia.iloc[:,0])
  suma = 0
  #print(solprevia)
  print(M)
  for x in range(M):
    xprev =0
    x2 = 0
    print(x)
    for i in range(len(solprevia)):
      #print(i,solprevia.iloc[i,0])
      if x == solprevia.iloc[i,0]:
        #print(solprevia.iloc[i,3])
        xprev +=solprevia.iloc[i,3]
    for i in range(len(sol2)):
      if x == sol2.iloc[i,0]:
        x2 += sol2.iloc[i,3]
    suma += (xprev-x2)**2
  print("Módulo de la diferencia :", math.sqrt(suma))