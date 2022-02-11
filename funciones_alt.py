import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt

#########################################################################################################################################################################################################################################
#Funciones Inicializadoras

def initialize_model(info,instancia):
  model = gp.Model()
  #Extraemos los parametros
  model._ndestinations = int(info[info[0] == 'NDESTINATIONS'][1])
  model._nperiods      = int(info[info[0] == 'NPERIODS'][1])
  model._discount_rate = np.float64(1/(1+np.float64(info[info[0] == 'DISCOUNT_RATE'][1])))
  model._nconstraints  = int(info[info[0] == 'NCONSTRAINTS'][1])
  model._phases        = int(info[info[0] == 'PHASES'][1])
  model._nphases       = len(instancia.iloc[:,model._phases].unique())
  if min(instancia.iloc[:,model._phases].unique()) == -1:
    model._nphases = model._nphases-1
  try:
    model._benches   = int(info[info[0] == 'RBENCHES'][1])
    model._rbenches  = True
  except:
    model._benches   = int(info[info[0] == 'BENCHES'][1])
    model._rbenches  = False
  model._nbenches  = len(instancia.iloc[:,model._benches].unique())
  model._bench_min = min(instancia.iloc[:,model._benches].unique())
  model._bench_max = max(instancia.iloc[:,model._benches].unique())
  #Extraemos los datos de las restricciones
  aux_i           = info[info[0] == 'NCONSTRAINTS'].index[0] + 1
  aux_constraints = []

  for i in range(aux_i,aux_i + model._nconstraints):
    txt = info[1][i]
    txt = txt.split(" ")
    aux_constraints.append([int(txt[1]),txt[3],txt[4],int(txt[5])])

  model._infocons = aux_constraints
  #Extraemos los datos de la funcion objetivo
  aux_obj = []
  for i in range(aux_i + model._nconstraints,aux_i + model._nconstraints+2):
    txt = info[1][i]
    txt = txt.split(" ")
    aux_obj.append(int(txt[1]))

  model._infoobj = aux_obj

  model._bincrements = []
  model._qincrements = []
  model._oincrements = []
  model._blocks      = []
  for i in range(model._bench_min,model._bench_max+1):
    data_bench = instancia[instancia[model._benches] == i]
    b_phases   = [[] for j in range(model._nphases)]
    q_phases   = [0 for j in range(model._nphases)]
    o_phases   = [0 for j in range(model._nphases)]
    for p in range(model._nphases):
      aux_data = data_bench[data_bench[model._phases] == p]
      aux_list = [aux_data[0].iloc[j] for j in range(len(aux_data))] 
      model._blocks += aux_list
      b_phases[p] = aux_list
      q_phases[p] = aux_data[4].sum()                                            #ASUMIMOS QUE LA COLUMNA QUE INDICA EL TONELAJE DEL BLOQUE ES LA COLUMNA 5 (INDICE = 4)
      o_phases[p] = aux_data[4].sum()
    model._bincrements.append(b_phases)
    model._qincrements.append(q_phases)
    model._oincrements.append(o_phases)
  if model._rbenches:
    model._bincrements.reverse()
    model._qincrements.reverse()
    model._oincrements.reverse()
  return model

#Funcion reader: Recibe un string con la direccion del archivo .prob y un string con la direccion del archivo .blocks. Tambien puede recibir diccionarios para generar los data frame
def reader(d1,d2):
  if isinstance(d1, str):
    info      = pd.read_csv(d1, header = None)
    instancia = pd.read_csv(d2, header = None, sep = ' ')
    info[[0,1]] = info[0].str.split(':',expand=True)
    info[1]     = info[1].apply(lambda x: x[1:])
    model = initialize_model(info,instancia)
  else:
    info      = pd.DataFrame(d1)
    instancia = pd.DataFrame(d2)
    model     = initialize_model(info,instancia)
  return model,info,instancia

#########################################################################################################################################################################################################################################
# Funcion que crea el modelo

def create_model2(model,instancia,Q,t,vlist,qlist,option = 'pwl', flag_full = False, x_binary = False):
  model._flag_full = flag_full
  #Anadimos las variables y a los arreglos
  y  = model.addVars(len(instancia),model._ndestinations,obj = 0,lb=0,ub=1,vtype= GRB.CONTINUOUS,name="y")
  #Anadimos las variables mu al arreglo
  mu = model.addVars(model._nbenches, model._nphases,obj = 0, vtype = GRB.BINARY, name = 'mu')
  #Anadimos variables para los intervalos
  if x_binary:
    x = model.addVars(model._nbenches,model._nphases,obj = 0,lb=0,ub=1,vtype = GRB.BINARY,name="x")
  else:
    x = model.addVars(model._nbenches,model._nphases,obj = 0,lb=0,ub=1,vtype = GRB.CONTINUOUS,name="x")
  #Anadimos la funcion objetivo
  #Opcion 1: Piecewise Linear de Gurobi
  if option == 'pwl':
    #Se ajustan los datos de input
    qlist = np.array([(qlist[i])[0] for i in range(len(qlist))])
    vlist = np.array(vlist)
    idx = np.argsort(qlist)
    qlist = qlist[idx]
    vlist = vlist[idx]
    vlist = vlist*model._discount_rate
    #Se crea variable auxiliar
    q_var = model.addVar(lb=0,ub = Q,vtype= GRB.CONTINUOUS,name="q_var")
    #Se define la variable auxiliar como Q-q
    model.addConstr(q_var + sum(x[i,p]*model._oincrements[i][p] for i in range(model._nbenches) for p in range(model._nphases)) == Q ,'aux_cons')
    #Se crea la funcion objetivo
    model.setObjective(sum(y[i,d] * np.float64(instancia[model._infoobj[d]].iloc[i]) for i in model._blocks for d in range(model._ndestinations)),GRB.MAXIMIZE)
    #Se anade la funcion lineal por partes
    model.setPWLObj(q_var,qlist,vlist)
  #Anadimos las restricciones
  #Restricciones de capacidad
  if flag_full:
    lambda_var = model.addVars(len(model._infocons),vtype = GRB.BINARY,name = "lambda_var")
  index = 0
  for cons in model._infocons:
    aux = cons[3]
    if cons[1] == '*':
      model.addConstr(sum(y[i,d] * instancia[cons[0]].iloc[i]  for i in model._blocks for d in range(model._ndestinations)) <= aux,"capacity_"+str(index))
      if flag_full:
        model.addConstr(aux*lambda_var[index] + sum(y[i,d] * instancia[cons[0]].iloc[i]  for i in model._blocks for d in range(model._ndestinations)) >= aux ,"full_capacity_"+str(index))
    else:
      model.addConstr(sum(y[i,int(cons[1])] * instancia[cons[0]].iloc[i]  for i in model._blocks) <= aux,"capacity_"+str(index))
      if flag_full:
        model.addConstr(aux*lambda_var[index] + sum(y[i,int(cons[1])] * instancia[cons[0]].iloc[i]  for i in model._blocks) >= aux,"full_capacity_"+str(index))
    index += 1
  if flag_full:
    model.addConstr(sum(lambda_var[i] for i in range(index)) <= index-1) 
  #Restricciones de precedencia
  model.addConstrs((model._oincrements[i][p] * x[i,p] - mu[i,p]*model._qincrements[i][p] <= 0 for i in range(model._nbenches) for p in range(model._nphases) if model._qincrements[i][p]>0) , 'p1')
  model.addConstrs((model._oincrements[i][p] * x[i,p] - mu[i,p+1]*model._qincrements[i][p]>= 0 for i in range(model._nbenches) for p in range(model._nphases-1) if model._qincrements[i][p]>0 and model._qincrements[i][p+1]>0), 'p2_phases')
  model.addConstrs((model._oincrements[i][p] * x[i,p] - mu[i+1,p]*model._qincrements[i][p]>= 0 for i in range(model._nbenches-1) for p in range(model._nphases)if model._qincrements[i][p]>0 and model._qincrements[i+1][p]>0), 'p2_benches')
  #Restricciones de porcentajes; para cada incremento i, se debe extraer el mismo procentaje de toneladas de cada bloque en i
  model.addConstrs((x[i,p] == sum(y[b,d] for d in range(model._ndestinations)) for i in range(model._nbenches) for p in range(model._nphases) for b in model._bincrements[i][p]))
  model.addConstrs((x[i,p] == 0 for i in range(model._nbenches) for p in range(model._nphases) if  model._qincrements[i][p]==0 ))
  model.write('Ejemplo.lp')
#########################################################################################################################################################################################################################################
#Funciones auxiliares

def get_varx(model):
  # Funcion para obtener la variablexy del modelo
  x = []
  for i in range(model._nbenches):
    aux=[]
    for p in range(model._nphases):
      aux.append( model.getVarByName('x['+str(i)+','+str(p)+']').x)
    x.append(aux)
  return x

def update_increments(model,x):
  # Cambia las toneladas de cada incremento luego de extraerlas
  for i in range(model._nbenches):
    for p in range(model._nphases):
      model._qincrements[i][p] -= x[i][p]*model._oincrements[i][p]
  model.update()
  
def get_vary(model,instancia):
  # Funcion para obtener la variable y del modelo
  y = [[0,0] for i in range(len(instancia))]
  for b in model._blocks:
    for d in range(model._ndestinations):
      y[b][d] = model.getVarByName('y['+str(b)+','+str(d)+']').x
  return y

def get_u_obj(y,instancia,model):
  # Valor de la funcion u en el modelo
  return sum(y[i][d] * instancia[model._infoobj[d]].iloc[i] for i in model._blocks for d in range(model._ndestinations))

def get_i_obj(y,x,instancia,model,t):
  objetivo=[]
  toneladas=[]
  for i in range(model._nincrements):
    if x[i]==1:
      objetivo.append(  (model._discount_rate**t) * sum(y[b][d] * instancia[model._infoobj[d]].iloc[i] for b in model._bincrements[i] for d in range(model._ndestinations)))
      toneladas.append( [sum(model._oincrements[j] for i in range(0,j))])
  return objetivo,toneladas
  

def update_constraints(model,Q,x = None):
  aux_cons     = model.getConstrByName('aux_cons')
  aux_cons.rhs = Q
  mu = [] 
  for i in range(model._nbenches):
    aux = [var for var in model.getVars() if 'mu['+str(i) in var.VarName]
    mu.append(aux)
  for i in range(model._nbenches):
    for p in range(model._nphases):
      if model._oincrements[i][p]>0:
        p1 = model.getConstrByName('p1['+str(i)+','+str(p)+']')
        model.chgCoeff(p1, mu[i][p] , -model._qincrements[i][p])
        if p<model._nphases-1:
          if model._oincrements[i][p+1]>0:
            p2 = model.getConstrByName('p2_phases['+str(i)+','+str(p)+']')
            model.chgCoeff(p2, mu[i][p+1] , -model._qincrements[i][p])
        if i<model._nbenches-1:
          if model._oincrements[i+1][p]>0:
            p2 = model.getConstrByName('p2_benches['+str(i)+','+str(p)+']')
            model.chgCoeff(p2, mu[i+1][p] , -model._qincrements[i][p])
  if model._flag_full:
    lambda_var = [var for var in model.getVars() if "lambda_var" in var.VarName]
    index = 0
    for cons in model._infocons:
      c   = model.getConstrByName('capacity_'+str(index))
      cf  = model.getConstrByName('full_capacity_'+str(index))
      aux = min( Q ,cons[3])
      c.rhs  = aux
      cf.rhs = aux
      model.chgCoeff(cf, lambda_var[index] , aux)
      index += 1
  #var_x = [var for var in model.getVars() if "x" in var.VarName]
  #for j in range(model._nincrements):
   # var = var_x[j]
    #if x is None:
     # var.ub = 1.0
    #else:
     # bound = var.ub
      #if bound - x[j] <=0:
       # var.ub = 0
      #else:
       # var.ub = bound - x[j]
  model.update()
  
def reset_qincrements(model):
  aux_list = []
  for i in range(model._nbenches):
    aux_list.append([model._oincrements[i][p] for p in range(model._nphases)])
  model._qincrements = aux_list
  model.update()

def update_objective(model,instancia,qlist,vlist,t,option = 'pwl'):
  if option == 'pwl':
    #Se ajustan los datos de input
    qlist = np.array([(qlist[i])[0] for i in range(len(qlist))])
    vlist = np.array(vlist)
    idx = np.argsort(qlist)
    qlist = qlist[idx]
    vlist = vlist[idx]
    vlist = vlist*(model._discount_rate**t)
    if t == model._nperiods:
      vlist = np.zeros(len(vlist))
    #Se crea variable auxiliar
    q_var = model.getVarByName("q_var")
    #Se anade la funcion lineal por partes 
    model.setPWLObj(q_var,qlist,vlist)
    model.update()
    
def incrementens_p_block(model,instancia):
  increments = [-1 for i in range(len(instancia))]
  for i in range(len(model._bincrements)):
    for p in range(len(model._bincrements[i])):
      blocks = model._bincrements[i][p]
      for b in blocks:
        increments[b] = [i,p] 
  return increments

def codify_y(y0,y,t,model,instancia):
  for b in model._blocks:
    for d in range(model._ndestinations):
      if y[b][d] > 0:
        #Se codifica como bloque,destino, tiempo, valor de y, bench, phase
        y0.append([b,d,t,y[b][d],instancia[model._benches].iloc[b],instancia[model._phases].iloc[b]])

#########################################################################################################################################################################################################################################
#Funcion solver

def original_solver(model,instancia,option = 'pwl',flag_full = False, x_binary = False):
  model.setParam('OutputFlag',0)
  # Solver para el modelo usando regresion lineal/piece wise linear
  # Variables a rellenar para la regresion lineal/piece wise linear
  Q0  = sum(model._oincrements[i][p] for i in range(model._nbenches) for p in range(model._nphases))
  Q_k = Q0
  aux_q_array = [[(Q0*i)/5] for i in range(5,-1,-1)]
  aux_v_array  = [0 for i in range(5,-1,-1)]
  #Se crea el modelo
  print('Creacion modelo')  
  create_model2(model,instancia,Q0,1,aux_v_array,aux_q_array,option, flag_full, x_binary)
  print("Modelo creado")
  #Arreglos para guardar todos los valores de la funcion V y la variable Q-q
  v_array = [[0 for i in range(model._nperiods+1)]]
  q_array = [[[(Q0*i)/model._nperiods] for i in range(model._nperiods,-1,-1)]]
  #Se inicializa k
  k = 1
  #Arreglos para retornar la iteracion anterior al final del algoritmo
  x0_array = []
  y0_array = []
  #Tiempos de ejecucion
  times_k  = []
  print('Primer ciclo')
  while True:
    time0   = time.time()
    #Arreglos para las variables x,y y los valores u,q
    x_array = []
    y_array = []
    u_array = []
    q_bar_array = []
    #Se inicializa el tiempo
    t     = 1
    # Optimizacion de V
    # Deja de iterar cuando no hay nada en la mina o se acabó el tiempo total
    print("Segundo ciclo")
    while Q_k > 10**(-3) and t <= model._nperiods:
      #Optimizacion de model cuando resta Q_k en la mina en el tiempo t
      print('Comienzo optimizacion')
      model.optimize()
      print('Optimizacion Terminada')
      #Se obtienen los vectores solucion x,y,z y los valores u,q
      bar_x = get_varx(model)
      x_array.append(bar_x)
      bar_y = get_vary(model,instancia)
      bar_z = [bar_y[b][1] for b in range(len(bar_y))] #Suponemos que 1(el indice 1) es el destino refinadero
      bar_q =  sum(bar_x[i][p]*model._oincrements[i][p] for i in range(model._nbenches) for p in range(model._nphases) )
      u_bar_q_t = get_u_obj(bar_y,instancia,model)
      #Actualizacion de las toneladas por incremento
      update_increments(model,bar_x)
      u_array.append(u_bar_q_t)
      q_bar_array.append(bar_q)
      #Se quita el tonelaje extraido
      Q_k = Q_k - bar_q
      print('Periodos t = ',t,'. Toneladas extraidas =',bar_q,'. Toneladas restantes = ',Q_k)
      #Se codifica la solucion y de la forma: bloque, destino, tiempo, valor de y
      codify_y(y_array,bar_y,t-1,model,instancia)
      #Se aumenta el periodo
      t = t + 1
      #Se actualiza el modelo con respecto a las toneladas restantes 
      update_constraints(model,Q_k,bar_x)
      #Se actualiza el modelo con respecto al tiempo
      if t == model._nperiods:
        aux_v_array= np.array(aux_v_array) + np.random.normal(0,10**7,len(aux_v_array))
        update_objective(model,instancia,aux_q_array,aux_v_array,model._nperiods,option)
    #Se calcula lo que vale la mina al tener todas las toneladas a partir de los valores u
    print('Toneladas finales = ',Q_k)
    V_Q_t = sum(u_array[i]*(model._discount_rate**i) for i in range(len(u_array)))
    print('Valor de la mina = ',V_Q_t)
    #Se crean los vectores para guardar los valores V(Q-sum q_i) y Q - sum q_i
    aux_v_array = [V_Q_t - sum(u_array[j]*(model._discount_rate**(j)) for j in range(0,i)) for i in range(0,len(u_array)+1)] 
    #aux_v_array = []
    #aux_q_array=[]
    #for i in range(0, len(u_array)+ 1)  :
     # if int(i) % 3 != 0 or i==0:
       # vaux= V_Q_t
        #for j in range(0,i):
        #  vaux= vaux - u_array[j]*(model._discount_rate**(j))
      #aux_v_array.append(vaux)
    #for i in range(0, len(q_bar_array) +1):
     # if int(i) % 4 !=0 or i==0:
      #  qaux= Q0
       # for j in range(0,i):
        #  qaux= qaux - q_bar_array[j]
        #aux_q_array.append( [qaux])
    aux_q_array = [[Q0 - sum(q_bar_array[j] for j in range(0,i))] for i in range(0,len(q_bar_array)+1)]
    aux_v_array.append(0)
    aux_q_array.append([0])
    #Se guardan los vectores de V y Q de la iteracion k
    v_array.append(aux_v_array)
    q_array.append(aux_q_array)
    #Se calcula el tiempo final
    timef = time.time() - time0
    times_k.append(timef)   
    #Se aumenta k
    k += 1
    #Se revisa el criterio de parada
    vk0 = v_array[-2][0]
    vk1 = v_array[-1][0]
    print('Valor anterior = ',vk0,'.Valor nuevo = ',vk1,'Valor k =',k)
    print('Menor o igual:',vk1 <= vk0,'K > 2:',k>2)
    # Condiciones de término. Entrega el bar_x máximo antes de que la función objetivo disminuya
    if vk1 <= vk0 and k > 2:
        #Output: solucion y, solucion x, tiempos para cada k, arreglo de toneladas sacadas para cada k, arreglo de valores v para acada k
        return y0_array,x0_array,times_k,q_array,v_array
    #De no cumplir el criterio de parada se actualizan los valores anteriores a los valores actuales
    x0_array = x_array
    y0_array = y_array
    #Se resetean las toneladas iniciales
    Q_k = Q0
    #Se resetean las toneladas por incremento
    reset_qincrements(model)
    #Se resetean las restricciones del problema a las toneladas inciales
    update_constraints(model,Q_k)
    #Se cambia la funcion objetivo con respecto a la informacion obtenida
    update_objective(model,instancia,aux_q_array,aux_v_array,0,option)
 
#########################################################################################################################################################################################################################################
#Funcion que escribe la solucion y en un archivo
def writer_y(directory,y):
  f = open(directory, "w")
  string = ''
  for var in y:
    linea = ''
    for valor in var:
      linea += str(valor) + ' '
    linea = linea[:-1] + '\n'
    string += linea
  f.write(string)
  f.close()

def writer_times(directory,times):
  f = open(directory, "w")
  string = ''
  for valor in times:
    string += str(valor)+'\n'
  f.write(string)
  f.close()

def writer_v_k(directory,v_array):
  f = open(directory, "w")
  string = ''
  for valor in v_array:
    string += str(valor[0])+'\n'
  f.write(string)
  f.close()

#########################################################################################################################################################################################################################################
#Funcion que lee la solucion y
def read_y(directory):
  y = pd.read_csv(directory, header = None, sep = ' ',dtype = float)
  return y

def calculate_u(y,model,instancia):
  tf = int(max(y[2])+1)
  u  = [0 for i in range(tf)]
  t0 = min(y[2])
  for i in range(len(y)):
    b,d,t,val_y =  int(y[0].iloc[i]),int(y[1].iloc[i]),int(y[2].iloc[i]),y[3].iloc[i]
    if t0 == 0 :
      t += 1
    u[t-1] += val_y*instancia[model._infoobj[d]].iloc[b]*(model._discount_rate**(t-1))
  return u

#########################################################################################################################################################################################################################################

def check_factibility(instancia,model,y):
  tf = max(y[2])
  t0 = min(y[2])
  data= ["Valor previo","Aporte sin descuento", "Aporte con descuento", "Valor actual"]
  data3=[]
  for i in range(len(model._infocons)):
    data3.append("Restriccion de capacidad" + str(i))
  df= pd.DataFrame(columns=data,index=range(int(tf)))
  df3=pd.DataFrame(columns=data3,index=range(int(tf)))
  Vf=0
  Va=0
  tol = 10**(-3)
  
  if t0 == 0:
    tf = tf + 1
  data= ["Valor previo","Aporte sin descuento", "Aporte con descuento", "Valor actual"]
  data3=[]
  for i in range(len(model._infocons)):
    data3.append("Restriccion de capacidad" + str(i))
  df= pd.DataFrame(columns=data,index=range(int(tf)))
  df3=pd.DataFrame(columns=data3,index=range(int(tf)))
  Vf=0
  Va=0
  output = []
  aux_phases   = [0 for p in range(model._nphases)]
  p_increments = [aux_phases[:] for i in range(model._nbenches)]
  for i in range(int(tf)):
    restricciones=[]
    problem = [i]
    if t0 == 1:
      t = i+1
    else:
      t = i
    y_t = y[y[2] == t]
    index = 1
    A=0
    Ad=0
    for q in range(len(y_t)):
      b,d,val_y =  int(y_t[0].iloc[q]),int(y_t[1].iloc[q]),y_t[3].iloc[q]
      A = A + val_y* instancia[model._infoobj[d]].iloc[b]
    for cons in model._infocons:
      r = 0
      if cons[1] == '*':
        for q in range(len(y_t)):
          b,val_y = int(y_t[0].iloc[q]),y_t[3].iloc[q]
          r += val_y * instancia[cons[0]].iloc[b]
        restricciones.append( "*"+str(r) + "" + "<=" + "" + str(cons[3]))
      else:
        for q in range(len(y_t)):
          b,d,val_y =  int(y_t[0].iloc[q]),int(y_t[1].iloc[q]),y_t[3].iloc[q]
          if d == int(cons[1]):
            r += val_y * instancia[cons[0]].iloc[b]
        restricciones.append( "s"+str(r) + "" + "<=" + "" + str(cons[3]))
      if r-cons[3] > tol :
          problem.append('R. Capacidad '+str(index))
      index +=1
    df3.iloc[i]= restricciones
    Ad= A*(model._discount_rate)**i
    Vf= Va + Ad
    df.iloc[i]= [Va,A,Ad,Vf]
    Va=Vf 
    index = 0
    for j in range(model._bench_min,model._bench_max+1):
      data_bench = y_t[y_t[4] == j]
      for p in range(model._nphases):
        y_jp = data_bench[data_bench[5] == p]
        if len(y_jp) > 0:
          b0      = y_jp[0].iloc[0]
          y_0     = y_jp[y_jp[0] == b0]
          value_0 = y_0[3].sum()
          p_increments[index][p] += value_0
          b_list = y_jp[0].unique()
          for b in b_list:
            if b != b0:
              y_aux = y_jp[y_jp[0] == b]
              value = y_aux[3].sum()
              if value - value_0 > tol or value - value_0 < -tol:
                problem.append('Bloque no consistente')
      index += 1              
    if model._rbenches:
      p_increments.reverse()  
    for j in range(model._nbenches):
      for p in range(model._nphases):
        if model._oincrements[j][p] > 0:
          if p_increments[j][p] > 1 + tol:
            problem.append('Incremento excede su tonelaje')
          if p_increments[j][p] > tol:
            if j > 0:
              if model._oincrements[j-1][p] > 0 and p_increments[j-1][p] < 1-tol:
                problem.append('Precedencia entre benches')
            if p > 0:
              if model._oincrements[j][p-1] > 0 and p_increments[j][p-1] < 1-tol:
                problem.append('Precedencia entre phases')
    output.append(problem)
    if model._rbenches:
      p_increments.reverse()  
  feasible = True
  for i in range(len(output)):
    if len(output[i]) > 1:
       feasible = False
       break
  if model._rbenches:
      p_increments.reverse()
  tabla= pd.DataFrame(p_increments) 
  return feasible,output,tabla,df,df3 
  #hacer tablas:
  #Incremenntos: Tabla lo que saco por incremento por periodo
  #Valores: 3 columnas por año, van, lo que aporta el van sin decontar, con la tasa de decuento, cuanto hay acumulado

def Tablas(model,instancia,y):
  tf = max(y[2])
  t0 = min(y[2])
  if t0 == 0:
    tf = tf + 1
  data= ["Valor previo","Aporte sin descuento", "Aporte con descuento", "Valor actual"]
  data3=[]
  for i in range(len(model._infocons)):
    data3.append("Restriccion de capacidad" + str(i))
  data2= range(model._nincrements)
  df= pd.DataFrame(columns=data,index=range(int(tf)))
  df2=pd.DataFrame(columns=data2,index=range(int(tf)))
  df3=pd.DataFrame(columns=data3,index=range(int(tf)))
  Vf=0
  Va=0
  for i in range(int(tf)):
    if t0 == 1:
      t = i+1
    else:
      t = i 
    restricciones=[]
    incrementos=[]
    A=0
    Ad=0
    y_t = y[y[2] == t]
    for q in range(len(y_t)):
      b,d,val_y =  int(y_t[0].iloc[q]),int(y_t[1].iloc[q]),y_t[3].iloc[q]
      A = A + val_y* instancia[model._infoobj[d]].iloc[b]
    for cons in model._infocons:
      r = 0
      if cons[1] == '*':
          for q in range(len(y_t)):
            b,val_y = int(y_t[0].iloc[q]),y_t[3].iloc[q]
            r += val_y * instancia[cons[0]].iloc[b]
          restricciones.append( "*"+str(r) + "" + "<=" + "" + str(cons[3]))
      else:
          for q in range(len(y_t)):
            b,d,val_y =  int(y_t[0].iloc[q]),int(y_t[1].iloc[q]),y_t[3].iloc[q]
            if d == int(cons[1]):
              r += val_y * instancia[cons[0]].iloc[b]
          restricciones.append( "s"+str(r) + "" + "<=" + "" + str(cons[3]))
    for j in range(model._nincrements):
        y_j = y_t[y_t[4] == j]
        if len(y_j) > 0:
          b0     = y_j[0].iloc[0]
          y_0    = y_j[y_j[0] == b0]
          prom_0 = y_0[3].sum()
          incrementos.append(prom_0)
        else:
          incrementos.append(0)
    df3.iloc[i]= restricciones
    df2.iloc[i]= incrementos
    Ad= A*(model._discount_rate)**i
    Vf= Va + Ad
    df.iloc[i]= [Va,A,Ad,Vf]
    Va=Vf
  return df,df2,df3

def write_table(df,directory,Header=False, Index =False):
  f = open(directory, "w")
  df_string = df.to_string(header = Header, index = Index)
  f.write(df_string)
  f.close()
  
#########################################################################################################################################################################################################################################

def last_increment(y0,instancia,model):
  i_max = 0
  nincrements = len(instancia.iloc[:,-1].unique())
  for b in y0[0]:
    i_aux = instancia.iloc[int(b),-1]
    if i_aux == nincrements-1:
      i_max = nincrements-1
      break
    elif i_aux > i_max:
      i_max = i_aux
  if i_max < nincrements-1:
    aux = model._bincrements[:]
    aux = aux[:i_max+1]
    model._bincrements = aux[:]
    model._blocks = []
    model._nincrements = int(i_max+1)
    for set in model._bincrements:
      model._blocks += set
  return
#########################################################################################################################################################################################################################################
 
def cut_mine(model):
  #Obetenemos el lado izquierdo de la restriccion de capacidad maxima
  max_q = -1
  for cons in model._infocons:
    if cons[1] == '*':
      max_q = cons[-1]
  if max_q == -1:
    return False
  #Lo multiplicamos por la cantidad de periodos
  max_nq     = max_q* model._nperiods
  #Revisamos hasta que incremento sera posible extraer toneladas
  index      = 0
  index2=0
  #while index<model._nbenches:
   # if max_nq >= model._qincrements[index][0]:
    #    max_nq -= model._qincrements[index][0]
     #   index  += 1
    #else:
      #break
  for i in range(model._nbenches):
    index2=0
    while index2<model._nphases:
      if max_nq>= model._qincrements[i][index2]:
          max_nq -= model._qincrements[i][index2]
          index2 +=1
      else:
        break
    if index2== model._nphases:
      index +=1
  print(model._nbenches)
  #Si no es posible extraer toda la mina, se acota
  if index < model._nbenches-1:
    aux = model._bincrements[:]
    aux = aux[:index]
    #for x in aux:
     # x= x[:index2+1]
    #model._bincrements = aux[:]
    model._blocks = []
    model._nbenches = index
    model._bincrements = []
    for set in aux:
      model._bincrements.append(set)
      for set2 in set:
        model._blocks += set2
    print(model._nbenches)
    return True
  else:
    return False

def cut_mine_period(model):
  #Obetenemos el lado izquierdo de la restriccion de capacidad maxima
  max_q = -1
  for cons in model._infocons:
    if cons[1] == '*':
      max_q = cons[-1]
  if max_q == -1:
    return False
  #Lo multiplicamos por la cantidad de periodos
  #Revisamos hasta que incremento sera posible extraer toneladas
  index = 0
  index2=0
  #while index<model._nbenches:
   # if max_nq >= model._qincrements[index][0]:
    #    max_nq -= model._qincrements[index][0]
     #   index  += 1
    #else:
      #break
  for i in range(model._nbenches):
    index2=0
    while index2<model._nphases:
      if max_q>= model._qincrements[i][index2]:
          max_q -= model._qincrements[i][index2]
          index2 +=1
      else:
        break
    if index2== model._nphases:
      index +=1
  print(model._nbenches)
  #Si no es posible extraer toda la mina, se acota
  if index < model._nbenches-1:
    aux = model._bincrements[:]
    aux = aux[0:index+1]
    #for x in aux:
     # x= x[:index2+1]
    model._bincrements = aux[:]
    model._blocks = []
    model._nbenches = index
    for set in model._bincrements:
      for set2 in set:
        model._blocks += set2
    return True
  if index> model._nbenches-1:
    print("entre")
    j=0
    for x in model._bincrements2:
      if x not in model._bincrements and j< (index- model._nbenches +1):
        print(x)
        model._bincrements.append(x)
        for set in x:
          model._blocks += set
        j=j+1
    model._nbenches= index +1
    return True
  else:
    return False

#########################################################################################################################################################################################################################################

def sol_to_OMP(directory):
  y = read_y(directory)
  y = pd.read_csv(directory, header = None, sep = ' ',dtype = float)
  y = y[[0,1,2,3]]
  y = y.sort_values([0,1,2,3], ascending=True)
  y[3] = y[3].round(6)
  flag = False
  t0   = int(min(y[2]))
  if t0 == 1:
    flag = True
  string = ''
  for i in range(len(y)):
    for j in range(3):
      if j < 2 or (j == 2 and not Flag):
        string += str(int(y[j].iloc[i])) + ' '
      else:
        string += str(int(y[j].iloc[i]-1)) + ' '
    string += str(y[3].iloc[i])+'\n'
  new_directory = directory[:-3]+'sol'
  f = open(new_directory, "w")
  f.write(string)
  f.close()



def grafo(model,instancia):
  peso=[[0,0] for b in model._bincrements]
  i=0
  while i< model._nbenches * model._nphases:
    if i%model._nphases>0:
      peso[i][0]= model._qincrements[ (i%model._nphases)-1 , i// model._nphases]
    if i%model._nphases==0:
      peso[i][0]= 10**12
    if i//model._nphases >0:
      peso[i][1]= model._qincrements[ (i%model._nphases) , (i// model._nphases)-1]
    if i//model._nphases == 0:
      peso[i][1]= 10**12
    i=i+1
  return peso
    
# Python3 Program to print BFS traversal
# from a given source vertex. BFS(int s)
# traverses vertices reachable from s.
from collections import defaultdict

# This class represents a directed graph
# using adjacency list representation
class Graph:
	# Constructor
  def __init__(self):
		# default dictionary to store graph
    self.graph = defaultdict(list)
	# function to add an edge to graph
  def addEdge(self,u,v):
    self.graph[u].append(v)
	# Function to print a BFS of graph
  def BFS(self, s,peso):
		# Mark all the vertices as not visited
    visited = [False] * (max(self.graph) + 1)
    max_q   = (-1)
    for cons in model._infocons:
      if cons[1] == '*':
        max_q = cons[-1]
      if max_q == -1:
        return False

		# Create a queue for BFS
    queue = []

		# Mark the source node as
		# visited and enqueue it
    queue.append(s)
    visited[s] = True

    while queue:
			# Dequeue a vertex from
			# queue and print it
      s = queue.pop(0)
			# Get all adjacent vertices of the
			# dequeued vertex s. If a adjacent
			# has not been visited, then mark it
			# visited and enqueue it
      for i in self.graph[s]:
        if i//model._nbecnhes==s//model._nbenches:
          if visited[i] == False and max_q - peso[i]>=0:
            queue.append(i)
            visited[i[0]][i[1]] = True
            max_q=max_q - peso[i]
        if i[1]==s[1]:
          if visited[i] == False and max_q - peso[s[0]][s[1]][0]>=0:
            queue.append(i)
            visited[i[0]][i[1]] = True
            max_q=max_q - peso[s[0]][s[1]][0]
  
      return visited     

# Driver code

# Create a graph given in
# the above diagram
  