#Librerias extra

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
  model._nincrements   = len(instancia.iloc[:,-1].unique())
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
  for i in range(model._nincrements):
    aux_data = instancia[instancia.iloc[:,-1] == i]                                                 
    aux_list = [aux_data[0].iloc[i] for i in range(len(aux_data))]
    model._bincrements.append(aux_list)
    model._qincrements.append(aux_data[4].sum())                                            #ASUMIMOS QUE LA COLUMNA QUE INDICA EL TONELAJE DEL BLOQUE ES LA COLUMNA 5 (INDICE = 4)
    model._oincrements.append(aux_data[4].sum())
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

def create_model(model,instancia,Q,t,vlist,qlist,option = 'pwl', flag_full = False, p = None, q = None):
  model._flag_full = flag_full
  #Anadimos las variables y a los arreglos
  y  = model.addVars(len(instancia),model._ndestinations,lb=0,ub=1,vtype= GRB.CONTINUOUS,name="y")
  #Anadimos las variables mu al arreglo
  mu = model.addVars(model._nincrements,obj = 0, vtype = GRB.BINARY, name = 'mu')
  #Anadimos variables para los intervalos
  x  = model.addVars(model._nincrements,obj = 0,lb=0,ub=1,vtype= GRB.CONTINUOUS,name="x")
  #Anadimos la funcion objetivo
  #Opcion 1: Regresion lineal
  if option == 'lr':
    #Se crea y ajusta el regresor lineal
    regressor = LinearRegression().fit(qlist, vlist)
    #Se extraen las constantes 
    a0 = regressor.coef_[0]
    #Se crea variable auxiliar
    q_var = model.addVar(lb=0,vtype= GRB.CONTINUOUS,name="q_var")
    #Se define la variable auxiliar como Q-q
    model.addConstr(q_var + sum(x[i]*model._oincrements[i] for i in range(model._nincrements))  == Q,'aux_cons')    
    #Se crea la funcion objetivo
    model.setObjective(sum(y[i,d] * instancia[model._infoobj[d]].iloc[i] for i in range(len(instancia)) for d in range(model._ndestinations)) + model._discount_rate*a0*(q_var),GRB.MAXIMIZE)
  #Opcion 2: Piecewise Linear de Gurobi
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
    model.addConstr(q_var + sum(x[i]*model._oincrements[i] for i in range(model._nincrements))  == Q,'aux_cons') 
    #Se crea la funcion objetivo
    model.setObjective(sum(y[i,d] * instancia[model._infoobj[d]].iloc[i] for i in range(len(instancia)) for d in range(model._ndestinations)),GRB.MAXIMIZE)
    #Se anade la funcion lineal por partes
    model.setPWLObj(q_var,qlist,vlist)
  #Opcion 3: Considerar variables para el tiempo futuro
  if option == 'hat':
    #Anadimos la variables para el futuro x y mu gorro
    x_hat  = model.addVars(model._nincrements,model._nperiods-t,lb=0,ub=1,vtype=GRB.CONTINUOUS,name= "x_hat")
    mu_hat = model.addVars(model._nincrements,model._nperiods-t,vtype=GRB.BINARY,name= "mu_hat")
    #Se crea la funcion objetivo
    #model.setObjetive(sum(y[i][d] * instancia[model._infoobj[d]].iloc[i] for i in range(len(instancia)) for d in range(model._ndestinations)) + sum( ((model._discount_rate)**t)* x[i]*p[i] for t in range(model._nperiods) for i in model._nincrements ,GRB.MAXIMIZE))
    #Restriccion de extraccion maxima; no se puede extraer mas del 100% de un incremento
    model.addConstrs(x[i] + sum(x_hat[i,j] for j in range(model._nperiods-t)) <= 1 for i in range(model._nincrements))
    #Restriccion de precedencia
    model.addConstrs(x_hat[i,j] <= mu_hat[i,j] for i in range(model._nincrements) for j in range(model._nperiods-t))
    model.addConstrs((mu_hat[i,j] <= sum( x_hat[i-1,k] for k in range(0,j)) + x[i-1] for i in range(1,model._nincrements) for j in range(model._nperiods-t)))
    aux = sum(y[i,d] * instancia[model._infoobj[d]].iloc[i] for i in range(len(instancia)) for d in range(model._ndestinations))
    aux += sum(model._discount_rate**t *sum(p[i]*x_hat[i,j] for i in range(model._nincrements)) for j in range(model._nperiods-t))
    model.setObjective(aux, GRB.MAXIMIZE)
  #Anadimos las restricciones
  #Restricciones de capacidad
  if flag_full:
    lambda_var = model.addVars(len(model._infocons),vtype = GRB.BINARY,name = "lambda_var")
  index = 0
  for cons in model._infocons:
    aux = cons[3]
    if cons[1] == '*':
      model.addConstr(sum(y[i,d] * instancia[cons[0]].iloc[i]  for i in range(len(instancia)) for d in range(model._ndestinations)) <= aux,"capacity_"+str(index))
      if flag_full:
        model.addConstr(aux*lambda_var[index] + sum(y[i,d] * instancia[cons[0]].iloc[i]  for i in range(len(instancia)) for d in range(model._ndestinations)) >= aux ,"full_capacity_"+str(index))
    else:
      model.addConstr(sum(y[i,int(cons[1])] * instancia[cons[0]].iloc[i]  for i in range(len(instancia))) <= aux,"capacity_"+str(index))
      if flag_full:
        model.addConstr(aux*lambda_var[index] + sum(y[i,int(cons[1])] * instancia[cons[0]].iloc[i]  for i in range(len(instancia))) >= aux,"full_capacity_"+str(index))
    index += 1
  if flag_full:
    model.addConstr(sum(lambda_var[i] for i in range(index)) <= index-1) 
  #Restricciones de precedencia
  model.addConstrs((model._oincrements[i] * x[i] - mu[i]*model._qincrements[i] <= 0 for i in range(model._nincrements)), 'p1')
  model.addConstrs((model._oincrements[i] * x[i] - mu[i+1]*model._qincrements[i]>= 0 for i in range(model._nincrements-1)), 'p2')
  #Restricciones de porcentajes; para cada incremento i, se debe extraer el mismo procentaje de toneladas de cada bloque en i.
  model.addConstrs((x[i] == sum(y[b,d] for d in range(model._ndestinations)) for i in range(model._nincrements) for b in model._bincrements[i])) 

#########################################################################################################################################################################################################################################
#Funciones auxiliares

def get_varx(model):
  # Funcion para obtener la variable x del modelo
  x = [var.x for var in model.getVars() if "x" in var.VarName]
  return x

def update_increments(model,q):
  # Cambia las toneladas de cada incremento luego de extraerlas
  index = 0
  while q>0 and index < model._nincrements:
    if model._qincrements[index] >= q:
      model._qincrements[index] = model._qincrements[index] - q
      q = 0
    else:
      q = q - model._qincrements[index]
      model._qincrements[index] = 0
    index += 1
          
def get_vary(model,instancia):
  # Funcion para obtener la variable y del modelo
  y = [[0,0] for i in range(len(instancia))]
  for b in range(len(instancia)):
    for d in range(model._ndestinations):
      y[b][d] = model.getVarByName('y['+str(b)+','+str(d)+']').x
  return y

def get_u_obj(y,instancia,model):
  # Valor de la funcion u en el modelo
  return sum(y[i][d] * instancia[model._infoobj[d]].iloc[i] for i in range(len(instancia)) for d in range(model._ndestinations))

def update_constraints(model,Q,x = None):
  aux_cons     = model.getConstrByName('aux_cons')
  aux_cons.rhs = Q
  mu = [var for var in model.getVars() if "mu" in var.VarName]
  for i in range(model._nincrements):
    p1 = model.getConstrByName('p1['+str(i)+']')
    model.chgCoeff(p1, mu[i] , -model._qincrements[i])
    if i < model._nincrements - 1:
      p2 = model.getConstrByName('p2['+str(i)+']')
      model.chgCoeff(p2, mu[i+1] , -model._qincrements[i])
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
  for i in range(model._nincrements):
    aux_list.append(model._oincrements[i])
  model._qincrements = aux_list[:]
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
  increments = [0 for i in range(len(instancia))]
  for i in range(len(model._bincrements)):
    blocks = model._bincrements[i]
    for b in blocks:
      increments[b] = i
  return increments

def codify_y(y0,y,t,model,instancia):
  increments = incrementens_p_block(model,instancia)
  for b in range(len(instancia)):
    for d in range(model._ndestinations):
      if y[b][d] > 0:
        #Se codifica como bloque,destino,valor de y, incremento al que pertenece
        y0.append([b,d,t,y[b][d],increments[b]])

#########################################################################################################################################################################################################################################
#Funcion solver

def original_solver(model,instancia,option = 'pwl',flag_full = False):
  model.setParam('OutputFlag',0)
  # Solver para el modelo usando regresion lineal/piece wise linear
  # Variables a rellenar para la regresion lineal/piece wise linear
  Q0  = sum(model._oincrements[i] for i in range(model._nincrements))
  Q_k = Q0
  aux_q_array = [[(Q0*i)/5] for i in range(5,-1,-1)]
  aux_v_array  = [0 for i in range(5,-1,-1)]
  #Se crea el modelo
  #print('Creacion modelo')  
  create_model(model,instancia,Q0,1,aux_v_array,aux_q_array,option, flag_full)
  #print("Modelo creado")
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
    #print("Segundo ciclo")
    while Q_k > 0 and t <= model._nperiods:
      #Pptimizacion de model cuando resta Q_k en la mina en el tiempo t
      print('Comienzo optimizacion')
      model.optimize()
      print('Optimizacion Terminada')
      #Se obtienen los vectores solucion x,y,z y los valores u,q
      bar_x = get_varx(model)
      #print('Lo que saco',bar_x)
      x_array.append(bar_x)
      bar_y = get_vary(model,instancia)
      bar_z = [bar_y[b][1] for b in range(len(bar_y))] #Suponemos que 1(el indice 1) es el destino refinadero
      bar_q =  sum(bar_x[i]*model._oincrements[i] for i in range(model._nincrements))
      u_bar_q_t = get_u_obj(bar_y,instancia,model)
      #Actualizacion de las toneladas por incremento
      update_increments(model,bar_q)
      aux_qq = []
      for h in range(model._nincrements):
        aux_qq.append(model._qincrements[h]/model._oincrements[h])
      #print('Lo que me queda ',aux_qq)
      #print('Original',model._oincrements)
      #model.write('/content/drive/MyDrive/Colab Notebooks/Ejemplo'+str(t)+'.lp')
      #Se guardan los valores obtenidos
      u_array.append(u_bar_q_t)
      q_bar_array.append(bar_q)
      #Se quita el tonelaje extraido
      Q_k = Q_k - bar_q
      print('Periodos t = ',t,'. Toneladas extraidas =',bar_q,'. Toneladas restantes = ',Q_k)
      #Se codifica la solucion y de la forma: bloque, destino, tiempo, valor de y
      codify_y(y_array,bar_y,t,model,instancia)
      #Se aumenta el periodo
      t = t + 1
      #Se actualiza el modelo con respecto a las toneladas restantes 
      update_constraints(model,Q_k,bar_x)
      #Se actualiza el modelo con respecto al tiempo
      if t == model._nperiods:
        update_objective(model,instancia,aux_q_array,aux_v_array,model._nperiods,option)
    #Se calcula lo que vale la mina al tener todas las toneladas a partir de los valores u
    print('Toneladas finales = ',Q_k)
    V_Q_t = sum(u_array[i]*(model._discount_rate**i) for i in range(len(u_array)))
    print('Valor de la mina = ',V_Q_t)
    #Se crean los vectores para guardar los valores V(Q-sum q_i) y Q - sum q_i
    aux_v_array = [V_Q_t - sum(u_array[j]*(model._discount_rate**(j)) for j in range(0,i)) for i in range(0,len(u_array)+1)]
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
    if vk0>=vk1 and k> 2:
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
  tol = 10**(-3)
  tf = max(y[2])
  t0 = min(y[2])
  if t0 == 0:
    tf = tf + 1
  output = []
  p_increments = [0 for i in range(model._nincrements)]
  for i in range(int(tf)):
    problem = [str(i)]
    if t0 == 1:
      t = i+1
    else:
      t = i
    y_t = y[y[2] == t]
    index = 1
    for cons in model._infocons:
      r = 0
      if cons[1] == '*':
        for q in range(len(y_t)):
          b,val_y = int(y_t[0].iloc[q]),y_t[3].iloc[q]
          r += val_y * instancia[cons[0]].iloc[b]
      else:
        for q in range(len(y_t)):
          b,d,val_y =  int(y_t[0].iloc[q]),int(y_t[1].iloc[q]),y_t[3].iloc[q]
          if d == int(cons[1]):
            r += val_y * instancia[cons[0]].iloc[b]
      if r-cons[3] > tol :
          problem.append('R. Capacidad '+str(index))
      index +=1
    for j in range(model._nincrements):
      y_j = y_t[y_t[4] == j]
      if len(y_j) > 0:
        b0     = y_j[0].iloc[0]
        y_0    = y_j[y_j[0] == b0]
        prom_0 = y_0[3].sum()
        p_increments[j] += prom_0
        b_list = y_j[0].unique()
        for b in b_list:
          if b != b0:
            y_aux = y_j[y_j[0] == b]
            prom  = y_aux[3].sum()
            if prom - prom_0 > tol or prom - prom_0 < -tol:
              problem.append('Bloque no consistente')
      
    for j in range(1,model._nincrements):
        if p_increments[j] > tol:
          if p_increments[j-1] < 1-tol:
            problem.append('Precedencia entre incrementos')    
    output.append(problem)
  
  feasible = True
  for i in range(len(output)):
    if len(output[i]) > 1:
       feasible = False
       break
  return feasible,output,p_increments  
  
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

#######

def write_tablas(Tabla,directory,i,Header=False, Index =False):
  directory = directory[:-4] +"Tabla"+ str(i) + ".txt"
  Tabla.to_csv(directory,header=Header,index=Index)

    