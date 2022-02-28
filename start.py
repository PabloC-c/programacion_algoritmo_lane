import os
import sys

archivo_prob = sys.argv[1]
archivo_block = sys.argv[2]
soluciones = sys.argv[3]
restriccion = sys.argv[4]
parada = sys.argv[5]
modelo = sys.argv[6]
cluster = False

if restriccion == "Libre":
    restriccion = int(0)
else:
    restriccion = int(1)

if parada == "Concava":
    parada = int(0)
else:
    parada = int(1) 
     
if modelo  == "Ordenado":
    print("ord")
    extension = ""
else:
    print("gen")
    extension = "_alt"   
    
    
cmd = "python programa"+extension+".py "+ archivo_prob +" "+ archivo_block+" "+str(restriccion) +" "+ str(parada) +" "+ soluciones
print(cmd)
os.system(cmd)