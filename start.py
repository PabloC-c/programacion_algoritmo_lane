import os
import fileinput
import sys

## Funcion para reemplazar email

def replaceAll(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(line,replaceExp)
        sys.stdout.write(line)
        
##    

# Minas en numeral
minas = { "kd" : "0", "chaiten" : "1"
                     ,"marvin" : "2", "palomo" : "3"}
# Tipos de modelo       
tipo_de_modelo ={"Libre" : " 0", "Restringido":" 1"}

# Paradas
tipo_de_parada = {"Concava":" 0","Cauchy":" 1"}     
                                            
print("Insertar nombre de la mina :")
nombre_mina = input()

print("Utiliza cluster (Si/No)")
utilizar_cluster = input()

if utilizar_cluster == "Si":
    print("Insertar email :")
    mail = input()    
    
print("Precedencias (Ordenadas/Generales)")
precedencias = input()

print("Modelo (Libre/Restringido)")
modelo = input()

print("Parada (Concava/Cauchy)")
parada = input()

if utilizar_cluster == "No":
    if precedencias == "Ordenadas":
        cmd = "python programa.py " + minas[nombre_mina] + tipo_de_modelo[modelo] + tipo_de_parada[parada]
        os.system(cmd)

    elif precedencias == "Generales":
        cmd = "python programa_alt.py " + minas[nombre_mina] + tipo_de_modelo[modelo] + tipo_de_parada[parada]
        os.system(cmd)    
    else :
        print("error precedencias")

elif utilizar_cluster == "Si":
    cmd = "Comando_sbatch/"+parada+"/"+modelo
    os.chdir(cmd)
    mina = int(minas[nombre_mina][-1])+1
    if precedencias == "Generales":
        texto ="_alt"
    else :
        texto = ""
    if modelo == "Libre":
        archivo = "programa"+str(mina)+texto+"_free.sh"
        cmd = "sbatch "+ archivo
        replaceAll(archivo,"--mail-user","#SBATCH --mail-user="+mail+"\n")
        print("comando",cmd)
        os.system(cmd)
    if modelo == "Restringido":
        archivo = "programa"+str(mina)+texto+"_restr.sh"
        cmd = "sbatch "+ archivo
        replaceAll(archivo,"--mail-user","#SBATCH --mail-user="+mail+"\n")
        print("comando",cmd)
        os.system(cmd)            
                   

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            