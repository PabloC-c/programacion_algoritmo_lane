# programacion algoritmo lane
Instrucciones para ejecutar la implementación del algoritmo de Lane. Más especificaciones sobre el algoritmo y su implementación está en la carpeta "Informe"

# Instrucciones previas
- Es necesario crear una carpeta para guardar las soluciones entregadas
- La informacion de la mina debe estar en dos archivos con formato .prob y .blocks

# Ejecucción
El algoritmo es ejecutado con el comando "python start.py ordenado archivo_prob archivo_blocks archivo_sols restringido parada x_binario ". Este commando permite escoger las siguientes opciones :
- ordenado :
    - Valor binario 0 o 1. Nos permite elegir si queremos ejecutar el modelo con precedencias ordenadas o generales. 1 para precedencias ordenadas y 0 para generales 
- archivo_prob :
    - Dirección del archivo .prob a utilizar    
- archivo_blocks :
    - Dirección del archivo .blocks a utilizar
- archivo_sols:
    - Direccion + nombre del archivo .sol         
- restringido :
    - Valor binario 0 o 1. Nos permite elegir si queremos ejecutar el modelo restringido o libre. 1 para restringido y 0 para libre     
- parada :
    - Condicion de parada del algoritmo. concava para la condicion concava y cauchy para condicion de cauchy 
- x_binario :
    - Valor binario 0 o 1. Nos permite elegir si queremos que la variable x sea binaria o no. 1 para x binario y 0 para x continua

# Recomendaciones.
En caso de intentar resolver un problema muy grande, es recomendable utilizar sbatch para ejecutar el programa.

# Ejemplo:
Si quisieramos ejecutar el archivo con presedencias ordenadas y el modelo libre, considerando el metodo de parada de cauchy y la variable x binaria, las direcciones de los archivos son kd.prob y kd.blocks y ademas
queremos guardar la solucion en el archivo kd_ejemplo_sol.sol. Para esto debemos ejecutar el comando:

  - python start.py 1 kd.prob kd.blocks kd_ejemplo_sol.sol 0 cauchy 1