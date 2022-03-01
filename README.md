# programacion algoritmo lane
Instrucciones para ejecutar la implementación del algoritmo de Lane. Más especificaciones sobre el algoritmo y su implementación está en la carpeta "Informe"

# Instrucciones previas
- Es necesario crear una carpeta para guardar las soluciones entregadas
- La informacion de la mina debe estar en dos archivos con formato .prob y .blocks

# Ejecucción
El algoritmo es ejecutado con el comando "python start.py ordenado archivo_prob archivo_blocks archivo_sols restringido parada x_binario opcion_v". Este commando permite escoger las siguientes opciones :
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
- opcion_v :
    - Estimacion de la funcion V (funcion valor futuro) por partes. homo para asumir homogeneidad en la mina y phase para calcular una estimacion para cada phase de la mina. Esta opcion solo aplica para el modelo de precedencias 
      generales    

# Recomendaciones.
En caso de intentar resolver un problema muy grande, es recomendable utilizar sbatch para ejecutar el programa.

# Ejemplo:
Si quisieramos ejecutar el archivo con presedencias generales y el modelo libre, considerando el metodo de parada de cauchy, la variable x binaria y la funcion valor futuro estimada para cada phase. Las direcciones de los archivos son kd.prob y kd.blocks y ademas
queremos guardar la solucion en el archivo kd_ejemplo_sol.sol. Para esto debemos ejecutar el comando:

  - python start.py 0 kd.prob kd.blocks kd_ejemplo_sol.sol 0 cauchy 1 phase