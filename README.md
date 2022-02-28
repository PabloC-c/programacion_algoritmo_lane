# programacion algoritmo lane
Instrucciones para ejecutar la implementación del algoritmo de Lane. Más especificaciones sobre el algoritmo y su implementación está en la carpeta "Informe"

# Instrucciones previas
- Es necesario crear una carpeta para guardar las soluciones entregadas
- La informacion de la mina debe estar en dos archivos con formato .prob y .blocks

# Ejecucción
El algoritmo es ejecutado con el comando "python start.py Archivo_prob Archivo_block Soluciones Restriccion Parada Modelo". Este commando permite escoger las siguientes opciones :
- Archivo_prob :
    - Dirección del archivo .prob a utilizar    
- Archivo_block :
    - Dirección del archivo .block a utilizar      
- Soluciones :
    - Carpeta donde se guardarán las soluciones    
- Restriccion (Restringido/Libre) :
    - Utilizar modelo Restringido (Obliga al algoritmo a extraer en cada iteración) o Libre (No obliga a extraer) 
- Parada (Concava/Cauchy) :
    - Utilizar una parada Concava (El algoritmo para cuando el nuevo valor objetivo es mas bajo que el anterior) o el método de Cauchy (Busca la convergencia del algoritmo)
- Modelo (Ordenado/General):
    - Utilizar el modelo con precedencias ordenadas (solo incrementos) o con precedencias generales (bancos y fases)

# Recomendaciones.
En caso de intentar resolver un problema muy grande, es recomendable utilizar sbatch para ejecutar el programa.
