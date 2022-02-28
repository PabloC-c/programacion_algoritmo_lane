# programacion algoritmo lane
Instrucciones para ejecutar la implementación del algoritmo de Lane. Más especificaciones sobre el algoritmo y su implementación está en la carpeta "Informe"

# Instrucciones previas
- Mover carpeta "Instancias" dos direcciones antes de este repositorio (cd ../../) <br/>
- Crear las siguientes carpetas en cd ../../ :  
    - sols2_free
    - sols2_restr
    - sols_model2_free
    - sols_model2_restr

# Ejecucción
El algoritmo es ejecutado con el comando "python start.py". Este commando permite escoger las siguientes opciones :
- Nombre de la mina :
    - Nombre de la mina a optimizar (kd/chaiten/marvin/palomo)    
- Utiliza cluster :
    - Utilizar cluster en caso de que el proceso de optimización se tarde demasiado.
        - En caso de utilizar cluster, es necesario entregar un email para notificar el estado del programa.       
- Precedencias :
    - Utilizar precedencias Ordenadas (solo incrementos) o Generales (bancos y fases)    
- Modelo :
    - Utilizar modelo Restringido (Obliga al algoritmo a extraer en cada iteración) o Libre (No obliga a extraer) 
- Parada :
    - Utilizar una parada Concava (El algoritmo para cuando el nuevo valor objetivo es mas bajo que el anterior) o el método de Cauchy (Busca la convergencia del algoritmo)

En caso de utilizar el cluster, los archivos .out y .err se crean en la carpeta "Comando_sbatch" dentro de sus directorios respectivos.
