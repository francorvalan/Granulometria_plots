import pandas as pd
import numpy as np
import re 
def search_amount_size(df, muestra, size_search):
    """
    Busca el la cantidad acumulada de material pasante menor o igual 
    al tamaño dado para la muestra especifica.

    Args:
        df (DataFrame): DataFrame melted con columnas 'Muestra', 'Tamaño' y 'log_Tamaño'.
        muestra (str): Nombre de la muestra a filtrar.
        size_search (float): Tamaño en mm para buscar.
    """
    # Tamaños de referencia segun AASHTO
    
    Sizes={'Arcilla': 0.002, 'Limo': 0.075, 'Arena': 4.75}
    if type(size_search) == str:
        size_search = Sizes.get(size_search)  # Convertir a mm si es necesario

    df_filtered = df[df['Muestra'] == muestra]
    if df_filtered.empty:
        raise ValueError(f"La muestra '{muestra}' no contiene datos para el tamaño buscado.")
    if not np.any(np.isclose(df_filtered['Tamaño'], size_search)):
        raise ValueError(f"No se encontró el tamaño {size_search} mm en la muestra '{muestra}'.")
    return df_filtered[np.isclose(df_filtered['Tamaño'], size_search)]['% Pasante'].values[0]

def interpolar_valor(df, x,muestra_example):
    """
    Realiza una interpolación lineal para predecir el valor de % Pasante dado un Tamaño (x) en mm.
    df debe tener columnas 'log_Tamaño' y '% Pasante'.
    """
    df=df[df['Muestra'] == muestra_example]  # Filtrar por muestra específica
    x_log = np.log10(x)  # Convertir x a log10
    df_ordenado = df.sort_values('Tamaño').reset_index(drop=True)

    # Asegurarse de que el valor x esté dentro del rango
    if x_log < df_ordenado['log_Tamaño'].min() or x_log > df_ordenado['log_Tamaño'].max():
        raise ValueError("x está fuera del rango de log_Tamaño.")

    # Buscar los dos puntos que rodean x
    menor = df_ordenado[df_ordenado['log_Tamaño'] <= x_log].iloc[-1]
    mayor = df_ordenado[df_ordenado['log_Tamaño'] >= x_log].iloc[0]

    x0, y0 = menor['log_Tamaño'], menor['% Pasante']
    x1, y1 = mayor['log_Tamaño'], mayor['% Pasante']

    # Si x coincide exactamente con un valor existente
    if x0 == x1:
        return y0

    # Interpolación lineal y0 + pendiente*delta_x
    y = y0 + abs(x_log - x0) * abs(y1 - y0) / abs(x1 - x0)
    # print(f'menor = {menor}')
    # print(f'x0 = {x0}, y0 = {y0}')
    # print(f'\n\nmayor = {mayor}')
    # print(f'x1 = {x1}, y1 = {y1}')
    return y

def interpolar_percentil(df, percentil,muestra_example):
    """
    Dado un DataFrame con columnas 'log_Tamaño' y '% Pasante',
    encuentra el valor de log_Tamaño asociado a un cierto percentil.
    """
    df=df[df['Muestra'] == muestra_example]  # Filtrar por muestra específica
    df_ordenado = df.sort_values('Tamaño').reset_index(drop=True)
    if percentil < df_ordenado['% Pasante'].min() or percentil > df_ordenado['% Pasante'].max():
        raise ValueError("El percentil está fuera del rango de % Pasante.")
    
    if any(df_ordenado['% Pasante']==percentil):
        x = float(df_ordenado[df_ordenado['% Pasante'] == percentil]['log_Tamaño'].values[0])
        return x
    else:
        menor = df_ordenado[df_ordenado['% Pasante'] < percentil].iloc[-1]
        mayor = df_ordenado[df_ordenado['% Pasante'] > percentil].iloc[0]

        y0, x0 = menor['% Pasante'], menor['log_Tamaño']
        y1, x1 = mayor['% Pasante'], mayor['log_Tamaño']

        if y0 == y1:
            return x0

        # Interpolación lineal inversa
        x = x0 + (percentil - y0) * (x1 - x0) / (y1 - y0)
        return x
    
def A_line(LL):
    return(0.73*(LL-20))
def U_line(LL):
    return(0.9*(LL-8))

def plasticidad(LL, PI):
    """
    Calcula la plasticidad de una muestra específica.
    """
    A_location="Upper" if PI > A_line(LL) else "Lower"
    if A_location == "Upper":
        if PI > 7:
            if LL<50:
                Plasticity="CL o OL"
            else:
                Plasticity="CH o OH"
        elif PI <= 7 and PI > 4:
            if LL<50:
                Plasticity="CL o ML"
        else:
            print("PI <= 4, no se puede determinar la plasticidad") 
    else:
        if LL<50:
            Plasticity="ML o OL"
        else:
            Plasticity="MH o OH"
            
    return Plasticity

def plasticidad_plot(LL, PI):
    # Límites del gráfico

    # Líneas A y U
    def A_line(LL): return 0.73 * (LL - 20)
    def U_line(LL): return 0.9 * (LL - 8)

    # Crear la figura
    plt.figure(figsize=(6, 4))

    LL_s = np.linspace(0, 120, 500)
    plt.plot(LL_s, A_line(LL_s), 'k-', label='A-line')
    plt.plot(LL_s, U_line(LL_s), 'k--', label='U-line')

    plt.fill(
        [0, 20, 50, 50, 0], 
        [0, 0, A_line(50), 0, 0],
        color='lightgreen', alpha=0.3#, label='ML u OL'
    )
    plt.fill(
        [15.78, 29.59, 50, 50], 
        [U_line(15.78), A_line(29.59), A_line(50), U_line(50)],
        color='lightpink', alpha=0.3#, label='CL u OL'
    )
    plt.fill(
        [12.44, 25.48, 29.59, 15.78], 
        [U_line(12.44), A_line(25.48), A_line(29.59), U_line(15.78)],
        color='orange', alpha=0.3#, label='CL u OL'
    )

    plt.fill(
        [50,100, 100, 50],
        [0,0, A_line(100), A_line(50)], 
        color='lightblue', alpha=0.3#, label='ML u OL'
    )

    plt.fill(
        [50, 100, 85, 50], 
        [A_line(50), A_line(100),U_line(85), U_line(50)],
        color='khaki', alpha=0.3#, label='CH u OH'
    )
    plt.vlines(50,0,70, colors='gray', linestyles='--')  # Línea vertical en LL=50

    # Añadir el punto de la muestra
    plt.plot(LL, PI, 'o', color='black', markersize=3)

    # Ejes y límites
    plt.xlim(0, 100)
    plt.ylim(0, 70)
    plt.xlabel('Límite Líquido, LL')
    plt.ylabel('Índice de Plasticidad, PI')
    plt.grid(True)

    # Agregar etiquetas en las zonas
    plt.text(17, 5, 'CL - ML', fontsize=9)
    plt.text(35, 20, 'CL u OL', fontsize=9)
    plt.text(35, 5, 'ML u OL', fontsize=9)
    plt.text(75, 20, 'MH u OH', fontsize=9)
    plt.text(60, 38, 'CH u OH', fontsize=9)

    # Mostrar las líneas en la leyenda si querés
    plt.legend()

    plt.title('Diagrama de Plasticidad')
    plt.tight_layout()
    plt.show()



class ClasificacionSuelo:
        def __init__(self, muestra):
            self.muestra = muestra
            self.Errores = []
            self.Grupo = None
            self.Clase = None
            self.Gravas = None
            self.Arenas = None
            self.Finos = None
            self.Limos = None
            self.Arcillas = None
            self.LL=None
            self.PI=None
            self.Plasticidad=None
            self.D10 = None 
            self.D30 = None
            self.D60 = None
            self.Cu = None
            self.Cc = None
        def update(self, **kwargs):
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(f"La propiedad '{key}' no existe en ClasificacionSuelo.")
def texture_preprocess(df,muestra,LL=None, PI=None):
    if muestra not in df['Muestra'].values:
        raise ValueError(f"La muestra '{muestra}' no se encuentra en el la base de datos suministrada.")
    df = df[df['Muestra'] == muestra]  # Filtrar por muestra específica

    if df.empty:
        raise ValueError(f"La base de datos suministrada no contiene datos")

    #Errores= []	
    Result = ClasificacionSuelo(muestra)

    def append_error(fx, message,Result,**kwargs):
        try:
            return(fx(**kwargs))
        except ValueError:
            Result.Errores.append(message)
            return None       
    if LL is None or np.isnan(LL):
        Result.Errores.append(f'Faltan datos de Límite Líquido (LL)')
    else:
        Result.LL = LL
    if PI is None or np.isnan(PI):
        Result.Errores.append(f'Faltan datos de Plasticidad (PI)')
    else:
        Result.PI = PI
    
    Result.Arcillas = append_error(search_amount_size,
                           'Falta granulometria de Arcillas (0,002 mm)',
                           Result,df=df, muestra=muestra, size_search='Arcilla')
    if Result.Arcillas is None:
        Result.Arcillas = append_error(interpolar_valor,
                           'No se pudo interpolar el valor de Arcillas (0,002 mm) debido a que no hay datos suficientes.',
                           Result,df=df,x=0.002, muestra_example=muestra)

    Result.Finos = append_error(search_amount_size,
                           'Falta granulometria de Limos (0,075 mm)',
                           Result,df=df, muestra=muestra, size_search='Limo')
    if Result.Finos is None:
        Result.Finos = append_error(interpolar_valor,
                            'No se pudo interpolar el valor de Limos (0,075 mm) debido a que no hay datos suficientes.',
                            Result,df=df,x=0.075, muestra_example=muestra)


    Menor_gravas = append_error(search_amount_size,
                        'Falta granulometria de Arenas 4,75 mm)',
                        Result,df=df, muestra=muestra, size_search='Arena')
    if Menor_gravas is None:
        Menor_gravas =append_error(interpolar_valor,
                           'No se pudo interpolar el valor de Arenas (4,75 mm) debido a que no hay datos suficientes.',
                           Result,df=df,x=4.75, muestra_example=muestra) 

    D10 = append_error(interpolar_percentil,
                           'No se pudo interpolar D10 debido a falta de datos.',
                           Result,df=df, percentil=10, muestra_example=muestra)
    if D10 is not None:
        D10 = 10**D10
        Result.D10=D10
    D30 = append_error(interpolar_percentil,
                           'No se pudo interpolar D30 debido a falta de datos.',
                           Result,df=df, percentil=30, muestra_example=muestra)
    if D30 is not None:
        D30 = 10**D30
        Result.D30=D30
    D60 = append_error(interpolar_percentil,
                           'No se pudo interpolar D60 debido a falta de datos.',
                           Result,df=df, percentil=60, muestra_example=muestra)
    if D60 is not None:
        D60 = 10**D60
        Result.D60=D60
    
    if D60 is not None and D10 is not None:
        Result.Cu= (D60/D10) if D10 != 0 else None # Calcular el Coeficiente de uniformidad
        Result.Cc= (D30**2/(D60*D10)) # Calcular el Coeficiente de gradación


    if PI is None or np.isnan(PI) or  LL is None or np.isnan(LL):
        Result.Plasticidad = 'ML o OL'  # Asignar una categoría por defecto si no se proporcionan LL y PI
        Result.Errores.append('Se asumió plasticidad baja (ML o OL)')
    else:
        Result.Plasticidad= plasticidad(LL, PI)

    if Menor_gravas is not None :
        Result.Gravas = 100 - Menor_gravas
    else:
        Result.Errores.append('Falta granulometria de Gravas (>4,75 mm)')

    if Menor_gravas is not None and  Result.Finos is not None:
        Result.Arenas = Menor_gravas -  Result.Finos

    if Result.Arcillas is not None and Result.Finos is not None:
        Result.Limos = Result.Finos - Result.Arcillas
    else:
        Result.Limos = None

    return Result 

def USCS_granular(Result):
    """
    Determina la clasificación USCS para suelos finos basándose en LL y PI.
    Args:
    Result (objeto): Un objeto de la clase ClasificacionSuelo que contiene las propiedades necesarias para la clasificación.
    """

    # Verificar si todas las variables necesarias están presentes
    Variables_necesarias_clasificacion = [Result.Gravas,Result.Arenas, Result.Finos, Result.Cu, Result.Cc, Result.Plasticidad]
    if any(x is None for x in Variables_necesarias_clasificacion):
        Result.Errores.append("No se puede determinar la clasificación USCS ante la falta de datos.")
        Result.Errores = '\n'.join(Result.Errores) if Result.Errores else None
        return Result
    else:
        if Result.Gravas> Result.Arenas:
            if Result.Finos<5:
                if Result.Cu >=4 and Result.Cc >=1 and Result.Cc <=3:
                    Result.Grupo = "GW"
                    if Result.Arenas < 15:
                        Result.Clase = "Grava bien graduada"
                    else:
                        Result.Clase = "Grava bien graduada con arenas"
                elif Result.Cu < 4 or Result.Cc < 1 or Result.Cc > 3:
                    Result.Grupo = "GP"
                    if Result.Arenas < 15:     
                        Result.Clase = "Grava mal graduada"
                    else:
                        Result.Clase = "Grava mal graduada con arenas"
            elif 5 <= Result.Finos <= 12:
                if Result.Cu >=4 and Result.Cc >=1 and Result.Cc <=3:
                    if re.search('CL o OL|CH o OH|CL-ML',Result.Plasticidad):
                        Result.Grupo ="GW-GC"
                        if Result.Arenas < 15:
                            Result.Clase = "Grava bien graduada con arcilla"
                        else:
                            Result.Clase = "Grava bien graduada con arcilla y arenas"
                    elif re.search('ML o OL|MH o OH',Result.Plasticidad):
                        Result.Grupo ="GW-GM"
                        if Result.Arenas < 15:
                            Result.Clase = "Grava bien graduada con limo"
                        else:
                            Result.Clase = "Grava bien graduada con limo y arenas"
                    else: print("No se puede determinar la clasificación USCS para suelos finos")
                elif Result.Cu < 4 or Result.Cc < 1 or Result.Cc > 3:
                    if re.search('CL o OL|CH o OH|CL-ML',Result.Plasticidad):
                        Result.Grupo ="GP-GC"
                        if Result.Arenas < 15:
                            Result.Clase = "Grava mal graduada con arcilla"
                        else:
                            Result.Clase = "Grava mal graduada con arcilla y arenas"
                    elif re.search('ML o OL|MH o OH',Result.Plasticidad):
                        Result.Grupo ="GP-GM"
                        if Result.Arenas < 15:
                            Result.Clase = "Grava mal graduada con limo"
                        else:
                            Result.Clase = "Grava mal graduada con limo y arenas"
                    else: print("No se puede determinar la clasificación USCS para suelos finos")
            elif Result.Finos > 12:
                if re.search('ML o OL|MH o OH',Result.Plasticidad):
                    Result.Grupo = "GM"
                    if Result.Arenas < 15:
                        Result.Clase = "Grava limosa"
                    else:
                        Result.Clase = "Grava limosa con arenas"
                elif re.search('CL o OL|CH o OH',Result.Plasticidad):
                    Result.Grupo = "GC"
                    if Result.Arenas < 15:
                        Result.Clase = "Grava arcillosa"
                    else:   
                        Result.Clase = "Grava arcillosa con arenas"
                elif re.search('CL-ML',Result.Plasticidad):
                    Result.Grupo = "GC-GM"
                    if Result.Arenas < 15:
                        Result.Clase = "Grava limosa-arcillosa"
                    else:   
                        Result.Clase = "Grava limosa-arcillosa con arenas"
                else: # Gravas <= Arenas:
                    print("Erorr: No se puede determinar la clasificación USCS")

        elif Result.Gravas <= Result.Arenas:
            if Result.Finos < 5:
                if Result.Cu >=6 and Result.Cc >=1 and Result.Cc <=3:
                    Result.Grupo = "SW"
                    if Result.Gravas < 15:
                        Result.Clase = "Arena bien graduada"
                    else:
                        Result.Clase = "Arena bien graduada con gravas"
                elif Result.Cu < 6 or Result.Cc < 1 or Result.Cc > 3:
                    Result.Grupo = "SP"
                    if Result.Gravas < 15:
                        Result.Clase = "Arena mal graduada"
                    else:
                        Result.Clase = "Arena mal graduada con gravas"

            elif 5 <= Result.Finos <= 12:
                if Result.Cu >=6 and Result.Cc >=1 and Result.Cc <=3:
                    if re.search('ML o OL|MH o OH',Result.Plasticidad):
                        Grupo = "SW-SM"
                        if Result.Gravas < 15:
                            Result.Clase = "Arena bien graduada con limo"
                        else:
                            Result.Clase = "Arena bien graduada con limo y gravas"
                    elif re.search('CL o OL|CH o OH|CL-ML',Result.Plasticidad):
                        Result.Grupo = "SW-SC"
                        if Result.Gravas < 15:
                            Result.Clase = "Arena bien graduada con arcilla"
                        else:
                            Result.Clase = "Arena bien graduada con arcilla y gravas"
                    else: print("No se puede determinar la clasificación USCS para suelos finos")
                elif Result.Cu < 6 or Result.Cc < 1 or Result.Cc > 3:
                    if re.search('ML o OL|MH o OH',Result.Plasticidad):
                        Result.Grupo = "SP-SM"
                        if Result.Gravas < 15:
                            Result.Clase = "Arena mal graduada con limo"
                        else:
                            Result.Clase = "Arena mal graduada con limo y gravas"
                    elif re.search('CL o OL|CH o OH|CL-ML',Result.Plasticidad):
                        Result.Grupo = "SP-SC"
                        if Result.Gravas < 15:
                            Result.Clase = "Arena mal graduada con arcilla"
                        else:
                            Result.Clase = "Arena mal graduada con arcilla y gravas"
                            
                    else: print("No se puede determinar la clasificación USCS para suelos finos")

            elif Result.Finos > 12:
                if re.search('ML o OL|MH o OH',Result.Plasticidad):
                    Result.Grupo = "SM"
                    if Result.Gravas < 15:
                        Result.Clase = "Arena limosa"
                    else:
                        Result.Clase = "Arena limosa con gravas"
                elif re.search('CL o OL|CH o OH',Result.Plasticidad):
                    Result.Grupo = "SC"
                    if Result.Gravas < 15:
                        Result.Clase = "Arena arcillosa"
                    else:   
                        Result.Clase = "Arena arcillosa con gravas"
                elif re.search('CL-ML',Result.Plasticidad):
                    Result.Grupo = "SC-SM"
                    if Result.Gravas < 15:
                        Result.Clase = "Arena limosa-arcillosa"
                    else:   
                        Result.Clase = "Arena limosa-arcillosa con gravas"
                else:  # Error
                    print("Erorr: No se puede determinar la clasificación USCS")

        Result.Errores = '\n'.join(Result.Errores) if Result.Errores else None
        return (Result)

def USCS_finos(Result,Organico=False,Peso_seco=None):
    """
    Determina la clasificación USCS para suelos finos basándose en LL y PI.
    Args:
    Result (objeto): Un objeto de la clase ClasificacionSuelo que contiene las propiedades necesarias para la clasificación.
    Organico (bool): Indica si el suelo es orgánico. Por defecto es False.
    Peso_seco (float): Peso seco del suelo, si se conoce. Por defecto es None.
    """


    if Result.PI is None or np.isnan(Result.PI) or  Result.LL is None or np.isnan(Result.LL):
        PI_upper_A_line = False  # Asignar una categoría por defecto si no se proporcionan LL y PI
        Result.Errores.append('Se asumió plasticidad baja, por debajo de la linea "A"')
    else:
        PI_upper_A_line = Result.PI > A_line(Result.LL)

    N200 = Result.Limos
    Variables_necesarias_clasificacion = [Result.LL,Result.PI,N200,PI_upper_A_line,Result.Arenas,Result.Gravas]
    if any(x is None for x in Variables_necesarias_clasificacion):
        Result.Errores.append("No se puede determinar la clasificación USCS ante la falta de datos.")
        Result.Errores = '\n'.join(Result.Errores) if Result.Errores else None
        return Result
    else:
        if Result.LL < 50:
            if not Organico:
                if Result.PI>7 and PI_upper_A_line:
                    Result.Grupo= "CL"
                    if N200 < 30:
                        if N200 <15:
                            Result.Clase = "Arcilla fina"
                        else:# 15 <= N200 < 30:
                            if Result.Arenas>= Result.Gravas:
                                Result.Clase = "Arcilla fina con arenas"
                            else:
                                Result.Clase = "Arcilla fina con gravas"
                    else:
                        if Result.Arenas>= Result.Gravas:
                            if  Result.Gravas < 15:
                                Result.Clase = "Arcilla fina arenosa"
                            else:
                                Result.Clase = "Arcilla fina arenosa con gravas"
                        else:
                            if Result.Arenas < 15:
                                Result.Clase = "Arcilla fina gravosa"
                            else:
                                Result.Clase = "Arcilla fina gravosa con arenas"
                elif 4 <= Result.PI <= 7 and PI_upper_A_line:
                    Result.Grupo= "CL-ML"
                    if N200 < 30:
                        if N200 <15:
                            Result.Clase = "Arcilla limosa"
                        else: # 15 <= N200 < 30:
                            if Result.Arenas>= Result.Gravas:
                                Result.Clase = "Arcilla limosa con arenas"
                            else:
                                Result.Clase = "Arcilla limosa con gravas"
                    else:
                        if Result.Arenas>= Result.Gravas:
                            if  Result.Gravas < 15:
                                Result.Clase = "Arcilla arenosa-limosa"
                            else:
                                Result.Clase = "Arcilla arenosa-limosa con gravas"
                        else:
                            if Result.Arenas < 15:
                                Result.Clase = "Arcilla gravosa-limosa"
                            else:
                                Result.Clase = "Arcilla gravosa-limosa con arenas"

                else: # PI < 4 or !PI_upper_A_line
                    Result.Grupo= "ML"
                    if N200 < 30:
                        if N200 <15:
                            Result.Clase = "Limo"
                        else:# 15 <= N200 < 30:
                            if Result.Arenas>= Result.Gravas:
                                Result.Clase = "Limo con arenas"
                            else:
                                Result.Clase = "Limo con gravas"
                                
                    if Result.Arenas>= Result.Gravas:
                        if Result.Gravas < 15:
                            Result.Clase = "Limo arenoso"
                        else:
                            Result.Clase = "Limo arenoso con gravas"
                    else:
                        if Result.Arenas < 15:
                            Result.Clase = "Limo gravoso"
                        else:
                            Result.Clase = "Limo gravoso con arenas"
            else: # Organico
                Result.Grupo= "OH"
                Result.Errores.append("Sueldo orgánico, no se puede determinar la clasificación USCS")
                print("Sueldo orgánico, no se puede determinar la clasificación USCS")
        else: # LL >= 50
            if not Organico:
                if PI_upper_A_line:
                    Result.Grupo = "CH"
                    if N200 < 30:
                        if N200 <15:
                            Result.Clase = "Arcilla gruesa"
                        else:
                            if Result.Arenas>= Result.Gravas:
                                Result.Clase = "Arcilla gruesa con arenas"
                            else:
                                Result.Clase = "Arcilla gruesa con gravas"
                    else:
                        if Result.Arenas>= Result.Gravas:
                            if  Result.Gravas < 15:
                                Result.Clase = "Arcilla gruesa arenosa"
                            else:
                                Result.Clase = "Arcilla gruesa arenosa con gravas"
                        else:
                            if Result.Arenas < 15:
                                Result.Clase = "Arcilla gruesa gravosa"
                            else:
                                Result.Clase = "Arcilla gruesa gravosa con arenas"
                else:# IP <= A_line(LL):
                    Result.Grupo = "MH"
                    if N200 < 30:
                        if N200 <15:
                            Result.Clase = "Limo elástico"
                        else:
                            if Result.Arenas>= Result.Gravas:
                                Result.Clase = "Limo elástico con arenas"
                            else:
                                Result.Clase = "Limo elástico con gravas"
                    else: # 30 <= N200:
                        if Result.Arenas>= Result.Gravas:
                            if  Result.Gravas < 15:
                                Result.Clase = "Limo elástico arenoso"
                            else:
                                Result.Clase = "Limo elástico arenoso con gravas"
                        else: #  Arenas < Gravas:
                            if Result.Arenas < 15:
                                Result.Clase = "Limo elástico gravoso"
                            else:
                                Result.Clase = "Limo elástico gravoso con arenas"
            if Organico:
                Result.Grupo= "OH"
                Result.Errores.append("Sueldo orgánico, no se puede determinar la clasificación USCS")
                print("Sueldo orgánico, no se puede determinar la clasificación USCS")
        Result.Errores = '\n'.join(Result.Errores) if Result.Errores else None
        return (Result)

def to_dataframe(self):
    import pandas as pd
    return pd.DataFrame([
        {'Variable': attr, f'{self.muestra}': getattr(self, attr)}
        for attr in vars(self) if attr != 'muestra'
    ])

def df_process(df):
    import pandas as pd
    import numpy as np
    df_plasticidad = df.iloc[:2].copy()
    df_plasticidad = df_plasticidad.melt(id_vars=[df_plasticidad.columns[0]], var_name='Muestra', value_name='Limite')
    # Separar df_tamiz: contiene la fila 0 (encabezado), la fila 2 ("Tamaño") y desde la fila 3 en adelante
    df_tamiz = pd.concat([
        df.iloc[3:],
    ]).reset_index(drop=True)


    columna_tamaño  = df_tamiz.columns[0]
    df_tamiz = df_tamiz.melt(id_vars=[columna_tamaño], var_name='Muestra', value_name='% Pasante')
    df_tamiz = df_tamiz.dropna()
    df_tamiz[columna_tamaño] = pd.to_numeric(df_tamiz[columna_tamaño], errors='coerce')
    df_tamiz['% Pasante'] = pd.to_numeric(df_tamiz['% Pasante'], errors='coerce')
    df_tamiz.rename(columns={df_tamiz.columns[0]: 'Tamaño'}, inplace=True)

    df_tamiz['log_Tamaño'] = np.log10((df_tamiz['Tamaño']))
    df_tamiz = df_tamiz.dropna()
    return df_tamiz,df_plasticidad