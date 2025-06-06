import pandas as pd
import numpy as np
import re 
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from itertools import cycle, islice
# Función para crear el gráfico con zoom mejorado
Ausenco_pallet = ['#101820','#004764',"#c6d1da",'#0095c8',"#b6c41d",
                  "#ffdb49",'#db7121','#a1292f','#662766',"#7E5B68",
                  "#267730","#4EBB5C","#4F5B66"]
def obtener_colores(paleta, n_colores):
    if paleta == "Ausenco":
        if n_colores <= len(Ausenco_pallet):
            return Ausenco_pallet[:n_colores]
        else:
            return list(islice(cycle(Ausenco_pallet), n_colores))
    else:
        cmap = plt.get_cmap(paleta)
        return [cmap(i / (n_colores - 1)) for i in range(n_colores)]


def crear_grafico(df, columna_tamaño, muestras_seleccionadas, colores, xlim=None, mostrar_puntos=False,
                  zoom=False,titulo=None,Agrupar_muestras=False,grupo2=None,nombre_grupos=None):

    df_transformado = df.dropna()
    df_transformado = df_transformado[df_transformado['Muestra'].isin(muestras_seleccionadas)]
    # Crear la figura
    fig, ax = plt.subplots(figsize=(10, 6))
    #fig, ax = plt.subplots(figsize=(18, 5))
    REFERENCIAS = {
        '0,002 mm':(2e-3),  # 0.002 mm
        '#200\n(0,075mm)':0.075,    # 0.08 mm
        '#4\n(4,75 mm)': 4.75     # 5 mm
    }

    # Graficar cada muestra seleccionada
    for muestra in muestras_seleccionadas:
        subset = df_transformado[df_transformado['Muestra'] == muestra]
        if not subset.empty:
            x = subset['Tamaño'].values
            y = subset['% Pasante'].values
            
            # Ordenar los valores
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            y = y[sort_idx]

            # Suavizar la curva usando interpolación
            # Crear función de interpolación
            #f = PchipInterpolator(x, y)
            
            # Definir rango para la interpolación
            #x_min = x.min() if xlim is None else max(x.min(), xlim[0])
            #x_max = x.max() if xlim is None else min(x.max(), xlim[1])
            
            # Crear puntos para curva suavizada (200 puntos en el rango visible)
            #x_nuevo = np.linspace(x_min, x_max, 20000)
            #y_nuevo = f(x_nuevo)
                        # Crear puntos para curva suavizada (200 puntos en el rango visible)
            x_nuevo = x
            y_nuevo = y
            # Graficar la curva con el color seleccionado
            linetype = '--' if Agrupar_muestras and muestra in grupo2 else '-'
            ax.plot(x_nuevo, y_nuevo, label=muestra, 
                    color=colores[muestra], linewidth=2, linestyle=linetype)
            
            # Agregar puntos si está habilitado (solo los que están en el rango visible)
            if mostrar_puntos:
                if xlim is not None:
                    mask = (x >= xlim[0]) & (x <= xlim[1])
                    x_points = x[mask]
                    y_points = y[mask]
                else:
                    x_points = x
                    y_points = y
                
                ax.scatter(x_points, y_points, color=colores[muestra], 
                          s=50, edgecolor='white', linewidth=1, zorder=3)
                
    def barras_textura(x_left,x_right,y,label,df):
        x_min = max(x_left,df['Tamaño'].min())
        
        x_max= min(x_right,df['Tamaño'].max())
        lines_color= '#383838'
        lines_width= 1.5
        lines_alpha=0.6
        if label== 'Gravas':
            ax.annotate('', xytext=((4.75),y), 
                        xy=(x_max, y),
                        arrowprops=dict(arrowstyle="-|>",alpha=lines_alpha,lw=lines_width, color=lines_color))
            ax.scatter([x_min], [y], marker='|', s=50, color=lines_color,alpha=lines_alpha,lw=lines_width)

        if label== 'Arenas':
            ax.hlines(y=y, xmin=x_min, 
            xmax= x_max,alpha=lines_alpha,
            colors=lines_color, linestyles='solid')
            ax.scatter([x_min, x_max], [y, y], marker='|', s=50, color=lines_color,alpha=lines_alpha)

        if df['Tamaño'].min() < (2e-3) and label=='Limos': # hay arcillas
            ax.hlines(y=y, xmin=x_min, 
            xmax= x_max,alpha=lines_alpha,
            colors=lines_color, linestyles='solid')
            ax.scatter([x_min, x_max], [y, y], marker='|', s=50, color=lines_color,alpha=lines_alpha)

        if df['Tamaño'].min() > (2e-3) and label=='Limos':
            ax.annotate('', xytext=(x_min*1,y), 
                        xy=(x_max*.97, y),
                        arrowprops=dict(arrowstyle="<|-",alpha=lines_alpha,lw=lines_width, color=lines_color))
            ax.scatter([x_max*.96], [y], marker='|', s=50, color=lines_color,alpha=lines_alpha)

        if df['Tamaño'].min() < (2e-3) and label=='Arcillas':
            ax.annotate('', xytext=(x_min*1,y), 
                        xy=(x_max*.97, y),
                        arrowprops=dict(arrowstyle="<|-",alpha=lines_alpha,lw=lines_width, color=lines_color))
            ax.scatter([x_max*.96], [y], marker='|', s=50, color=lines_color,alpha=lines_alpha)
    
        def middle_log(x_min,x_max):
            return 10**((np.log10(x_min)+np.log10(x_max))/2)
        # Gravas
        ax.text(middle_log(x_min, x_max), y+0.9, label,  alpha=0.5,
                        ha='center', va='bottom', rotation='horizontal',
                                bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=0.2))
    
    if not zoom:
        if df_transformado['Tamaño'].max() > (4.75):
            barras_textura(x_left=(4.75),x_right=100,y=101,label='Gravas',df=df_transformado)
        #barras_textura(x_left=(4.75),x_right=100,y=102,label='Gravas',df=df_transformado)
        barras_textura(x_left=(0.075),x_right=(4.75),y=101,label='Arenas',df=df_transformado)
        barras_textura(x_left=(0.002),x_right=(0.075),y=101,label='Limos',df=df_transformado)
        if df_transformado['Tamaño'].min() < (2e-3):
            barras_textura(x_left=(0.00),x_right=(0.002),y=101,label='Arcillas',df=df_transformado)
    
    y_max = ax.get_ylim()[1]  # Esto devuelve el límite superior actual del eje Y
    # Convertir a escala log10
    ref_values = {k: (v) for k, v in REFERENCIAS.items()}
    if not zoom:
        ax.annotate("", xytext=((3e-2), 5), xy=((0.075), 5),
            arrowprops=dict(arrowstyle="<-"))
        ax.text((4.8e-2), 6, 'Finos', ha='center', va='bottom', rotation='horizontal',alpha=0.8,
                        bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    if df_transformado['Tamaño'].min()< (2e-3): # hay arcillas

        # Dibujar líneas de referencia (solo si están dentro del rango visible)
        for nombre, xpos in ref_values.items():
            if xlim is None or (xlim[0] <= xpos <= xlim[1]):
                ax.axvline(x=xpos, color='#121212', linestyle='--', alpha=0.7)
                ax.text(xpos, y_max+0.2, nombre, ha='center', va='bottom', 
                        bbox=dict(facecolor='white', alpha=0, edgecolor='none'))
    else:
        ref_values = {k: v for k, v in ref_values.items() if k != '0,002 mm'}
        # Dibujar líneas de referencia (solo si están dentro del rango visible)
        for nombre, xpos in ref_values.items():
            if xlim is None or (xlim[0] <= xpos <= xlim[1]):
                ax.axvline(x=xpos, color='#121212', linestyle='--', alpha=0.7)
                ax.text(xpos, y_max+0.2, nombre, ha='center', va='bottom', 
                        bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    def log_tick_formatter(val, pos):
        if val == 0:
            return "0"
        elif val < 1:
            return f"{val:.3f}"
        else:
            return f"{val:.1f}"

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

    ax.tick_params(axis='x', which='major', labelsize=10)

    # Locator para líneas menores (entre 1e-2 y 1e-1, por ejemplo: 2e-2, 3e-2, ...)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
    if not zoom:
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Ahora sí: aplicar estilos distintos a major y minor gridlines
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.6, color='black')   # Líneas principales negras
    ax.grid(True, which='minor', linestyle='--', linewidth=0.7, alpha=0.6, color='gray') # Líneas menores gris

    # Personalización del gráfico
    ax.set_title(titulo if titulo else 'Curva Granulométrica', fontsize=16, pad=40)
    ax.set_xlabel('Tamaño (mm)', fontsize=12)
    ax.set_ylabel('% Pasante', fontsize=12)
    # Leyenda principal (fuera del gráfico, a la derecha)
    # Segunda leyenda dentro del gráfico
    if Agrupar_muestras:
        linea_grupo1 = Line2D([0], [0], color='black', lw=2, linestyle='-', label=nombre_grupos[0])
        linea_grupo2 = Line2D([0], [0], color='black', lw=2, linestyle='--', label=nombre_grupos[1])
        leg2 =ax.legend(handles=[linea_grupo1, linea_grupo2], title='', loc='lower right')
        ax.add_artist(leg2)  # clave: mantener esta leyenda
    
    leg1 = ax.legend(title='Muestras', bbox_to_anchor=(1.36, 1.0), loc='upper right')
    # Ampliar margen derecho para que se vea la leyenda externa
    fig.subplots_adjust(right=0.75)
    
    if not zoom:
        ax.set_ylim(0, 106)
    
    # Aplicar límites de zoom si se especifican
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    
    if xlim is None:
        ax.set_xlim(df_transformado['Tamaño'].min()*0.70, 
                    df_transformado['Tamaño'].max()*1.04)
    
    return fig

def search_amount_size(df, muestra, size_search,digits=3):
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
    value = df_filtered[np.isclose(df_filtered['Tamaño'], size_search)]['% Pasante'].values[0]
    return np.round(value, digits)  # Redondear a 3 decimales

def interpolar_valor(df, x,muestra_example,digits=3):
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
    return np.round(y,digits)

def interpolar_percentil(df, percentil,muestra_example,digits=3):
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
        return np.round(x,digits)
    
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
def texture_preprocess(df,muestra,LL=None, PI=None,digits=3):
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
        D10 = np.round(10**D10,digits)
        Result.D10=D10
    D30 = append_error(interpolar_percentil,
                           'No se pudo interpolar D30 debido a falta de datos.',
                           Result,df=df, percentil=30, muestra_example=muestra)
    if D30 is not None:
        D30 = np.round(10**D30,digits)
        Result.D30=D30
    D60 = append_error(interpolar_percentil,
                           'No se pudo interpolar D60 debido a falta de datos.',
                           Result,df=df, percentil=60, muestra_example=muestra)
    if D60 is not None:
        D60 = 10**D60
        Result.D60=np.round(D60,digits)
    
    if D60 is not None and D10 is not None:
        Result.Cu= np.round((D60/D10),digits) if D10 != 0 else None # Calcular el Coeficiente de uniformidad
        Result.Cc= np.round((D30**2/(D60*D10)),digits) # Calcular el Coeficiente de gradación


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