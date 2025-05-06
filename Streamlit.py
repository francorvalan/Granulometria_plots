import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from scipy.interpolate import PchipInterpolator
from io import BytesIO
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# Configuración de la página
st.set_page_config(page_title="Análisis Granulométrico", layout="wide")

# Título de la aplicación
st.title("📊 Análisis de Curvas Granulométricas")
st.markdown("""
Suba su archivo Excel con datos granulométricos y personalice la visualización.
La aplicación graficará automáticamente las curvas en escala logarítmica.
""")


# Función para crear el gráfico con zoom mejorado
def crear_grafico(df, columna_tamaño, muestras_seleccionadas, colores, xlim=None, mostrar_puntos=False,
                  zoom=False,titulo=None,Agrupar_muestras=False,grupo2=None,nombre_grupos=None):
    # Preparamos el dataframe transformado
    df_transformado = df.melt(id_vars=[columna_tamaño], var_name='Muestra', value_name='% Pasante')
    df_transformado = df_transformado.dropna()
    
    # Calculamos log10 del tamaño
    df_transformado['log_Tamaño'] = np.log10(df_transformado[columna_tamaño])
    df_transformado = df_transformado.dropna()
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
            
            # Crear función de interpolación
            f = PchipInterpolator(x, y)
            
            # Definir rango para la interpolación
            x_min = x.min() if xlim is None else max(x.min(), xlim[0])
            x_max = x.max() if xlim is None else min(x.max(), xlim[1])
            
            # Crear puntos para curva suavizada (200 puntos en el rango visible)
            x_nuevo = np.linspace(x_min, x_max, 20000)
            y_nuevo = f(x_nuevo)
            
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
        print('hay arcillas')
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


# Barra lateral para carga de archivos
with st.sidebar:
    st.header("Carga y Configuración")
    archivo_subido = st.file_uploader("Subir archivo Excel", type=['xlsx', 'xls'],
                                      help="Arrastra y suelta tu archivo o haz clic para seleccionarlo"  # Texto de ayuda
                                      )
    
    if archivo_subido is not None:
        try:
            df = pd.read_excel(archivo_subido)
            # Asumimos que la primera columna es 'Tamaño' y las demás son muestras
            columna_tamaño = df.columns[0]
            columnas_muestras = df.columns[1:]
            df_transformado = df.melt(id_vars=[columna_tamaño], var_name='Muestra', value_name='% Pasante')
            df_transformado = df_transformado.dropna()
            df_transformado['log_Tamaño'] = np.log10(df_transformado[columna_tamaño])
            df_transformado = df_transformado.dropna()
            
            # Obtenemos el rango completo de tamaños (en escala log10)
            min_log = df_transformado['Tamaño'].min()
            max_log = df_transformado['Tamaño'].max()
            
            st.success("¡Archivo cargado correctamente!")

                    # Opción para mostrar puntos
            mostrar_puntos = st.checkbox("Mostrar puntos de datos", value=False)
            
            # Widget para seleccionar rango de zoom (en escala logarítmica)
            st.header("Configuración de Zoom")
            zoom_habilitado = st.checkbox("Activar vista con zoom")

            if zoom_habilitado:

                x_min_zoom= st.number_input("Rango inferior", value=min_log, placeholder=min_log,step=0.001,format="%.3f")   
                
                x_max_zoom= st.number_input("Rango superior", value=max_log, placeholder=max_log,step=0.001,format="%.3f")  
            
            Agrupar_muestras = st.checkbox("Agrupar muestras") 

                
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
    else:
        st.info("Por favor suba un archivo Excel")
        st.stop()

    
# Contenido principal
if archivo_subido is not None:
    # Preparamos el dataframe transformado para obtener el rango completo
    columna_tamaño = df.columns[0]
    columnas_muestras = df.columns[1:]
    df_transformado = df.melt(id_vars=[columna_tamaño], var_name='Muestra', value_name='% Pasante')
    df_transformado = df_transformado.dropna()
    df_transformado['log_Tamaño'] = np.log10(df_transformado[columna_tamaño])
    
    # Obtenemos el rango completo de tamaños (en escala log10)
    min_log = df_transformado['log_Tamaño'].min()
    max_log = df_transformado['log_Tamaño'].max()
    
    # Obtenemos las muestras únicas
    muestras_unicas = df_transformado['Muestra'].unique()
    
    # Creamos dos columnas para controles y gráfico
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # ... (código existente)
        st.subheader("Título del gráfico")
        titulo_grafico = st.text_input(
            "Título del gráfico",
            value="Granulometría "
        )

        st.subheader("Selección de Curvas")
        
        # Selección múltiple de muestras a mostrar
        muestras_seleccionadas = st.multiselect(
            "Seleccione las muestras a graficar",
            options=muestras_unicas,
            default=muestras_unicas
        )
        
        # Selección de colores para cada muestra
        st.subheader("Personalización de Colores")
        colores = {}
        # Paleta de colores por defecto
        palette = plt.colormaps.get_cmap('tab10').resampled(len(muestras_seleccionadas))
        
        option = st.selectbox(
            "Seleccionar la muestra para personalizar el color",
            options=muestras_seleccionadas)

        

        for i, muestra in enumerate(muestras_seleccionadas):
            color_default = palette(i)
            color_default_hex = '#%02x%02x%02x' % (
                int(color_default[0]*255),
                int(color_default[1]*255),
                int(color_default[2]*255)
            )
            colores[muestra] = color_default_hex

        colores[option] = st.color_picker(
        f"Color para {option}",
        value=colores[option],
        )
       
        Grupo2 = None
        Nombre_grupos = None
        if Agrupar_muestras:
            st.subheader("Nombres de los grupos")
            Nombre_G1 = st.text_input(
                "Grupo 1",
                value="Grupo 1"
            )

            Nombre_G2 = st.text_input(
                "Grupo 2",
                value="Grupo 2"
            )

            Grupo2 = st.multiselect(
                Nombre_G2,
                muestras_seleccionadas,
                default=None,
            )
            Nombre_grupos=[Nombre_G1,Nombre_G2]
    
    with col2:
        st.header("Gráfico Principal")
        
        # Crear y mostrar el gráfico principal
        fig_principal = crear_grafico(df, columna_tamaño, muestras_seleccionadas, colores, None, 
                                      mostrar_puntos,zoom=False,titulo=titulo_grafico,
                                      Agrupar_muestras=Agrupar_muestras,grupo2=Grupo2,nombre_grupos=Nombre_grupos)
        st.pyplot(fig_principal)
        
        # Botón para descargar el gráfico principal
        buf = BytesIO()
        fig_principal.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="Descargar gráfico principal",
            data=buf.getvalue(),
            file_name=f'{titulo_grafico}_principal.png',
            mime="image/png"
        )
        
        # Mostrar gráfico con zoom si está habilitado
        if zoom_habilitado:
            st.header("Vista con Zoom")
            fig_zoom = crear_grafico(df, columna_tamaño, muestras_seleccionadas, colores, 
                                   (x_min_zoom, x_max_zoom), mostrar_puntos,zoom=zoom_habilitado,titulo=titulo_grafico,
                                   Agrupar_muestras=Agrupar_muestras,grupo2=Grupo2,nombre_grupos=Nombre_grupos)
            st.pyplot(fig_zoom)
            
            # Botón para descargar el gráfico con zoom
            buf_zoom = BytesIO()
            fig_zoom.savefig(buf_zoom, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                label="Descargar gráfico con zoom",
                data=buf_zoom.getvalue(),
                file_name=f'{titulo_grafico}_detalle.png',
                mime="image/png"
            )
        
        # Mostrar datos originales
        if st.checkbox("Mostrar datos originales"):
            st.dataframe(df)