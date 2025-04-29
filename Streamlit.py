import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from io import BytesIO


# Configuración de la página
st.set_page_config(page_title="Análisis Granulométrico", layout="wide")

# Título de la aplicación
st.title("📊 Análisis de Curvas Granulométricas")
st.markdown("""
Suba su archivo Excel con datos granulométricos y personalice la visualización.
La aplicación graficará automáticamente las curvas en escala logarítmica.
""")


# Función para crear el gráfico con zoom mejorado
def crear_grafico(df, columna_tamaño, muestras_seleccionadas, colores, xlim=None, mostrar_puntos=False,zoom=False,titulo=None):
    # Preparamos el dataframe transformado
    df_transformado = df.melt(id_vars=[columna_tamaño], var_name='Muestra', value_name='% Pasante')
    df_transformado = df_transformado.dropna()
    
    # Calculamos log10 del tamaño
    df_transformado['log_Tamaño'] = np.log10(df_transformado[columna_tamaño])
    
    # Crear la figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Líneas verticales de referencia (en mm)
    REFERENCIAS = {
        'Arcillas': 2e-3,  # 0.002 mm
        'Limos': 8e-2,    # 0.08 mm
        'Arenas': 5e0     # 5 mm
    }

    # Graficar cada muestra seleccionada
    for muestra in muestras_seleccionadas:
        subset = df_transformado[df_transformado['Muestra'] == muestra]
        if not subset.empty:
            x = subset['log_Tamaño'].values
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
            x_nuevo = np.linspace(x_min, x_max, 200)
            y_nuevo = f(x_nuevo)
            
            # Graficar la curva con el color seleccionado
            ax.plot(x_nuevo, y_nuevo, label=muestra, 
                    color=colores[muestra], linewidth=2)
            
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
    
    y_max = ax.get_ylim()[1]  # Esto devuelve el límite superior actual del eje Y
    # Convertir a escala log10
    ref_values = {k: np.log10(v) for k, v in REFERENCIAS.items()}
    
    # Dibujar líneas de referencia (solo si están dentro del rango visible)
    for nombre, xpos in ref_values.items():
        if xlim is None or (xlim[0] <= xpos <= xlim[1]):
            ax.axvline(x=xpos, color='gray', linestyle='--', alpha=0.7)
            ax.text(xpos, y_max+0.2, nombre, ha='center', va='bottom', 
                    bbox=dict(facecolor='white', alpha=0, edgecolor='none'))
    
    # Personalización del gráfico
    ax.set_title(titulo if titulo else 'Curva Granulométrica', fontsize=16, pad=20)
    ax.set_xlabel('log10(Tamaño) (mm)', fontsize=12)
    ax.set_ylabel('% Pasante', fontsize=12)
    ax.legend(title='Muestras', bbox_to_anchor=(1.05, 1), loc='upper left')
    if not zoom:
        ax.set_ylim(0, 102)
    
    # Aplicar límites de zoom si se especifican
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
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
            
            # Obtenemos el rango completo de tamaños (en escala log10)
            min_log = df_transformado['log_Tamaño'].min()
            max_log = df_transformado['log_Tamaño'].max()
            
            st.success("¡Archivo cargado correctamente!")

                    # Opción para mostrar puntos
            mostrar_puntos = st.checkbox("Mostrar puntos de datos", value=False)
            
            # Widget para seleccionar rango de zoom (en escala logarítmica)
            st.header("Configuración de Zoom")
            zoom_habilitado = st.checkbox("Activar vista con zoom")
            
            if zoom_habilitado:
                # Creamos un slider de rango en escala logarítmica
                rango_log = st.slider(
                    "Seleccione el rango de log10(Tamaño) para el zoom",
                    min_value=float(min_log),
                    max_value=float(max_log),
                    value=(float(min_log), float(max_log)),
                    step=0.01
                )
                
                x_min_zoom = rango_log[0]
                x_max_zoom = rango_log[1]
                
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
        palette = plt.cm.get_cmap('tab10', len(muestras_seleccionadas))
        
        for i, muestra in enumerate(muestras_seleccionadas):
            color_default = palette(i)
            color_default_hex = '#%02x%02x%02x' % (
                int(color_default[0]*255),
                int(color_default[1]*255),
                int(color_default[2]*255)
            )
            colores[muestra] = st.color_picker(
                f"Color para {muestra}",
                value=color_default_hex
            )
        

    
    with col2:
        st.header("Gráfico Principal")
        
        # Crear y mostrar el gráfico principal
        fig_principal = crear_grafico(df, columna_tamaño, muestras_seleccionadas, colores, None, 
                                      mostrar_puntos,zoom=False,titulo=titulo_grafico)
        st.pyplot(fig_principal)
        
        # Botón para descargar el gráfico principal
        buf = BytesIO()
        fig_principal.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="Descargar gráfico principal",
            data=buf.getvalue(),
            file_name="grafico_granulometrico_principal.png",
            mime="image/png"
        )
        
        # Mostrar gráfico con zoom si está habilitado
        if zoom_habilitado:
            st.header("Vista con Zoom")
            fig_zoom = crear_grafico(df, columna_tamaño, muestras_seleccionadas, colores, 
                                   (x_min_zoom, x_max_zoom), mostrar_puntos,zoom=zoom_habilitado,titulo=titulo_grafico)
            st.pyplot(fig_zoom)
            
            # Botón para descargar el gráfico con zoom
            buf_zoom = BytesIO()
            fig_zoom.savefig(buf_zoom, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                label="Descargar gráfico con zoom",
                data=buf_zoom.getvalue(),
                file_name="grafico_granulometrico_zoom.png",
                mime="image/png"
            )
        
        # Mostrar datos originales
        if st.checkbox("Mostrar datos originales"):
            st.dataframe(df)