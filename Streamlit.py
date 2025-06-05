import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from scipy.interpolate import PchipInterpolator
from io import BytesIO
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import functions as fn
from functools import reduce
from openpyxl import load_workbook
from openpyxl.styles import Font, Border, Side

st.set_page_config(page_title="Análisis Granulométrico", layout="wide")
# Inyectar fuente Roboto
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* Aplicar Roboto solo a títulos y texto principal */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stText, .stTitle, .stHeader {
        font-family: 'Roboto', sans-serif !important;
    }
            
    section[data-testid="stSidebar"] {
    background-color: #002f42; /* gris claro, podés cambiarlo */
    }

    /* Fondo del área principal */
    div[data-testid="stAppViewContainer"] > main {
        background-color: #002f42; /* blanco, podés personalizar */
    }

    /* Opcional: cambiar color de fondo general del body */
    body {
        background-color: #002f42;
    }

            
    div[data-testid="stFileUploader"] {
    background-color: #002f42; /* celeste claro */
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #002f42;
    }
    </style>
    """, unsafe_allow_html=True)


df_transformado = None
# Función para crear el gráfico con zoom mejorado
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
            
            # Crear función de interpolación
            f = PchipInterpolator(x, y)
            
            # Definir rango para la interpolación
            x_min = x.min() if xlim is None else max(x.min(), xlim[0])
            x_max = x.max() if xlim is None else min(x.max(), xlim[1])
            
            # Crear puntos para curva suavizada (200 puntos en el rango visible)
            x_nuevo = np.linspace(x_min, x_max, 20000)
            y_nuevo = f(x_nuevo)
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


#st.image("./Logo/ausenco-logo.png", width=450) 
st.logo(
    "./Logo/ausenco-logo.png",
    link="https://www.ausenco.com/",
    #icon_image="./Logo/ausenco-logo.png",
)
tabs = st.tabs(["Aplicación", "Manual de Usuario"])

with tabs[0]:
    # Configuración de la página
    #
    
    # Título de la aplicación
    st.title("📊 Análisis de Curvas Granulométricas")
    st.markdown("""
    Suba su archivo Excel con datos granulométricos y personalice la visualización.
    La aplicación graficará automáticamente las curvas en escala logarítmica.
    """)
    # Barra lateral para carga de archivos
    with st.sidebar:
        st.header("Carga y Configuración")
        archivo_subido = st.file_uploader("Subir archivo Excel", type=['xlsx', 'xls'],
                                        help="Arrastra y suelta tu archivo o haz clic para seleccionarlo"  # Texto de ayuda
                                        )
        print('archivo_subido',archivo_subido)
        if archivo_subido is not None:
            try:
                df = pd.read_excel(archivo_subido)
                df_transformado,df_plasticidad = fn.df_process(df)
                # Asumimos que la primera columna es 'Tamaño' y las demás son muestras
                columna_tamaño = 'Tamaño'
                columnas_muestras = df.columns[1:]
                df_transformado = df_transformado.dropna()
                
                # Obtenemos el rango completo de tamaños (en escala log10)
                min_log = df_transformado['Tamaño'].min()
                max_log = df_transformado['Tamaño'].max()
                
                st.success("¡Archivo cargado correctamente!")

                        # Opción para mostrar puntos
                mostrar_puntos = st.checkbox("Mostrar puntos de datos", value=True)
                
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
            #st.stop()
        
    # Contenido principal
    if df_transformado is not None and df_plasticidad is not None:
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
            fig_principal = crear_grafico(df_transformado, columna_tamaño, muestras_seleccionadas, colores, None, 
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
                fig_zoom = crear_grafico(df_transformado, columna_tamaño, muestras_seleccionadas, colores, 
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
            
            dfs = {}  # Diccionario para guardar los dataframes por muestra
            for muestra in muestras_seleccionadas:
                IP_muestra = df_plasticidad[(df_plasticidad['Muestra'] == muestra) & (df_plasticidad['Nombre'] == 'IP')]['Limite'].values[0]
                LL_muestra =  df_plasticidad[(df_plasticidad['Muestra'] == muestra) & (df_plasticidad['Nombre'] == 'LL')]['Limite'].values[0]
                #print(f'Muestra: {muestra}, IP: {IP_muestra}, LL: {LL_muestra}')
                Sample_processed = fn.texture_preprocess(df=df_transformado,muestra=muestra,LL=LL_muestra, PI=IP_muestra)
                if Sample_processed.Finos is not None:
                    if  Sample_processed.Finos > 50:
                        fn.USCS_finos(Sample_processed)
                    else:
                        fn.USCS_granular(Sample_processed)
                else:
                    Sample_processed.Errores = '\n'.join(Sample_processed.Errores) if Sample_processed.Errores else None
                df_m = fn.to_dataframe(Sample_processed)
                dfs[muestra] = df_m.rename(columns={df_m.columns[1]: muestra})



            # Convertir los dicts a lista y hacer merge secuencial por "Variable"
            df_final = reduce(lambda left, right: pd.merge(left, right, on='Variable', how='outer'), dfs.values())

            unidades = {
                'D10': 'D10 (mm)',
                'D30': 'D30 (mm)',
                'D60': 'D60 (mm)',
                'Arcillas': 'Arcillas (%)',
                'Arenas': 'Arenas (%)',
                'Gravas': 'Gravas (%)',
                'Finos': 'Finos (%)',
                'Limos': 'Limos (%)'
            }
            df_final['Variable'] = df_final['Variable'].replace(unidades)

            # Mostrar datos originales
            if st.checkbox("Mostrar datos originales",value=True):
                st.dataframe(df_final)
            # Botón para descargar los datos procesados
            buf_datos = BytesIO()
            df_final.to_excel(buf_datos, index=False, engine='openpyxl')

            # 2. Aplicar formato con openpyxl
            buf_datos.seek(0)
            wb = load_workbook(buf_datos)
            ws = wb.active

            # Definir formato
            negrita = Font(bold=True)
            borde = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

            for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
                for cell in row:
                    cell.border = borde
                    if cell.column == 1:  # primera columna
                        cell.font = negrita

            # 3. Guardar de nuevo en buffer
            buf_datos = BytesIO()
            wb.save(buf_datos)
            buf_datos.seek(0)

            # 4. Botón de descarga en Streamlit
            st.download_button(
                label="Descargar datos procesados",
                data=buf_datos,
                file_name=f'{titulo_grafico}_Clasificación_USCS.xlsx',
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("Por favor, cargue un archivo Excel para comenzar.")


with tabs[1]:
    st.header("Manual de Usuario 📘")
    
    st.markdown("""
    ### Cómo usar la aplicación:
    1. Cargue un archivo de datos granulométricos y de plasticidad en el formato indicado (Abajo se presenta un ejemplo del formato de tabla requerida la cual puede ser descargada).
    2. Seleccione las muestras a procesar.
    3. Personalice los colores de las curvas y el título del gráfico.
    4. Active la opción de zoom si desea enfocar un rango específico de tamaños.
    5. Agrupe las muestras si es necesario y defina el nombre de los grupos.   
    6. Visualice y Descargue los resultados de clasificación USCS.
    
    
    > ⚠️ **Notas importantes**:

    - Ante la falta de datos, dejar la plantilla sin valores en las celdas correspondientes.
    - Asegúrese de que los nombres de las columnas coincidan con el formato del modelo.
    - Ante falta de datos granulométricos exactos (Arcillas, Limos y Arenas), se realizarán **interpolaciones lineales** cuando sea posible.
    - La aplicación **no valida los datos**, por lo que es responsabilidad del usuario asegurarse del formato y contenido correcto.
    - Si faltan los datos de **LL y/o IP**, se asume **plasticidad baja** (bajo la línea "A") para suelos finos, y clasificación **ML u OL** para suelos gruesos.
    """)
    
    # Ejemplo de DataFrame modelo

    df_modelo = pd.DataFrame({
        'Nombre': ['LL', 'IP', 'Tamaño', 76.2, 63.5, 50.8, 38.1, 25.400000000000002,
       19.05, 12.700000000000001, 9.525, 4.75, 2, 1.651, 0.833, 0.6,
       0.425, 0.42, 0.3, 0.246, 0.212, 0.147, 0.104, 0.074, 0.06812,
       0.056229999999999995, 0.053, 0.04641, 0.043000000000000003, 0.038],
        'BH-01': [20, 6, '% Pasante', 100, 74.9009900990099, np.nan, 69.4539603960396,
       61.09653465346535, 57.67475247524752, 47.488613861386135,
       43.317821782178214, 36.413861386138606, 30.931683168316823, np.nan,
       np.nan, np.nan, 13.696534653465335, np.nan, np.nan, 12.09059405940593, np.nan,
       11.826732673267315, np.nan, 11.763366336633652, np.nan, np.nan,
       11.2549504950495, np.nan, 10.919801980198017, 10.669306930693068],
       'BH-02':[ np.nan, np.nan, '% Pasante', 100, 92.66483516483517,np.nan,
       92.66483516483517, 78.72417582417583, 70.86923076923077,
       60.52582417582418, 56.354395604395606, 47.72857142857143,
       38.925274725274726,np.nan,np.nan,np.nan, 15.95054945054946,np.nan,np.nan,
       13.280219780219795,np.nan, 12.176923076923089,np.nan,
       11.760439560439579,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
       'BH-03':[60, 40, '% Pasante', 100, 66.85140073081608,np.nan,
       48.745432399512794, 43.30085261875762, 40.657734470158346,
       36.12058465286236, 34.79293544457978, 31.997563946406828,
       28.440925700365412,np.nan,np.nan,np.nan, 16.24847746650427,np.nan,np.nan,
       15.444579780755177,np.nan, 15.310596833130333,np.nan, 15.2557856272838,
      np.nan,np.nan, 14.168392204628503,np.nan, 13.873672350791722,
       13.38581607795372],
       'BH-04':[50, 10, '% Pasante', 100, 100, 60.91475409836066,
       52.299672131147545, 35.24196721311476, 29.595737704918037,
       24.031475409836077, 22.713770491803288, 17.250819672131158,np.nan,
       14.129508196721318, 9.968852459016404, 7.695409836065579,
       5.587213114754107,np.nan, 3.7613114754098405,np.nan,
       2.6459016393442596, 1.9409836065573813, 1.4144262295081944,
       1.119016393442621,np.nan,np.nan, 0.9245901639344254,np.nan,
       0.8150819672131178, 0.7534426229508284],
       'BH-05':[np.nan,np.nan, '% Pasante', 100, 100, 53.96971214017522,
       49.38585732165207, 31.66070087609512, 28.154317897371712,
       22.731664580725905, 20.262828535669584, 15.163829787234036,np.nan,
       11.643554443053816, 7.223153942428027, 4.988485607008755,
       3.6036295369211473,np.nan, 2.6455569461827224,np.nan,
       2.0386733416770966, 1.6822277847309124, 1.4708385481852275,
       1.3239048811013703,np.nan,np.nan, 1.252065081351688,np.nan,
       1.2138923654568146, 1.1936170212765944],
       'BH-06':[30, 50, '% Pasante', 100, 100, 50.44140625, 45.7359375,
       38.89015625, 36.247499999999995, 33.0371875, 31.078437500000007,
       27.79468750000001,np.nan, 23.001875000000013, 16.02171875000002,
       12.579375000000013, 10.378437500000018,np.nan, 8.384218750000016,
      np.nan, 7.297187500000021, 6.289531250000024, 5.404843750000026,
       4.666250000000019,np.nan,np.nan, 4.3720312500000205,np.nan,
       4.223281250000014, 4.110156250000017]
    })

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_modelo.to_excel(writer, index=False, sheet_name='Ejemplo')

    processed_data = output.getvalue()

    # Mostrar tabla y botón
    st.subheader("Ejemplo de formato de datos requerido:")
    st.dataframe(df_modelo)

    st.download_button(
        label="📥 Descargar formato ejemplo (.xlsx)",
        data=processed_data,
        file_name='formato_ejemplo.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ) 

st.markdown("---")
st.markdown("📩 Contacto: [Francisco.Corvalan@ausenco.com](mailto:Francisco.Corvalan@ausenco.com)")