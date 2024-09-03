#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# Ruta del archivo Excel
file_path = 'C:/Users/CL160369868/OneDrive - Enel Spa/Documentos compartidos - Latam Control Room/09.- KPI/Archivos Base/Session List/SessionList_2024.xlsx'

# Leer el archivo Excel y convertirlo en un DataFrame
df = pd.read_excel(file_path)

# Mostrar las primeras filas del DataFrame para verificar la importación
print(df.head())


# In[ ]:


# Ruta del archivo Excel
file_path = 'C:/Users/CL160369868/OneDrive - Enel Spa/Documentos compartidos - Latam Control Room/09.- KPI/Archivos Base/EFP Mensual/2407.xls'

# Leer el archivo Excel y convertirlo en un DataFrame usando xlrd
dfst = pd.read_excel(file_path, engine='xlrd')

# Mostrar las primeras filas para verificar la importación
print(dfst.head())


# In[ ]:


pip install xlrd


# In[ ]:





# # Revision del DF

# In[ ]:


df.columns


# In[ ]:


df.columns.tolist()


# In[ ]:


df['Session start'] = pd.to_datetime(df['Session start'])
df = df[df['Session start'].dt.year == 2024]


# In[ ]:


import plotly.express as px

#end_of_charge_counts = df['End of charge reason'].value_counts().reset_index()
#end_of_charge_counts.columns = ['End of charge reason', 'count']  # Renombrar columnas
end_of_charge_counts = df['End of charge reason'].value_counts(normalize=True).reset_index()
end_of_charge_counts.columns = ['End of charge reason', 'percentage']  # Renombrar columnas

# Convertir a porcentaje (opcional, si prefieres en lugar de proporciones)
end_of_charge_counts['percentage'] *= 100


fig = px.treemap(end_of_charge_counts,
                 path=['End of charge reason'],
                 values='percentage',
                 title='Treemap of End of charge')
fig.data[0].texttemplate = "%{label}<br>%{value:.2f}%"

# Adjust margins for legend placement and control legend size with 'itemsizing'
fig.update_layout(
    margin=dict(t=50, l=250, r=20, b=0),  # Adjust margins for legend
    title_font_size=24,
    font=dict(size=14),
    showlegend=True,
    width=800,
    height=600,
    legend=dict(
        title='Razón de fin de carga',
        orientation='v',
        yanchor="top",
        y=0.95,
        xanchor="right",
        x=0.95,
        bordercolor="Black",
        borderwidth=1,

        # Adjust legend size (optional)
        itemsizing='constant',  # Constant size for legend items
        font=dict(size=12, color='darkblue')
    ),
    clickmode='event+select'
)

fig.show()
fig.write_html("Estados.html")


# In[ ]:





# In[ ]:





# # Razones de falla en inicio de carga 2024

# In[ ]:


df['Energy consumption'].sum()/1000000


# In[ ]:


FaultRecharge= df[df['Energy consumption']<=1000]


# In[ ]:


len(FaultRecharge)/len(df)


# In[ ]:


df['Session start'].info()


# In[ ]:


end_of_charge_counts = FaultRecharge['End of charge reason'].value_counts().reset_index()
end_of_charge_counts.columns = ['End of charge reason', 'count']  # Renombrar columnas
figSSFY = px.pie(end_of_charge_counts, 
            width=800,
            height=1000,
            values='count', 
            names='End of charge reason', 
            title='Distribución de razones de fin de carga')

figSSFY.show()
figSSFY.write_html("Detenciones.html")
print(len(FaultRecharge))


# In[ ]:


from datetime import datetime
from dateutil.relativedelta import relativedelta

# Obtener la fecha actual
fecha_actual = datetime.now()
mes_actual = datetime.now().strftime('%m')

# Restar un mes para obtener el mes anterior
mes_anterior = (fecha_actual - relativedelta(months=1)).strftime('%m')

df['Session start'] = pd.to_datetime(df['Session start'])
dfM=df[(df['Session start'] >= '2024-' + mes_anterior + '-01') & 
    (df['Session start'] < '2024-' + mes_actual + '-01')]
# Filtrar el DataFrame
FaultRechargeMes = dfM[dfM['Energy consumption'] <= 1000]
end_of_charge_counts = FaultRechargeMes['End of charge reason'].value_counts().reset_index()
end_of_charge_counts.columns = ['End of charge reason', 'count']  # Renombrar columnas
figSSFM = px.pie(end_of_charge_counts, 
            width=800,
            height=800,
            values='count', 
            names='End of charge reason', 
            title='Distribución de razones de fin de carga')

print(len(FaultRechargeMes))
figSSFM.show()
figSSFM.write_html("Falla Inicio Ultimo mes.html")


# In[ ]:


mes_actual = datetime.now().strftime('%m')


# In[ ]:


mes_actual


# In[ ]:


from dateutil.relativedelta import relativedelta

# Obtener la fecha actual
fecha_actual = datetime.now()

# Restar un mes para obtener el mes anterior
mes_anterior = (fecha_actual - relativedelta(months=1)).strftime('%m')


# In[ ]:


mes_anterior


# In[ ]:


print('2024-'+mes_actual+'-01')
      


# # Analisis de Anomalias

# In[ ]:


# Primero, calculamos los cuartiles y el rango intercuartílico (IQR) para detectar anomalías

# Análisis de la duración de las sesiones

duration_millis = dfM['Session duration millis'].dropna()

# Calcular el rango intercuartílico (IQR)
Q1_duration = duration_millis.quantile(0.25)
Q3_duration = duration_millis.quantile(0.75)
IQR_duration = Q3_duration - Q1_duration

# Definir límites para detectar outliers
lower_bound_duration = Q1_duration - 1.5 * IQR_duration
upper_bound_duration = Q3_duration + 1.5 * IQR_duration

# Identificar sesiones con duración anómala
anomalous_durations = dfM[(dfM['Session duration millis'] < lower_bound_duration) | 
                                      (dfM['Session duration millis'] > upper_bound_duration)]

# Análisis del consumo de energía
energy_consumption = dfM['Energy consumption'].dropna()

# Calcular el rango intercuartílico (IQR)
Q1_energy = energy_consumption.quantile(0.25)
Q3_energy = energy_consumption.quantile(0.75)
IQR_energy = Q3_energy - Q1_energy

# Definir límites para detectar outliers
lower_bound_energy = Q1_energy - 1.5 * IQR_energy
upper_bound_energy = Q3_energy + 1.5 * IQR_energy

# Identificar sesiones con consumo de energía anómalo
anomalous_energy = dfM[(dfM['Energy consumption'] < lower_bound_energy) | 
                                   (dfM['Energy consumption'] > upper_bound_energy)]

# Análisis de conexiones y desconexiones
connections = dfM['Number of connection'].dropna()

# Calcular el rango intercuartílico (IQR)
Q1_connections = connections.quantile(0.25)
Q3_connections = connections.quantile(0.75)
IQR_connections = Q3_connections - Q1_connections

# Definir límites para detectar outliers
lower_bound_connections = Q1_connections - 1.5 * IQR_connections
upper_bound_connections = Q3_connections + 1.5 * IQR_connections

# Identificar sesiones con número de conexiones anómalo
anomalous_connections = dfM[(dfM['Number of connection'] < lower_bound_connections) | 
                                        (dfM['Number of connection'] > upper_bound_connections)]

# Resumir los resultados
anomalies_summary = {
    'Anomalous Durations': len(anomalous_durations),
    'Anomalous Energy Consumptions': len(anomalous_energy),
    'Anomalous Connections': len(anomalous_connections)
}

#anomalies_summary, anomalous_durations.head(), anomalous_energy.head(), anomalous_connections.head()

anomalous_durations.to_excel('anomalous_durations.xlsx', index=False)
anomalous_energy.to_excel('anomalous_energy.xlsx', index=False)
anomalous_connections.to_excel('anomalous_connections.xlsx', index=False)

print("Exportación completada. Archivos guardados como 'anomalous_durations.xlsx', 'anomalous_energy.xlsx', 'anomalous_connections.xlsx'.")


# ## Conexiones Anomalas

# In[ ]:


import plotly.graph_objects as go
connections_data = dfM[['Session ID', 'Number of connection']].dropna()

# Marcar las sesiones anómalas
connections_data['Anomalous'] = connections_data['Number of connection'].apply(
    lambda x: 'Anomalous' if (x < lower_bound_connections or x > upper_bound_connections) else 'Normal'
)

# Crear el gráfico de corbata
fig = go.Figure()

# Añadir las conexiones normales
fig.add_trace(go.Box(
    y=connections_data[connections_data['Anomalous'] == 'Normal']['Number of connection'],
    name='Normal',
    marker_color='blue',
    boxmean=True  # Mostrar la media
))

# Añadir las conexiones anómalas
fig.add_trace(go.Box(
    y=connections_data[connections_data['Anomalous'] == 'Anomalous']['Number of connection'],
    name='Anomalous',
    marker_color='red',
    boxmean=True  # Mostrar la media
))

# Configurar el layout
fig.update_layout(
    title='Análisis de Conexiones Anómalas - Gráfico de Corbata',
    yaxis_title='Número de Conexiones',
    xaxis_title='Tipo de Sesión',
    showlegend=False,
    height=800
)

# Mostrar el gráfico
fig.show()


# In[ ]:


#import pandas as pd
#import plotly.express as px

# Supongamos que 'data_first_3000' contiene también la columna 'Serial number'
# Asegúrate de que 'Serial number' esté en el DataFrame original

# Datos para graficar
connections_data = dfM[['Session ID', 'Number of connection', 'Serial number']].dropna()

# Marcar las sesiones anómalas
connections_data['Anomalous'] = connections_data['Number of connection'].apply(
    lambda x: 'Anomalous' if (x < lower_bound_connections or x > upper_bound_connections) else 'Normal'
)
total_anomalous = connections_data[connections_data['Anomalous'] == 'Anomalous'].shape[0]

# Crear el gráfico
figACM = px.scatter(connections_data, x='Session ID', y='Number of connection', color='Anomalous',
                 title='Análisis de Conexiones Anómalas',
                 labels={'Session ID': 'ID de Sesión', 'Number of connection': 'Número de Conexiones'},
                 hover_data=['Serial number'])  # Agregar 'Serial number' al tooltip
figACM.update_layout(height=800)

figACM.add_annotation(
    text=f'Total de Conexiones Anómalas: {total_anomalous}',
    xref="paper", yref="paper",
    x=0.5, y=1.1,  # Posición de la anotación
    showarrow=False,
    font=dict(size=14, color="red")
)

# Mostrar el gráfico
figACM.show()


# In[ ]:


# Definir un umbral para considerar una desconexión como alta
high_disconnection_threshold = 5  # Esto es un ejemplo, puedes ajustarlo según los datos

# Filtrar los datos para incluir solo los eventos con desconexiones altas
high_disconnection_events = anomalous_connections[anomalous_connections['Number of disconnection'] > high_disconnection_threshold]

# Agrupar por 'Serial number' y contar los eventos de desconexiones altas
high_disconnection_counts = high_disconnection_events.groupby('Serial number')['Number of disconnection'].count().reset_index()

# Renombrar la columna para mayor claridad
high_disconnection_counts.columns = ['Serial number', 'High Disconnection Events']

# Crear el mapa de árbol
figACM2 = px.treemap(high_disconnection_counts, path=['Serial number'], values='High Disconnection Events',
                 title='Cargadores con Mayor Cantidad de Eventos de Desconexiones Altas',
                 color='High Disconnection Events', color_continuous_scale='RdYlGn_r')

# Ampliar el tamaño del gráfico
figACM2.update_layout(height=600, width=800)

# Mostrar el gráfico
figACM2.show()


# ## Anomalias de Energia

# In[ ]:


# Asegurarse de que 'Energy consumption' sea numérico
dfM['Energy consumption'] = pd.to_numeric(dfM['Energy consumption'], errors='coerce')

# Eliminar filas con valores nulos en 'Energy consumption'
energy_data = dfM[['Session ID', 'Energy consumption']].dropna()

# Calcular el IQR para 'Energy consumption'
Q1_energy = energy_data['Energy consumption'].quantile(0.25)
Q3_energy = energy_data['Energy consumption'].quantile(0.75)
IQR_energy = Q3_energy - Q1_energy

# Definir límites para detectar outliers
lower_bound_energy = Q1_energy - 1.5 * IQR_energy
upper_bound_energy = Q3_energy + 1.5 * IQR_energy

# Identificar sesiones con consumo de energía anómalo
energy_data['Anomalous'] = energy_data['Energy consumption'].apply(
    lambda x: 'Anomalous' if (x < lower_bound_energy or x > upper_bound_energy) else 'Normal'
)

# Crear el gráfico de dispersión
figAEM = px.scatter(energy_data, x='Session ID', y='Energy consumption', color='Anomalous',
                 title='Consumo de Energía Atípico',
                 labels={'Session ID': 'ID de Sesión', 'Energy consumption': 'Consumo de Energía (kWh)'})

# Ampliar el tamaño del gráfico
figAEM.update_layout(height=800, width=1000)

# Mostrar el gráfico
figAEM.show()


# In[ ]:


# Lista de números de serie a excluir
exclusion_list = [
    "20200907DC003", "20200907DC009", "20200907DCX03",
    "20200907DCX04", "20200907DCX05", "SN2301300046", "SN2310261929","20200907DC006","C6E15BCC21KNYJDYWA"
]

# Filtrar los datos para excluir los seriales de la lista
data_filtered = dfM[~dfM['Serial number'].isin(exclusion_list)]

# Asegurarse de que 'Energy consumption' sea numérico
data_filtered['Energy consumption'] = pd.to_numeric(data_filtered['Energy consumption'], errors='coerce')

# Eliminar filas con valores nulos en 'Energy consumption'
energy_data = data_filtered[['Session ID', 'Energy consumption', 'Serial number']].dropna()

# Calcular el IQR para 'Energy consumption'
Q1_energy = energy_data['Energy consumption'].quantile(0.25)
Q3_energy = energy_data['Energy consumption'].quantile(0.75)
IQR_energy = Q3_energy - Q1_energy

# Definir límites para detectar outliers
lower_bound_energy = Q1_energy - 1.5 * IQR_energy
upper_bound_energy = Q3_energy + 1.5 * IQR_energy

# Identificar sesiones con consumo de energía anómalo
energy_data['Anomalous'] = energy_data['Energy consumption'].apply(
    lambda x: 'Anomalous' if (x < lower_bound_energy or x > upper_bound_energy) else 'Normal'
)

# Crear el gráfico de dispersión
figAEM2 = px.scatter(energy_data, x='Session ID', y='Energy consumption', color='Anomalous',
                 title='Consumo de Energía Atípico',
                 labels={'Session ID': 'ID de Sesión', 'Energy consumption': 'Consumo de Energía (kWh)'},
                 hover_data=['Serial number'])  # Agregar 'Serial number' al tooltip

# Ampliar el tamaño del gráfico
figAEM2.update_layout(height=800, width=1000)

# Mostrar el gráfico
figAEM2.show()


# In[ ]:


anomalous_energy_data = energy_data[energy_data['Anomalous'] == 'Anomalous']
anomalous_summary = anomalous_energy_data.groupby('Serial number').agg({
    'Energy consumption': ['count', 'max']
}).reset_index()
anomalous_summary.columns = ['Serial number', 'Anomalous_Events', 'Max_Anomalous_Energy']
# Crear el gráfico de burbujas
figAEM3 = px.scatter(anomalous_summary, 
                 x='Serial number', 
                 y='Anomalous_Events',
                 size='Max_Anomalous_Energy', 
                 color='Max_Anomalous_Energy',
                 title='Bubble Chart of Anomalous Energy Events by Serial Number',
                 labels={'Serial number': 'Serial Number', 
                         'Anomalous_Events': 'Number of Anomalous Events',
                         'Max_Anomalous_Energy': 'Max Energy Consumption (kWh)'},
                 hover_data={'Max_Anomalous_Energy': ':.2f Wh'})

# Personalizar el diseño para una mejor visualización
figAEM3.update_layout(height=600, width=1000, 
                  xaxis=dict(showgrid=False), 
                  yaxis=dict(showgrid=True),
                  showlegend=True)

# Mostrar el gráfico de burbujas
figAEM3.show()


# In[ ]:


# Assuming the data_filtered has 'Serial number' and 'Energy consumption'

# We are only interested in anomalous energy data
anomalous_energy_data = energy_data[energy_data['Anomalous'] == 'Anomalous']

# Create a BoxPlot
figAEM4 = px.box(anomalous_energy_data, 
             x='Serial number', 
             y='Energy consumption', 
             title='BoxPlot of Anomalous Energy Consumption by Serial Number',
             labels={'Serial number': 'Serial Number', 
                     'Energy consumption': 'Energy Consumption (kWh)'},
             points='all')  # 'points="all"' shows all outliers

# Customize the layout for better visualization
figAEM4.update_layout(height=600, width=1000, 
                  xaxis=dict(showgrid=False), 
                  yaxis=dict(showgrid=True),
                  showlegend=False)

# Show the BoxPlot
figAEM4.show()


# In[ ]:





# ## Tiempo de sesiones anomalos

# In[ ]:


def millis_to_dhms(millis):
    seconds = millis / 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{int(days)}d {int(hours)}h {int(minutes)}m"

# Aplicar la función a la columna de duración en milisegundos

anomalous_durations['Session duration (minutes)'] = anomalous_durations['Session duration millis'] / 60000

# Crear un gráfico de dispersión (scatter plot) para visualizar las duraciones anómalas
figATM = px.scatter(
    anomalous_durations,
    x='Session ID',
    y='Session duration (minutes)',
    hover_data={'Serial number': True, 'Session duration (minutes)': True},
    title='Anomalous Session Durations',
    labels={'Session ID': 'Session ID', 'Session duration (minutes)': 'Duration (minutes)'},
    template='plotly'
)

# Configurar el diseño para mejorar la visualización
figATM.update_layout(height=800, width=1000, 
    xaxis_title="Session ID",
    yaxis_title="Duration (minutes)",
    hovermode="closest",
    showlegend=False
)

# Mostrar el gráfico
figATM.show()


# In[ ]:


# Función para convertir milisegundos a formato "días, horas, minutos"
def millis_to_dhms(millis):
    seconds = millis / 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{int(days)}d {int(hours)}h {int(minutes)}m"

# Supongamos que tienes el DataFrame 'anomalous_durations' ya cargado

# Contar la cantidad de eventos anómalos por número de serie
anomalous_events_count = anomalous_durations.groupby('Serial number').size().reset_index(name='Event count')

# Obtener el tiempo máximo en formato "días, horas, minutos" por número de serie
max_duration = anomalous_durations.groupby('Serial number')['Session duration millis'].max().reset_index()
max_duration['Max duration (dhms)'] = max_duration['Session duration millis'].apply(millis_to_dhms)
max_duration['Max duration (minutes)'] = max_duration['Session duration millis'] / 60000

# Unir ambos resultados en un solo DataFrame
anomalous_series_summary = pd.merge(anomalous_events_count, max_duration, on='Serial number')

# Crear un gráfico de burbujas para mostrar el número de eventos anómalos y la duración máxima
figATM2 = px.scatter(
    anomalous_series_summary,
    x='Serial number',
    y='Max duration (minutes)',
    size='Event count',
    color='Max duration (minutes)',  # Asocia el color al tiempo máximo de duración
    color_continuous_scale='Bluered',
    hover_data={'Serial number': True, 'Event count': True, 'Max duration (dhms)': True},
    title='Anomalous Session Durations by Serial Number',
    labels={'Serial number': 'Serial Number', 'Max duration (minutes)': 'Max Duration (minutes)'},
    template='plotly'
)


# Configurar el diseño para mejorar la visualización
figATM2.update_layout    (height=600, width=1000, 
    xaxis_title="Serial Number",
    yaxis_title="Max Duration (minutes)",
    hovermode="closest"
)

# Mostrar el gráfico
figATM2.show()


# In[ ]:


def millis_to_dhms(millis):
    seconds = millis / 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{int(days)}d {int(hours)}h {int(minutes)}m"

# Supongamos que tienes el DataFrame 'anomalous_durations' ya cargado

# Contar la cantidad de eventos anómalos por número de serie
anomalous_events_count = anomalous_durations.groupby('Serial number').size().reset_index(name='Event count')

# Obtener el tiempo máximo en formato "días, horas, minutos" por número de serie
max_duration = anomalous_durations.groupby('Serial number')['Session duration millis'].max().reset_index()
max_duration['Max duration (dhms)'] = max_duration['Session duration millis'].apply(millis_to_dhms)
max_duration['Max duration (minutes)'] = max_duration['Session duration millis'] / 60000

# Calcular la duración promedio de las sesiones anómalas por número de serie
average_duration = anomalous_durations.groupby('Serial number')['Session duration millis'].mean().reset_index()
average_duration['Average duration (minutes)'] = average_duration['Session duration millis'] / 60000

# Unir la información del conteo de eventos, la duración máxima y la duración promedio en un solo DataFrame
anomalous_series_summary = pd.merge(anomalous_events_count, max_duration, on='Serial number')
anomalous_series_summary = pd.merge(anomalous_series_summary, average_duration, on='Serial number')

# Crear un gráfico de burbujas para mostrar el número de eventos anómalos, la duración máxima y la duración promedio
figATM3 = px.scatter(
    anomalous_series_summary,
    x='Serial number',
    y='Average duration (minutes)',
    size='Event count',
    color='Max duration (minutes)',  # Asocia el color al tiempo máximo de duración
    color_continuous_scale='Bluered',
    hover_data={'Serial number': True, 'Event count': True, 'Max duration (dhms)': True, 'Average duration (minutes)': True},
    title='Anomalous Session Durations by Serial Number',
    labels={'Serial number': 'Serial Number', 'Average duration (minutes)': 'Average Duration (minutes)'},
    template='plotly'
)

# Configurar el diseño para mejorar la visualización
figATM3.update_layout(height=600, width=1000, 
    xaxis_title="Serial Number",
    yaxis_title="Average Duration (minutes)",
    hovermode="closest",
    coloraxis_colorbar=dict(
        title="Max Duration (minutes)"
    )
)

# Mostrar el gráfico
figATM3.show()


# In[ ]:





# In[ ]:





# # Tendencias de Cargas y Sesiones
# ## Vision Anual

# In[ ]:


# Convert the 'Session start' column to datetime format for temporal analysis
data_first_3000 = df.copy()
data_first_3000['Session start'] = pd.to_datetime(data_first_3000['Session start'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Filtrar solo los datos del año 2024
data_first_3000 = data_first_3000[data_first_3000['Session start'].dt.year == 2024]

# Extract date and hour information for analysis
data_first_3000['Date'] = data_first_3000['Session start'].dt.date
data_first_3000['Hour'] = data_first_3000['Session start'].dt.hour

# Group by date to analyze the energy consumption per day
energy_per_day = data_first_3000.groupby('Date')['Energy consumption'].sum().reset_index(name='Total Energy (kWh)')

# Group by hour to analyze the energy consumption per hour
energy_per_hour = data_first_3000.groupby('Hour')['Energy consumption'].sum().reset_index(name='Total Energy (kWh)')

# Plot the energy consumption per day
fig_energy_day = px.line(energy_per_day, x='Date', y='Total Energy (kWh)', title='Total Energy Consumption per Day')

# Plot the energy consumption per hour
fig_energy_hour = px.bar(energy_per_hour, x='Hour', y='Total Energy (kWh)', title='Total Energy Consumption per Hour')

data_first_3000['Week'] = data_first_3000['Session start'].dt.isocalendar().week
last_week = data_first_3000['Week'].max()
energy_per_week = data_first_3000[data_first_3000['Week'] < last_week].groupby('Week')['Energy consumption'].sum().reset_index(name='Total Energy (kWh)')
fig_energy_week = px.line(energy_per_week, x='Week', y='Total Energy (kWh)', title='Total Energy Consumption per Week')
fig_energy_day.show()

fig_energy_week.show()

fig_energy_hour.show()






# Convertir la columna 'Session start' a formato datetime
data_first_3000['Session start'] = pd.to_datetime(data_first_3000['Session start'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Filtrar solo los datos del año 2024 y hacer una copia para evitar la advertencia
data_first_3000 = data_first_3000[data_first_3000['Session start'].dt.year == 2024].copy()

# Extraer la información de fecha y hora
data_first_3000['Date'] = data_first_3000['Session start'].dt.date
data_first_3000['Hour'] = data_first_3000['Session start'].dt.hour

# Agrupar por fecha para analizar el número de sesiones por día
sessions_per_day = data_first_3000.groupby('Date').size().reset_index(name='Number of Sessions')

# Agrupar por hora para analizar el número de sesiones por hora
sessions_per_hour = data_first_3000.groupby('Hour').size().reset_index(name='Number of Sessions')

# Graficar el número de sesiones por día
fig_day = px.line(sessions_per_day, x='Date', y='Number of Sessions', title='Number of Sessions per Day')

# Graficar el número de sesiones por hora
fig_hour = px.bar(sessions_per_hour, x='Hour', y='Number of Sessions', title='Number of Sessions per Hour')

# Mostrar las gráficas
fig_day.show()
fig_hour.show()





ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Convert 'Session start' to datetime if not already done
data_first_3000['Session start'] = pd.to_datetime(data_first_3000['Session start'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Crear la columna 'Week' a partir de la fecha de inicio de la sesión
data_first_3000['Week'] = data_first_3000['Session start'].dt.isocalendar().week
data_first_3000['Day of Week'] = data_first_3000['Session start'].dt.day_name()

# Exclude the last week from the data
max_week = data_first_3000['Week'].max()
filtered_data = data_first_3000[data_first_3000['Week'] < max_week]

# Group by week number and day of the week to count the number of sessions
filtered_heatmap_data = filtered_data.groupby(['Week', 'Day of Week']).size().unstack(fill_value=0)

# Reorder days of the week for better visualization
filtered_heatmap_data = filtered_heatmap_data[ordered_days]

# Create the heatmap using Plotly
fig_filtered_heatmap = go.Figure(data=go.Heatmap(
    z=filtered_heatmap_data.values,
    x=filtered_heatmap_data.columns,
    y=filtered_heatmap_data.index,
    colorscale='rdylgn_r'
))

# Update layout for better readability
fig_filtered_heatmap.update_layout(height=600, width=1000, 
    title='Number of Sessions per Week vs Day of the Week (Excluding Last Week)',
    xaxis_title='Day of the Week',
    yaxis_title='Week Number',
    yaxis=dict(
        dtick=1
    )
)

# Display the heatmap
fig_filtered_heatmap.show()






filtered_data['Date'] = filtered_data['Session start'].dt.strftime('%d/%m/%Y')

# Group by week number and day of the week, summing the energy consumption and keeping the date information
grouped_data = filtered_data.groupby(['Week', 'Day of Week']).agg({
    'Energy consumption': 'sum',
    'Date': 'first'  # Just to get the representative date in that group
}).unstack(fill_value=0)

# Extract the energy consumption and date
energy_consumption = grouped_data['Energy consumption']/1000000
dates = grouped_data['Date']

# Reordenar los días de la semana para que vayan de lunes a domingo

energy_consumption = energy_consumption[ordered_days]
dates = dates[ordered_days]

# Create the heatmap with dates in the hover text
fig_energy_heatmap = go.Figure(data=go.Heatmap(
    z=energy_consumption.values,
    x=energy_consumption.columns,
    y=energy_consumption.index,
    colorscale='rdylgn_r',
    hoverongaps=False,
    text=dates.values,
    hovertemplate="Week: %{y}<br>Day: %{x}<br>Date: %{text}<br>Energy: %{z:.2f} MWh<extra></extra>"
))

# Update layout for better readability
fig_energy_heatmap.update_layout(height=600, width=1000, 
    title='Total Energy Consumption per Week vs Day of the Week (Excluding Last Week)',
    xaxis_title='Day of the Week',
    yaxis_title='Week Number',
    yaxis=dict(
        dtick=1
    )
)

# Display the heatmap
fig_energy_heatmap.show()



# ## Analisis Temporal Mensual

# In[ ]:


# Convert the 'Session start' column to datetime format for temporal analysis
# Convert the 'Session start' column to datetime format for temporal analysis
data_first_3000 = dfM.copy()
data_first_3000['Session start'] = pd.to_datetime(data_first_3000['Session start'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Filtrar solo los datos del año 2024
data_first_3000 = data_first_3000[data_first_3000['Session start'].dt.year == 2024]

# Extract date and hour information for analysis
data_first_3000['Date'] = data_first_3000['Session start'].dt.date
data_first_3000['Hour'] = data_first_3000['Session start'].dt.hour

# Group by date to analyze the energy consumption per day
energy_per_day = data_first_3000.groupby('Date')['Energy consumption'].sum().reset_index(name='Total Energy (kWh)')

# Group by hour to analyze the energy consumption per hour
energy_per_hour = data_first_3000.groupby('Hour')['Energy consumption'].sum().reset_index(name='Total Energy (kWh)')

# Plot the energy consumption per day
fig_energy_day = px.line(energy_per_day, x='Date', y='Total Energy (kWh)', title='Total Energy Consumption per Day')

# Plot the energy consumption per hour
fig_energy_hour = px.bar(energy_per_hour, x='Hour', y='Total Energy (kWh)', title='Total Energy Consumption per Hour')

data_first_3000['Week'] = data_first_3000['Session start'].dt.isocalendar().week
last_week = data_first_3000['Week'].max()
energy_per_week = data_first_3000[data_first_3000['Week'] < last_week].groupby('Week')['Energy consumption'].sum().reset_index(name='Total Energy (kWh)')
fig_energy_week = px.line(energy_per_week, x='Week', y='Total Energy (kWh)', title='Total Energy Consumption per Week')
fig_energy_day.show()

fig_energy_week.show()

fig_energy_hour.show()






# Convertir la columna 'Session start' a formato datetime
data_first_3000['Session start'] = pd.to_datetime(data_first_3000['Session start'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Filtrar solo los datos del año 2024 y hacer una copia para evitar la advertencia
data_first_3000 = data_first_3000[data_first_3000['Session start'].dt.year == 2024].copy()

# Extraer la información de fecha y hora
data_first_3000['Date'] = data_first_3000['Session start'].dt.date
data_first_3000['Hour'] = data_first_3000['Session start'].dt.hour

# Agrupar por fecha para analizar el número de sesiones por día
sessions_per_day = data_first_3000.groupby('Date').size().reset_index(name='Number of Sessions')

# Agrupar por hora para analizar el número de sesiones por hora
sessions_per_hour = data_first_3000.groupby('Hour').size().reset_index(name='Number of Sessions')

# Graficar el número de sesiones por día
fig_day = px.line(sessions_per_day, x='Date', y='Number of Sessions', title='Number of Sessions per Day')

# Graficar el número de sesiones por hora
fig_hour = px.bar(sessions_per_hour, x='Hour', y='Number of Sessions', title='Number of Sessions per Hour')

# Mostrar las gráficas
fig_day.show()
fig_hour.show()





ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# Convert 'Session start' to datetime if not already done
data_first_3000['Session start'] = pd.to_datetime(data_first_3000['Session start'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Crear la columna 'Week' a partir de la fecha de inicio de la sesión
data_first_3000['Week'] = data_first_3000['Session start'].dt.isocalendar().week
data_first_3000['Day of Week'] = data_first_3000['Session start'].dt.day_name()

# Exclude the last week from the data
max_week = data_first_3000['Week'].max()
filtered_data = data_first_3000[data_first_3000['Week'] < max_week]

# Group by week number and day of the week to count the number of sessions
filtered_heatmap_data = filtered_data.groupby(['Week', 'Day of Week']).size().unstack(fill_value=0)

# Reorder days of the week for better visualization
filtered_heatmap_data = filtered_heatmap_data[ordered_days]

# Create the heatmap using Plotly
fig_filtered_heatmap = go.Figure(data=go.Heatmap(
    z=filtered_heatmap_data.values,
    x=filtered_heatmap_data.columns,
    y=filtered_heatmap_data.index,
    colorscale='rdylgn_r'
))

# Update layout for better readability
fig_filtered_heatmap.update_layout(height=600, width=1000, 
    title='Number of Sessions per Week vs Day of the Week (Excluding Last Week)',
    xaxis_title='Day of the Week',
    yaxis_title='Week Number',
    yaxis=dict(
        dtick=1
    )
)

# Display the heatmap
fig_filtered_heatmap.show()






filtered_data['Date'] = filtered_data['Session start'].dt.strftime('%d/%m/%Y')

# Group by week number and day of the week, summing the energy consumption and keeping the date information
grouped_data = filtered_data.groupby(['Week', 'Day of Week']).agg({
    'Energy consumption': 'sum',
    'Date': 'first'  # Just to get the representative date in that group
}).unstack(fill_value=0)

# Extract the energy consumption and date
energy_consumption = grouped_data['Energy consumption']/1000000
dates = grouped_data['Date']

# Reordenar los días de la semana para que vayan de lunes a domingo

energy_consumption = energy_consumption[ordered_days]
dates = dates[ordered_days]

# Create the heatmap with dates in the hover text
fig_energy_heatmap = go.Figure(data=go.Heatmap(
    z=energy_consumption.values,
    x=energy_consumption.columns,
    y=energy_consumption.index,
    colorscale='rdylgn_r',
    hoverongaps=False,
    text=dates.values,
    hovertemplate="Week: %{y}<br>Day: %{x}<br>Date: %{text}<br>Energy: %{z:.2f} MWh<extra></extra>"
))

# Update layout for better readability
fig_energy_heatmap.update_layout(height=600, width=1000, 
    title='Total Energy Consumption per Week vs Day of the Week (Excluding Last Week)',
    xaxis_title='Day of the Week',
    yaxis_title='Week Number',
    yaxis=dict(
        dtick=1
    )
)

# Display the heatmap
fig_energy_heatmap.show()




# In[ ]:





# In[ ]:


# Prepare data for plotting by dropping missing values
# Filtrar solo los datos del año 2024
data_first_3000 = dfM.copy()
data_first_3000 = data_first_3000[data_first_3000['Session start'].dt.year == 2024]



duration_data = data_first_3000['Session duration millis'].dropna()
energy_data = data_first_3000['Energy consumption'].dropna()
connections_data = data_first_3000['Number of connection'].dropna()
MaxPower_Data = pd.to_numeric(data_first_3000['Maximum power'], errors='coerce').dropna()

# Plot distribution of session duration
fig_duration = px.histogram(duration_data, nbins=50, title='Distribución de la Duración de las Sesiones (en milisegundos)')
fig_duration.show()

# Plot distribution of energy consumption
fig_energy = px.histogram(energy_data, nbins=50, title='Distribución del Consumo de Energía (en kWh)')
fig_energy.show()

# Plot distribution of energy consumption
fig_MaxPower = px.histogram(MaxPower_Data, nbins=50, title='Peak de Potencia por sesion (kW)')
fig_MaxPower.show()

# Plot distribution of number of connections
fig_connections = px.histogram(connections_data, nbins=30, title='Distribución del Número de Conexiones por Sesión')
fig_connections.show()


# In[ ]:


import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Limpiar los datos para el análisis de correlación
# Convertir la duración de la sesión a milisegundos
data_first_3000['Session duration millis'] = pd.to_numeric(data_first_3000['Session duration millis'], errors='coerce')

# Seleccionar las columnas numéricas relevantes para la correlación
numerical_columns = ['Session duration millis', 'Energy consumption', 'Number of connection', 'Number of disconnection', 'Offline duration']
correlation_data = data_first_3000[numerical_columns].dropna()

# Calcular la matriz de correlación
correlation_matrix = correlation_data.corr()

# Crear un heatmap de la matriz de correlación usando plotly
fig = px.imshow(correlation_matrix, 
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale='RdBu_r', 
                title="Matriz de Correlación")

fig.show()


# In[ ]:


import plotly.express as px

# Count the number of occurrences of each session status
status_counts = data_first_3000['Status'].value_counts()

# Create a bar chart to visualize the distribution of session statuses
fig_status_distribution = px.bar(
    status_counts,
    x=status_counts.index,
    y=status_counts.values,
    title='Distribución de Estados de las Sesiones',
    labels={'x': 'Estado', 'y': 'Cantidad'},
    text=status_counts.values
)

# Show the plot
fig_status_distribution.show()


# In[ ]:





# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd

# Selecting the relevant columns for clustering
features = ['Session duration millis', 'Energy consumption', 'Number of connection', 
            'Number of disconnection', 'Offline duration']

# Dropping rows with missing values in the selected features
data_selected = dfM[features].dropna()

# Standardizing the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Applying K-Means clustering with an optimal number of clusters (let's start with 4)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Adding cluster labels to the original data
data_selected['Cluster'] = clusters

# Visualizing the clusters using Plotly
fig = px.scatter_matrix(data_selected,
                        dimensions=features,
                        color='Cluster',
                        title="Segmentation of Sessions Based on Selected Features",
                        labels={col: col.replace('_', ' ') for col in data_selected.columns},
                        color_continuous_scale=px.colors.sequential.Viridis)

fig.update_layout(autosize=False, width=1200, height=1200)
fig.show()


# In[ ]:


dfbk=df


# In[ ]:


filtered_data = df[['Status', 'Session duration millis', 'Energy consumption']].dropna()

# Convertir la duración de la sesión de milisegundos a minutos
filtered_data['Session duration (minutes)'] = filtered_data['Session duration millis'] / (1000 * 60)

# Crear un gráfico de dispersión para comparar el consumo de energía y la duración de la sesión por estado
fig = px.scatter(
    filtered_data,
    x='Session duration (minutes)',
    y='Energy consumption',
    color='Status',
    labels={
        'Session duration (minutes)': 'Duración de la Sesión (minutos)',
        'Energy consumption': 'Consumo de Energía (kWh)',
        'Status': 'Estado de la Sesión'
    },
    title='Comparación del Consumo de Energía vs Duración de la Sesión por Estado'
)

# Mostrar la gráfica interactiva
fig.show()


# In[ ]:


# Convert 'Session start' to datetime format
df['Session start'] = pd.to_datetime(df['Session start'], errors='coerce', dayfirst=True)

# Filter data for the month of July
month = df[df['Session start'].dt.month == int(mes_anterior)]
#month2 = df[(df['Session start'].dt.month == int(mes_anterior)) & (df['Type of recharge(AC/DC)t'] == 'DC')]

# Perform the comparative analysis by grouping by 'Serial number' and 'UID'
comparison = month.groupby(['Serial number', 'UID', 'Type of recharge(AC/DC)']).agg({
    'Session ID': 'count',  # Número de sesiones
    'Session duration millis': 'sum',  # Duración total de la sesión en milisegundos
    'Energy consumption': 'sum'  # Consumo total de energía
}).reset_index()

# Rename columns for clarity
comparison.columns = ['Serial number', 'UID','Type of recharge(AC/DC)', 'Number of Sessions', 'Total Duration (ms)', 'Total Energy Consumption (kWh)']

# Export the data to an Excel file
output_excel_path = 'comparative_analysis_july.xlsx'
comparison.to_excel(output_excel_path, index=False)


# Seleccionar las 10 primeras entradas para un gráfico más claro
top_10_comparison = comparison.nlargest(10, 'Number of Sessions')

# Crear un gráfico de barras con Plotly
fig = px.bar(
    top_10_comparison,
    x='Serial number',
    y=[ 'Total Energy Consumption (kWh)','Number of Sessions'],
    barmode='group',
    title='Top 10 Devices by Number of Sessions and Energy Consumption in July',
    labels={
        'value': 'Count / Energy (kWh)',
        'Serial number': 'Serial Number'
    }
)
# Mostrar el gráfico
fig.show()
bottom_10_comparison = comparison.nsmallest(10, 'Number of Sessions')

comparison2 = comparison[comparison['Type of recharge(AC/DC)'] == 'DC']


# Seleccionar las 10 primeras entradas para un gráfico más claro
top_10_comparison2 = comparison2.nlargest(10, 'Number of Sessions')

# Crear un gráfico de barras con Plotly
fig = px.bar(
    top_10_comparison2,
    x='Serial number',
    y=[ 'Total Energy Consumption (kWh)','Number of Sessions'],
    barmode='group',
    title='Top 10 Devices by Number of Sessions and Energy Consumption in July',
    labels={
        'value': 'Count / Energy (kWh)',
        'Serial number': 'Serial Number'
    }
)


# Mostrar el gráfico
fig.show()
bottom_10_comparison = comparison.nsmallest(10, 'Number of Sessions')

# Crear un gráfico de barras con Plotly
fig = px.bar(
    bottom_10_comparison,
    x='Serial number',
    y=['Number of Sessions', 'Total Energy Consumption (kWh)'],
    barmode='group',
    title='Bottom 10 Devices by Number of Sessions and Energy Consumption in July',
    labels={
        'value': 'Count / Energy (kWh)',
        'Serial number': 'Serial Number'
    }
)

# Mostrar el gráfico
fig.show()


# In[ ]:


user_comparison = month.groupby(['UID', 'Tenant of the user']).agg({
    'Session ID': 'count',  # Número de sesiones
    'Session duration millis': 'sum',  # Duración total de las sesiones en milisegundos
    'Energy consumption': 'sum',  # Consumo total de energía
    'Type of recharge(AC/DC)': 'first'  # Tipo de recarga (mantener como referencia)
}).reset_index()

# Renombrar columnas para mayor claridad
user_comparison.columns = ['UID', 'Tenant of the user', 'Number of Sessions', 'Total Duration (ms)', 'Total Energy Consumption (kWh)', 'Type of recharge(AC/DC)']

# Ordenar el DataFrame por el número de sesiones para facilitar la visualización
user_comparison = user_comparison.sort_values(by='Number of Sessions', ascending=False)
comparison.to_excel('user_comparison.xlsx', index=False)


# In[ ]:


# Dividir el consumo total de energía por 1,000,000
user_comparison['Total Energy Consumption (MWh)'] = user_comparison['Total Energy Consumption (kWh)'] / 1000000

# Ordenar el DataFrame por el consumo total de energía para facilitar la visualización
user_comparison = user_comparison.sort_values(by='Total Energy Consumption (MWh)', ascending=False)

# Seleccionar los 10 usuarios con mayor consumo de energía
top_10_users_energy = user_comparison.head(10)

# Crear un gráfico de barras con Plotly
fig = px.bar(
    top_10_users_energy,
    x='UID',
    y='Total Energy Consumption (MWh)',
    color='Tenant of the user',  # Diferenciar por Tenant
    text='Total Energy Consumption (MWh)',
    title='Top 10 Usuarios por Consumo Total de Energía en Julio (en MWh)',
    labels={
        'UID': 'Usuario (UID)',
        'Total Energy Consumption (MWh)': 'Consumo Total de Energía (MWh)'
    }
)

# Mejorar la presentación del gráfico

fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(height=800,uniformtext_minsize=8, uniformtext_mode='hide')

# Mostrar el gráfico
fig.show()


# In[ ]:


user_comparison = dfM.groupby(['UID', 'Tenant of the user']).agg({
    'Session ID': 'count',  # Número de sesiones
    'Session duration millis': 'sum',  # Duración total de las sesiones en milisegundos
    'Energy consumption': 'sum'  # Consumo total de energía
}).reset_index()

# Renombrar columnas para mayor claridad
user_comparison.columns = ['UID', 'Tenant of the user', 'Number of Sessions', 'Total Duration (ms)', 'Total Energy Consumption (kWh)']

# Filtrar solo los usuarios de los tenants 'EnelX_CL' y 'EnelX_CL_residencial'
filtered_users = user_comparison[user_comparison['Tenant of the user'].isin(['EnelX_CL', 'EnelX_CL_residencial'])]

# Dividir el consumo total de energía por 1,000,000
filtered_users['Total Energy Consumption (MWh)'] = filtered_users['Total Energy Consumption (kWh)'] / 1000000

# Ordenar el DataFrame por el consumo total de energía para facilitar la visualización
filtered_users = filtered_users.sort_values(by='Total Energy Consumption (MWh)', ascending=False)

# Seleccionar los 10 usuarios con mayor consumo de energía
top_10_users_energy = filtered_users.head(10)

# Crear un gráfico de barras con Plotly
fig = px.bar(
    top_10_users_energy,
    x='UID',
    y='Total Energy Consumption (MWh)',
    color='Tenant of the user',  # Diferenciar por Tenant
    text='Total Energy Consumption (MWh)',
    title='Top 10 Usuarios de EnelX_CL y EnelX_CL_residencial por Consumo Total de Energía en Julio (en MWh)',
    labels={
        'UID': 'Usuario (UID)',
        'Total Energy Consumption (MWh)': 'Consumo Total de Energía (MWh)'
    }
)

# Mejorar la presentación del gráfico
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(height=800, uniformtext_minsize=8, uniformtext_mode='hide')

# Mostrar el gráfico
fig.show()


# In[ ]:


# Group by 'Tenant of the user' to calculate the number of users and energy consumed
tenant_comparison = month.groupby('Tenant of the user').agg({
    'UID': 'nunique',  # Number of unique users
    'Energy consumption': 'sum'  # Total energy consumption
}).reset_index()

# Rename columns for clarity
tenant_comparison.columns = ['Tenant of the user', 'Number of Users', 'Total Energy Consumption (kWh)']

# Filter out tenants with zero users or zero energy consumption
tenant_comparison = tenant_comparison[(tenant_comparison['Number of Users'] > 0) & 
                                      (tenant_comparison['Total Energy Consumption (kWh)'] > 0)]

# Convert energy consumption to MWh
tenant_comparison['Total Energy Consumption (MWh)'] = tenant_comparison['Total Energy Consumption (kWh)'] / 1000000

# Create a bar chart with Plotly
fig = px.bar(
    tenant_comparison,
    x='Tenant of the user',
    y='Number of Users',
    color='Total Energy Consumption (MWh)',
    text='Number of Users',
    title='Número de Usuarios por Tenant con Consumo de Energía en Julio',
    labels={
        'Tenant of the user': 'Tenant',
        'Number of Users': 'Número de Usuarios',
        'Total Energy Consumption (MWh)': 'Consumo Total de Energía (MWh)'
    },
    color_continuous_scale='RdBu'  # Adjust color scale as needed
)

# Improve the presentation
fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
fig.update_layout(height=800,xaxis={'categoryorder':'total descending'},  # Sort the bars by the number of users in descending order
    coloraxis_colorbar=dict(
        title="Consumo de Energía (MWh)",
        tickvals=[tenant_comparison['Total Energy Consumption (MWh)'].min(), tenant_comparison['Total Energy Consumption (MWh)'].max()],
    )
)

# Show the chart
fig.show()


# In[ ]:


data_full = df

# Convertir la columna 'Session start' a formato datetime
data_full['Session start'] = pd.to_datetime(data_full['Session start'])
data_full = data_full[data_full['Tenant of the user'].isin(['EnelX_CL', 'EnelX_CL_residencial'])]

# Filtrar las columnas relevantes para el análisis
data_user_sessions = data_full[['UID', 'Session start']].copy()

# Ordenar las sesiones por UID y fecha de inicio
data_user_sessions.sort_values(by=['UID', 'Session start'], inplace=True)

# Calcular la diferencia de tiempo entre sesiones consecutivas para cada usuario
data_user_sessions['Time Between Sessions'] = data_user_sessions.groupby('UID')['Session start'].diff().dt.total_seconds()

# Calcular la frecuencia de uso (número de sesiones por usuario)
frequency_of_use = data_user_sessions['UID'].value_counts()

# Resumen estadístico del tiempo entre sesiones
time_between_sessions_summary = data_user_sessions['Time Between Sessions'].describe()

# Resumen de la frecuencia de uso por usuario
frequency_of_use_summary = frequency_of_use.describe()

# Imprimir resultados
print("Resumen de la Frecuencia de Uso por Usuario:")
print(frequency_of_use_summary)

print("\nResumen del Tiempo entre Sesiones Consecutivas (en segundos):")
print(time_between_sessions_summary)

# Si quieres guardar los resultados en archivos separados:

frequency_of_use.to_excel('frequency_of_use.xlsx', index=True)
data_user_sessions.to_excel('user_sessions_with_time_between.xlsx', index=False)


# In[ ]:


data_full = df

# Convertir la columna 'Session start' a formato datetime
data_full['Session start'] = pd.to_datetime(data_full['Session start'])

# Extraer el mes y año de la columna 'Session start'
data_full['Year-Month'] = data_full['Session start'].dt.to_period('M')

# Filtrar las columnas relevantes para el análisis
data_user_sessions = data_full[['UID', 'Session start', 'Year-Month']].copy()

# Ordenar las sesiones por UID, luego por mes y fecha de inicio
data_user_sessions.sort_values(by=['UID', 'Year-Month', 'Session start'], inplace=True)

# Crear una lista para almacenar los resultados
results_list = []

# Iterar por cada mes
for month in data_user_sessions['Year-Month'].unique():
    # Filtrar datos del mes actual
    monthly_data = data_user_sessions.loc[data_user_sessions['Year-Month'] == month].copy()
    
    # Calcular la diferencia de tiempo entre sesiones consecutivas para cada usuario en ese mes
    monthly_data.loc[:, 'Time Between Sessions'] = monthly_data.groupby('UID')['Session start'].diff().dt.total_seconds()
    
    # Calcular la frecuencia de uso (número de sesiones por usuario en ese mes)
    frequency_of_use = monthly_data['UID'].value_counts()
    
    # Resumen estadístico del tiempo entre sesiones en ese mes
    time_between_sessions_summary = monthly_data['Time Between Sessions'].describe()
    
    # Resumen de la frecuencia de uso por usuario en ese mes
    frequency_of_use_summary = frequency_of_use.describe()
    
    # Guardar los resultados en la lista
    results_list.append({
        'Month': str(month),
        'Frequency of Use Count': frequency_of_use_summary['count'],
        'Frequency of Use Mean': frequency_of_use_summary['mean'],
        'Frequency of Use Std': frequency_of_use_summary['std'],
        'Frequency of Use Min': frequency_of_use_summary['min'],
        'Frequency of Use 25%': frequency_of_use_summary['25%'],
        'Frequency of Use 50% (Median)': frequency_of_use_summary['50%'],
        'Frequency of Use 75%': frequency_of_use_summary['75%'],
        'Frequency of Use Max': frequency_of_use_summary['max'],
        'Time Between Sessions Count': time_between_sessions_summary['count'],
        'Time Between Sessions Mean': time_between_sessions_summary['mean'],
        'Time Between Sessions Std': time_between_sessions_summary['std'],
        'Time Between Sessions Min': time_between_sessions_summary['min'],
        'Time Between Sessions 25%': time_between_sessions_summary['25%'],
        'Time Between Sessions 50% (Median)': time_between_sessions_summary['50%'],
        'Time Between Sessions 75%': time_between_sessions_summary['75%'],
        'Time Between Sessions Max': time_between_sessions_summary['max']
    })

# Convertir la lista de resultados en un DataFrame
results_df = pd.DataFrame(results_list)

# Imprimir el DataFrame para verificar
print(results_df)

# Opcional: Guardar el DataFrame en un archivo CSV para análisis posterior
results_df.to_csv('monthly_retention_analysis.csv', index=False)


# In[ ]:


import os
import numpy as np
from sklearn.linear_model import LinearRegression
def load_data(df):
    data_full = df
    data_full['Session start'] = pd.to_datetime(data_full['Session start'])
    data_full['Year-Month'] = data_full['Session start'].dt.to_period('M')
    return data_full

# Function to perform monthly analysis
def analyze_monthly_data(data):
    data_user_sessions = data[['UID', 'Session start', 'Year-Month']].copy()
    data_user_sessions.sort_values(by=['UID', 'Year-Month', 'Session start'], inplace=True)
    
    results_list = []

    for month in data_user_sessions['Year-Month'].unique():
        monthly_data = data_user_sessions.loc[data_user_sessions['Year-Month'] == month].copy()
        monthly_data.loc[:, 'Time Between Sessions'] = monthly_data.groupby('UID')['Session start'].diff().dt.total_seconds()
        frequency_of_use = monthly_data['UID'].value_counts()
        time_between_sessions_summary = monthly_data['Time Between Sessions'].describe()
        frequency_of_use_summary = frequency_of_use.describe()
        
        results_list.append({
            'Month': str(month),
            'Frequency of Use Count': frequency_of_use_summary['count'],
            'Frequency of Use Mean': frequency_of_use_summary['mean'],
            'Frequency of Use Std': frequency_of_use_summary['std'],
            'Frequency of Use Min': frequency_of_use_summary['min'],
            'Frequency of Use 25%': frequency_of_use_summary['25%'],
            'Frequency of Use 50% (Median)': frequency_of_use_summary['50%'],
            'Frequency of Use 75%': frequency_of_use_summary['75%'],
            'Frequency of Use Max': frequency_of_use_summary['max'],
            'Time Between Sessions Count': time_between_sessions_summary['count'],
            'Time Between Sessions Mean': time_between_sessions_summary['mean'],
            'Time Between Sessions Std': time_between_sessions_summary['std'],
            'Time Between Sessions Min': time_between_sessions_summary['min'],
            'Time Between Sessions 25%': time_between_sessions_summary['25%'],
            'Time Between Sessions 50% (Median)': time_between_sessions_summary['50%'],
            'Time Between Sessions 75%': time_between_sessions_summary['75%'],
            'Time Between Sessions Max': time_between_sessions_summary['max']
        })

    return pd.DataFrame(results_list)

# Function to visualize trends using Plotly
def visualize_trends(results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Frequency of Use Mean Over Time
    fig = px.line(results_df, x='Month', y='Frequency of Use Mean', title='Mean Frequency of Use Over Time')
    fig.write_html(os.path.join(output_dir, 'frequency_of_use_trend.html'))
    
    # Plot Time Between Sessions Median Over Time
    fig = px.line(results_df, x='Month', y='Time Between Sessions 50% (Median)', title='Median Time Between Sessions Over Time')
    fig.write_html(os.path.join(output_dir, 'time_between_sessions_trend.html'))

# Function to perform regression analysis using Plotly
def perform_regression_analysis(results_df, output_dir):
    results_df['Month_Num'] = np.arange(len(results_df))

    # Frequency of Use Mean Regression
    X = results_df[['Month_Num']]
    y = results_df['Frequency of Use Mean']
    model = LinearRegression()
    model.fit(X, y)
    trend_line = model.predict(X)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df['Month'], y=y, mode='lines+markers', name='Observed'))
    fig.add_trace(go.Scatter(x=results_df['Month'], y=trend_line, mode='lines', name='Trend Line'))
    fig.update_layout(title='Linear Regression: Frequency of Use Mean Over Time',
                      xaxis_title='Month',
                      yaxis_title='Mean Frequency of Use')
    fig.write_html(os.path.join(output_dir, 'frequency_of_use_regression.html'))

# Main function to automate analysis
def automate_trend_analysis(file_path, output_dir):
    data = load_data(file_path)
    results_df = analyze_monthly_data(data)
    visualize_trends(results_df, output_dir)
    perform_regression_analysis(results_df, output_dir)
    
    # Save the results DataFrame
    results_df.to_csv(os.path.join(output_dir, 'monthly_trend_analysis.csv'), index=False)
    print("Trend analysis completed. Results saved in:", output_dir)

# Run the automated analysis

output_dir = 'plotly_trend_analysis_results'

automate_trend_analysis(df, output_dir)


# In[ ]:


data_last_month = dfM

# Crear una columna adicional para identificar la hora del día
data_last_month['Hour'] = data_last_month['Session start'].dt.hour

# Agrupar los datos por día y hora para analizar la distribución de cargas
load_distribution = data_last_month.groupby([data_last_month['Session start'].dt.date, 'Hour']).size().reset_index(name='Session Count')

# Identificar posibles cuellos de botella
peak_load = load_distribution.groupby('Hour')['Session Count'].mean().reset_index()

import matplotlib.pyplot as plt


# Crear un gráfico interactivo de la distribución de la carga de trabajo por hora
fig = px.line(peak_load, x='Hour', y='Session Count', title='Distribución de la Carga de Trabajo en el Último Mes',
              labels={'Hour': 'Hora del Día', 'Session Count': 'Promedio de Sesiones Activas'})

fig.update_traces(mode='markers+lines')
fig.update_layout(xaxis=dict(dtick=1), yaxis_title='Promedio de Sesiones Activas')

fig.show()


# In[ ]:





# In[ ]:


data = dfM


# Identificar las sesiones críticas basadas en criterios clave:
# - Duración de la sesión no vacía
# - Alto consumo de energía (mayor que la media)
# - Número de conexiones mayor que la media
critical_sessions = data[
    (data['Session duration'] != '-') | 
    (data['Energy consumption'] > data['Energy consumption'].mean()) |
    (data['Number of connection'] > data['Number of connection'].mean())
]

# Evaluación de riesgos: Calcula un impacto simple
critical_sessions['Impact'] = critical_sessions['Energy consumption'] * critical_sessions['Number of connection']

# Eliminar filas con valores NaN en la columna 'Impact'
critical_sessions_clean = critical_sessions.dropna(subset=['Impact'])

# Visualización 1: Distribución del Consumo de Energía en Sesiones Críticas
fig_energy = px.histogram(critical_sessions_clean, x='Energy consumption', title='Distribución del Consumo de Energía en Sesiones Críticas')
fig_energy.show()

# Visualización 2: Distribución del Número de Conexiones en Sesiones Críticas
fig_connections = px.histogram(critical_sessions_clean, x='Number of connection', title='Distribución del Número de Conexiones en Sesiones Críticas')
fig_connections.show()

# Visualización 3: Sesiones Críticas por Impacto con información adicional
fig_impact = px.scatter(
    critical_sessions_clean,
    x='Session ID',
    y='Impact',
    size='Impact',
    color='Impact',
    title='Sesiones Críticas por Impacto',
    labels={'Session ID': 'ID de Sesión', 'Impact': 'Impacto'},
    hover_data={
        'Serial number': True,  # Muestra el número de serie
        'Session ID': False,    # No repitas el ID de Sesión en el tooltip
        'Station name': True,           # Muestra el tipo de cargador (Nombre del cargador)
    }
)
fig_impact.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
fig_impact.show()

# Almacenar los resultados en un archivo Excel
output_file = 'resultados_sesiones_criticas.xlsx'
with pd.ExcelWriter(output_file) as writer:
    critical_sessions_clean.to_excel(writer, sheet_name='Sesiones Críticas', index=False)
    summary = critical_sessions_clean.describe(include='all')
    summary.to_excel(writer, sheet_name='Resumen', index=True)

print(f"Resultados guardados en {output_file}")


# In[ ]:


from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime


data = df


# Filtro de datos para el análisis de consumo de energía
energy_data = data[['Session start', 'Energy consumption']].dropna()

# Convertir la columna de fecha y hora a un objeto datetime
energy_data['Session start'] = pd.to_datetime(energy_data['Session start'], format='%d/%m/%Y %H:%M:%S')

# Asegurarse de que los datos estén ordenados por fecha
energy_data = energy_data.sort_values('Session start')

# Resampleo de los datos para tener un consumo agregado semanal
energy_data_weekly = energy_data.resample('W', on='Session start').sum()

# Excluir la última semana del conjunto de datos
energy_data_weekly = energy_data_weekly[:-1]

# Dividir la energía por 1,000,000 para convertirla a GWh
energy_data_weekly['Energy consumption'] = energy_data_weekly['Energy consumption'] / 1000000

# Agregar el número de semana a los datos históricos
energy_data_weekly['Week Number'] = energy_data_weekly.index.isocalendar().week

# Modelado usando ARIMA para forecasting
model = ARIMA(energy_data_weekly['Energy consumption'], order=(5, 1, 0))
model_fit = model.fit()

# Calcular el número de semanas hasta fin de año
last_date = energy_data_weekly.index[-1]
end_of_year = pd.to_datetime(f'{last_date.year}-12-31')
weeks_until_end_of_year = (end_of_year - last_date).days // 7

# Hacer la predicción
forecast = model_fit.forecast(steps=weeks_until_end_of_year)

# Crear un índice de fechas para la proyección
forecast_index = pd.date_range(start=energy_data_weekly.index[-1] + pd.Timedelta(weeks=1), periods=weeks_until_end_of_year, freq='W')

# Crear un DataFrame para las semanas proyectadas
forecast_weeks = pd.DataFrame({
    'Date': forecast_index,
    'Energy consumption': forecast,
    'Week Number': forecast_index.isocalendar().week
})

# Graficar los datos históricos y la proyección con Plotly
fig = go.Figure()

# Agregar la serie histórica
fig.add_trace(go.Scatter(
    x=energy_data_weekly.index, 
    y=energy_data_weekly['Energy consumption'], 
    mode='lines+markers', 
    name='Histórico',
    text=[f"Semana: {week}<br>Energía: {energy:.6f} MWh" for week, energy in zip(energy_data_weekly['Week Number'], energy_data_weekly['Energy consumption'])],
    hoverinfo='text'
))

# Agregar la proyección
fig.add_trace(go.Scatter(
    x=forecast_weeks['Date'], 
    y=forecast_weeks['Energy consumption'], 
    mode='lines+markers', 
    name='Proyección', 
    line=dict(color='red'),
    text=[f"Semana: {week}<br>Energía: {energy:.6f} MWh" for week, energy in zip(forecast_weeks['Week Number'], forecast_weeks['Energy consumption'])],
    hoverinfo='text'
))

# Configurar el layout del gráfico
fig.update_layout(
    title='Proyección Semanal del Consumo de Energía (MWh) hasta Fin de Año',
    xaxis_title='Fecha',
    yaxis_title='Energía Consumida (MWh)',
    legend=dict(x=0.01, y=0.99),
    template='plotly_white'
)

# Mostrar el gráfico
fig.show()


# In[ ]:


# Filtro de datos para el análisis de consumo de energía
energy_data = data[['Session start', 'Energy consumption']].dropna()

# Convertir la columna de fecha y hora a un objeto datetime
energy_data['Session start'] = pd.to_datetime(energy_data['Session start'], format='%d/%m/%Y %H:%M:%S')

# Asegurarse de que los datos estén ordenados por fecha
energy_data = energy_data.sort_values('Session start')

# Resampleo de los datos para tener un consumo agregado semanal
energy_data_weekly = energy_data.resample('W', on='Session start').sum()

# Excluir la última semana del conjunto de datos
energy_data_weekly = energy_data_weekly[:-1]

# Dividir la energía por 1,000,000 para convertirla a GWh
energy_data_weekly['Energy consumption'] = energy_data_weekly['Energy consumption'] / 1000000

# Agregar el número de semana a los datos históricos
energy_data_weekly['Week Number'] = energy_data_weekly.index.isocalendar().week

# Modelado usando ARIMA para forecasting
model = ARIMA(energy_data_weekly['Energy consumption'], order=(5, 1, 0))
model_fit = model.fit()

# Calcular el número de semanas hasta fin de año
last_date = energy_data_weekly.index[-1]
end_of_year = pd.to_datetime(f'{last_date.year}-12-31')
weeks_until_end_of_year = (end_of_year - last_date).days // 7

# Hacer la predicción
forecast = model_fit.forecast(steps=weeks_until_end_of_year)

# Crear un índice de fechas para la proyección
forecast_index = pd.date_range(start=energy_data_weekly.index[-1] + pd.Timedelta(weeks=1), periods=weeks_until_end_of_year, freq='W')

# Crear un DataFrame para las semanas proyectadas
forecast_weeks = pd.DataFrame({
    'Date': forecast_index,
    'Energy consumption': forecast,
    'Week Number': forecast_index.isocalendar().week
})

# Calcular el acumulado de la energía histórica
energy_data_weekly['Cumulative'] = energy_data_weekly['Energy consumption'].cumsum()

# Calcular el acumulado de la proyección, comenzando desde el último valor acumulado
forecast_weeks['Cumulative'] = forecast_weeks['Energy consumption'].cumsum() + energy_data_weekly['Cumulative'].iloc[-1]

# Graficar los datos acumulados y proyectados con Plotly en un gráfico de área apilada
fig = go.Figure()

# Agregar la serie histórica acumulada
fig.add_trace(go.Scatter(
    x=energy_data_weekly.index, 
    y=energy_data_weekly['Cumulative'], 
    mode='lines',
    name='Acumulado Histórico',
    fill='tozeroy',  # Llena el área desde el eje Y hasta la curva
    line=dict(color='blue')
))

# Agregar la proyección acumulada
fig.add_trace(go.Scatter(
    x=forecast_weeks['Date'], 
    y=forecast_weeks['Cumulative'], 
    mode='lines',
    name='Proyección Acumulada',
    fill='tonexty',  # Llena el área entre la curva anterior y la actual
    line=dict(color='red')
))

# Configurar el layout del gráfico
fig.update_layout(
    title='Consumo de Energía Acumulado y Proyectado (GWh) - Área Apilada',
    xaxis_title='Fecha',
    yaxis_title='Energía Acumulada (GWh)',
    showlegend=True,
    template='plotly_white'
)

# Mostrar el gráfico
fig.show()


# In[ ]:


# Filtro de datos para el análisis de consumo de energía
energy_data = data[['Session start', 'Energy consumption']].dropna()

# Convertir la columna de fecha y hora a un objeto datetime
energy_data['Session start'] = pd.to_datetime(energy_data['Session start'], format='%d/%m/%Y %H:%M:%S')

# Asegurarse de que los datos estén ordenados por fecha
energy_data = energy_data.sort_values('Session start')

# Resampleo de los datos para tener un consumo agregado semanal
energy_data_weekly = energy_data.resample('W', on='Session start').sum()

# Excluir la última semana del conjunto de datos
energy_data_weekly = energy_data_weekly[:-1]

# Dividir la energía por 1,000,000 para convertirla a GWh
energy_data_weekly['Energy consumption'] = energy_data_weekly['Energy consumption'] / 1000000

# Agregar el número de semana a los datos históricos
energy_data_weekly['Week Number'] = energy_data_weekly.index.isocalendar().week

# Modelado usando ARIMA para forecasting
model = ARIMA(energy_data_weekly['Energy consumption'], order=(5, 1, 0))
model_fit = model.fit()

# Calcular el número de semanas hasta fin de año
last_date = energy_data_weekly.index[-1]
end_of_year = pd.to_datetime(f'{last_date.year}-12-31')
weeks_until_end_of_year = (end_of_year - last_date).days // 7

# Hacer la predicción
forecast = model_fit.forecast(steps=weeks_until_end_of_year)

# Crear un índice de fechas para la proyección
forecast_index = pd.date_range(start=energy_data_weekly.index[-1] + pd.Timedelta(weeks=1), periods=weeks_until_end_of_year, freq='W')

# Crear un DataFrame para las semanas proyectadas
forecast_weeks = pd.DataFrame({
    'Date': forecast_index,
    'Energy consumption': forecast,
    'Week Number': forecast_index.isocalendar().week
})

# Calcular el acumulado de la energía histórica
energy_data_weekly['Cumulative'] = energy_data_weekly['Energy consumption'].sum()

# Calcular el acumulado de la proyección
forecast_cumulative = forecast_weeks['Energy consumption'].sum()

# Crear el gráfico de barras apiladas en una sola barra
fig = go.Figure()

# Agregar la barra histórica acumulada
fig.add_trace(go.Bar(
    x=['Consumo Acumulado'],
    y=[energy_data_weekly['Cumulative'].iloc[-1]],
    name='Acumulado Histórico',
    marker=dict(color='blue')
))

# Agregar la barra proyectada acumulada
fig.add_trace(go.Bar(
    x=['Consumo Acumulado'],
    y=[forecast_cumulative],
    name='Proyección Acumulada',
    marker=dict(color='red')
))

# Configurar el layout del gráfico
fig.update_layout(
    title='Consumo de Energía Acumulado y Proyectado (GWh) en una Sola Barra',
    xaxis_title='',
    yaxis_title='Energía (GWh)',
    barmode='stack',
    showlegend=True,
    template='plotly_white'
)

# Mostrar el gráfico
fig.show()


# In[ ]:


# Convertir la columna de inicio de sesión a formato de fecha
data=df


# Convertir la columna de inicio de sesión a formato de fecha y hora
data['Session start'] = pd.to_datetime(data['Session start'])

# Extraer el mes y el año para cada sesión
data['Month'] = data['Session start'].dt.to_period('M')

# Identificar el mes actual (último mes en los datos)
current_month = data['Month'].max()

# Separar los datos históricos (todos los meses excepto el actual)
data_historic = data[data['Month'] < current_month]

# Datos actuales (el último mes disponible)
data_current = data[data['Month'] == current_month]

# Calcular métricas para los datos históricos y actuales
# Duración total de sesiones por mes (en milisegundos convertido a horas)
historic_duration_per_month = data_historic.groupby('Month')['Session duration millis'].sum() / (1000 * 60 * 60)
historic_duration_avg = historic_duration_per_month.mean()

current_duration = data_current['Session duration millis'].sum() / (1000 * 60 * 60)

# Consumo total de energía (en kWh)
historic_energy = data_historic.groupby('Month')['Energy consumption'].sum().mean()
current_energy = data_current['Energy consumption'].sum()

# Número total de sesiones
historic_sessions = data_historic.groupby('Month').size().mean()
current_sessions = len(data_current)

# Crear la gráfica del promedio de la duración total de sesión mensual
fig_duration = go.Figure()

fig_duration.add_trace(go.Bar(
    x=['Duración promedio de sesión mensual (horas)'],
    y=[historic_duration_avg],
    name='Histórico',
    marker_color='blue'
))

fig_duration.add_trace(go.Bar(
    x=['Duración promedio de sesión mensual (horas)'],
    y=[current_duration],
    name='Actual',
    marker_color='orange'
))

fig_duration.update_layout(
    title='Duración Promedio de Sesión Mensual: Histórico vs. Actual',
    xaxis_title='Métrica',
    yaxis_title='Duración (horas)',
    barmode='group'
)

# Crear la gráfica de Consumo total de energía
fig_energy = go.Figure()

fig_energy.add_trace(go.Bar(
    x=['Consumo total de energía (kWh)'],
    y=[historic_energy],
    name='Histórico',
    marker_color='blue'
))

fig_energy.add_trace(go.Bar(
    x=['Consumo total de energía (kWh)'],
    y=[current_energy],
    name='Actual',
    marker_color='orange'
))

fig_energy.update_layout(
    title='Consumo Total de Energía: Histórico vs. Actual',
    xaxis_title='Métrica',
    yaxis_title='Consumo (kWh)',
    barmode='group'
)

# Crear la gráfica del Número total de sesiones
fig_sessions = go.Figure()

fig_sessions.add_trace(go.Bar(
    x=['Número total de sesiones'],
    y=[historic_sessions],
    name='Histórico',
    marker_color='blue'
))

fig_sessions.add_trace(go.Bar(
    x=['Número total de sesiones'],
    y=[current_sessions],
    name='Actual',
    marker_color='orange'
))

fig_sessions.update_layout(
    title='Número Total de Sesiones: Histórico vs. Actual',
    xaxis_title='Métrica',
    yaxis_title='Número de sesiones',
    barmode='group'
)

# Mostrar las gráficas
fig_duration.show()
fig_energy.show()
fig_sessions.show()


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np


data = df
# Selección de columnas relevantes para clustering
columns_to_use = [
    'Session duration millis', 'Energy consumption',
    'Number of connection', 'Number of disconnection', 'Offline duration'
]

# Filtrar columnas relevantes
data_filtered = data[columns_to_use]

# Reemplazar valores no numéricos (e.g., '-') con NaN y eliminar esas filas
data_filtered.replace('-', np.nan, inplace=True)
data_filtered.dropna(inplace=True)

# Convertir solo las columnas que contienen valores numéricos
for col in data_filtered.columns:
    if data_filtered[col].dtype == 'object':
        try:
            data_filtered[col] = pd.to_numeric(data_filtered[col])
        except ValueError:
            data_filtered.drop(columns=[col], inplace=True)

# Estandarización de los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_filtered)

# Determinar el número óptimo de clusters utilizando el método del codo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Visualizar el Método del Codo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

# Seleccionar el número óptimo de clusters y aplicar K-Means
optimal_clusters = 4  # Ajusta este número según la gráfica del codo
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Agregar los clusters al DataFrame original
data_filtered['Cluster'] = clusters

# Visualización 3D de los clusters
fig_3d = px.scatter_3d(
    data_filtered, 
    x='Session duration millis', 
    y='Energy consumption', 
    z='Number of connection', 
    color='Cluster',
    title='Clustering de Sesiones: Duración vs Consumo de Energía vs Conexiones',
    labels={'Session duration millis': 'Duración (ms)', 'Energy consumption': 'Consumo de Energía (kWh)', 'Number of connection': 'Conexiones'}
)

fig_3d.show()

# Guardar los resultados en un archivo Excel
data_filtered.to_excel('clustering_results.xlsx', index=False)


# In[ ]:


data = df

# Selección de columnas relevantes para clustering e incluir "Tenant of the user" y "User ID"
columns_to_use = [
    'Session duration millis', 'Energy consumption',
    'Number of connection', 'Number of disconnection', 'Offline duration'
]

# Filtrar columnas relevantes y agregar "Tenant of the user" y "User ID"
data_filtered = data[columns_to_use + ['Tenant of the user', 'User ID']]

# Reemplazar valores no numéricos (e.g., '-') con NaN y eliminar esas filas
data_filtered.replace('-', np.nan, inplace=True)
data_filtered.dropna(inplace=True)

# Convertir solo las columnas que contienen valores numéricos
for col in columns_to_use:
    if col in data_filtered.columns and data_filtered[col].dtype == 'object':
        try:
            data_filtered[col] = pd.to_numeric(data_filtered[col])
        except ValueError:
            data_filtered.drop(columns=[col], inplace=True)

# Actualizar las columnas que quedan después de la limpieza
final_columns = [col for col in columns_to_use if col in data_filtered.columns]

# Estandarización de los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_filtered[final_columns])

# Aplicar K-Means con un número de clusters óptimo (por ejemplo, 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Agregar los clusters al DataFrame original
data_filtered['Cluster'] = clusters

# Crear la matriz de scatter incluyendo los clusters, "Tenant of the user" y "User ID"
fig = px.scatter_matrix(
    data_filtered,
    dimensions=final_columns,  # Usar solo las columnas que quedaron
    color='Cluster',
    hover_data=['Tenant of the user', 'User ID'],  # Mostrar "Tenant of the user" y "User ID" al pasar el cursor
    title="Matriz de Scatter de las Características Seleccionadas con Clusters, Tenant y User ID",
    labels={col: col.replace('_', ' ') for col in final_columns}
)
fig.update_layout(autosize=False, width=1200, height=1200)

# Mostrar la matriz de scatter
fig.show()

# Exportar los resultados a Excel incluyendo "User ID"
output_file = 'clustering_results_with_tenant_userid.xlsx'
data_filtered.to_excel(output_file, index=False)

print(f'Resultados exportados a {output_file}')


# In[ ]:


# Selección de columnas relevantes para el análisis
columns_to_use = [
    'Session duration millis', 'Energy consumption',
    'Number of connection', 'Number of disconnection', 'Offline duration'
]

# Incluir "User ID" y "Tenant of the user"
data_filtered = data[columns_to_use + ['User ID', 'Tenant of the user']]

# Reemplazar valores no numéricos (e.g., '-') con NaN y eliminar esas filas
data_filtered.replace('-', np.nan, inplace=True)
data_filtered.dropna(inplace=True)

# Convertir las columnas numéricas
for col in columns_to_use:
    if col in data_filtered.columns and data_filtered[col].dtype == 'object':
        try:
            data_filtered[col] = pd.to_numeric(data_filtered[col])
        except ValueError:
            data_filtered.drop(columns=[col], inplace=True)

# Actualizar las columnas que quedan después de la limpieza
final_columns = [col for col in columns_to_use if col in data_filtered.columns]

# Agrupar por "User ID" y calcular estadísticas agregadas solo con las columnas que quedaron
user_grouped = data_filtered.groupby('User ID').agg({
    col: 'mean' for col in final_columns
}).reset_index()

# Incluir el "Tenant of the user"
user_grouped = pd.merge(user_grouped, data_filtered[['User ID', 'Tenant of the user']].drop_duplicates(), on='User ID')

# Estandarización de los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(user_grouped[final_columns])

# Aplicar K-Means con un número de clusters óptimo (por ejemplo, 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Agregar los clusters al DataFrame
user_grouped['Cluster'] = clusters

# Crear la matriz de scatter incluyendo los clusters y "Tenant of the user"
fig = px.scatter_matrix(
    user_grouped,
    dimensions=final_columns,  # Usar solo las columnas numéricas que quedaron
    color='Cluster',
    hover_data=['Tenant of the user', 'User ID'],  # Mostrar "Tenant of the user" y "User ID" al pasar el cursor
    title="Matriz de Scatter de las Características por Usuario con Clusters y Tenant",
    labels={col: col.replace('_', ' ') for col in final_columns}
)
fig.update_layout(autosize=False, width=1200, height=1200)
# Mostrar la matriz de scatter
fig.show()

# Exportar los resultados a Excel incluyendo "User ID"
output_file = 'user_clustering_results_v2.xlsx'
user_grouped.to_excel(output_file, index=False)

print(f'Resultados exportados a {output_file}')


# In[ ]:


# Asumimos que data_first_3000 ya está cargado y disponible
# Convertir las fechas a datetime para análisis temporal
data_first_3000=df
data_first_3000['Session start'] = pd.to_datetime(data_first_3000['Session start'], format='%d/%m/%Y %H:%M:%S')

# 1. Distribución de Razones de Finalización
end_reasons_distribution = data_first_3000['End of charge reason'].value_counts()
fig1 = px.bar(end_reasons_distribution, title='Distribución de Razones de Finalización')
fig1.show()

# 2. Comparación entre Razones (Duración y Consumo de Energía)
filtered_data = data_first_3000.dropna(subset=['Session duration', 'Energy consumption'])

# Comparación de duración
fig2 = px.box(filtered_data, x='End of charge reason', y='Session duration', title='Comparación de Duración por Razón de Finalización')
fig2.show()

# Comparación de consumo de energía
fig3 = px.box(filtered_data, x='End of charge reason', y='Energy consumption', title='Comparación de Consumo de Energía por Razón de Finalización')
fig3.show()

# 3. Eficiencia Energética por Razón
filtered_data['Efficiency'] = filtered_data['Energy consumption'] / (filtered_data['Session duration millis'] / 3600000) # Consumo de energía por hora
fig4 = px.box(filtered_data, x='End of charge reason', y='Efficiency', title='Eficiencia Energética por Razón de Finalización')
fig4.show()

# 4. Duración Promedio por Razón
average_duration = filtered_data.groupby('End of charge reason')['Session duration millis'].mean().reset_index()
fig5 = px.bar(average_duration, x='End of charge reason', y='Session duration millis', title='Duración Promedio por Razón de Finalización')
fig5.show()

# 5. Tendencias en Razones de Finalización
# Corregido: Convertir el 'Period' a string
end_reasons_over_time = data_first_3000.groupby([data_first_3000['Session start'].dt.to_period('M').astype(str), 'End of charge reason']).size().reset_index(name='count')

# Crear la gráfica de líneas
fig6 = px.line(end_reasons_over_time, x='Session start', y='count', color='End of charge reason', title='Tendencias en Razones de Finalización a lo Largo del Tiempo')
fig6.show()

# 6. Patrones Diarios/Semanales
data_first_3000['day_of_week'] = data_first_3000['Session start'].dt.day_name()
daily_patterns = data_first_3000.groupby(['day_of_week', 'End of charge reason']).size().reset_index(name='count')
fig7 = px.bar(daily_patterns, x='day_of_week', y='count', color='End of charge reason', title='Patrones Diarios de Razones de Finalización')
fig7.show()

# 7. Detección de Razones Anómalas
anomalous_reasons = filtered_data[filtered_data['End of charge reason'].isin(['Error', 'Desconexión Manual'])]
fig8 = px.scatter(anomalous_reasons, x='Session duration', y='Energy consumption', color='End of charge reason', title='Detección de Razones Anómalas')
fig8.show()

# 8. Relación con Estado de la Sesión
state_vs_end_reason = data_first_3000.groupby(['Status', 'End of charge reason']).size().reset_index(name='count')
fig9 = px.bar(state_vs_end_reason, x='Status', y='count', color='End of charge reason', title='Relación entre Estado de la Sesión y Razón de Finalización')
fig9.show()

# 9. Impacto en el Retorno de Usuarios
user_return = data_first_3000.groupby(['UID', 'End of charge reason']).size().reset_index(name='count')
fig10 = px.histogram(user_return, x='count', color='End of charge reason', title='Impacto de la Razón de Finalización en el Retorno de Usuarios', nbins=20)
fig10.show()

# 10. Eficiencia Energética Media por Razón
mean_efficiency = filtered_data.groupby('End of charge reason')['Efficiency'].mean().reset_index()
fig11 = px.bar(mean_efficiency, x='End of charge reason', y='Efficiency', title='Eficiencia Energética Media por Razón de Finalización')
fig11.show()


# In[ ]:


# Asumimos que data_first_3000 ya está cargado y disponible
# Convertir las fechas a datetime para análisis temporal
data_first_3000=dfM
data_first_3000['Session start'] = pd.to_datetime(data_first_3000['Session start'], format='%d/%m/%Y %H:%M:%S')

# 1. Distribución de Razones de Finalización
end_reasons_distribution = data_first_3000['End of charge reason'].value_counts()
fig1 = px.bar(end_reasons_distribution, title='Distribución de Razones de Finalización')
fig1.show()

# 2. Comparación entre Razones (Duración y Consumo de Energía)
filtered_data = data_first_3000.dropna(subset=['Session duration', 'Energy consumption'])

# Comparación de duración
fig2 = px.box(filtered_data, x='End of charge reason', y='Session duration', title='Comparación de Duración por Razón de Finalización')
fig2.show()

# Comparación de consumo de energía
fig3 = px.box(filtered_data, x='End of charge reason', y='Energy consumption', title='Comparación de Consumo de Energía por Razón de Finalización')
fig3.show()

# 3. Eficiencia Energética por Razón
filtered_data['Efficiency'] = filtered_data['Energy consumption'] / (filtered_data['Session duration millis'] / 3600000) # Consumo de energía por hora
fig4 = px.box(filtered_data, x='End of charge reason', y='Efficiency', title='Eficiencia Energética por Razón de Finalización')
fig4.show()

# 4. Duración Promedio por Razón
average_duration = filtered_data.groupby('End of charge reason')['Session duration millis'].mean().reset_index()
fig5 = px.bar(average_duration, x='End of charge reason', y='Session duration millis', title='Duración Promedio por Razón de Finalización')
fig5.show()

# 5. Tendencias en Razones de Finalización
# Corregido: Convertir el 'Period' a string
end_reasons_over_time = data_first_3000.groupby([data_first_3000['Session start'].dt.to_period('M').astype(str), 'End of charge reason']).size().reset_index(name='count')

# Crear la gráfica de líneas
fig6 = px.line(end_reasons_over_time, x='Session start', y='count', color='End of charge reason', title='Tendencias en Razones de Finalización a lo Largo del Tiempo')
fig6.show()

# 6. Patrones Diarios/Semanales
data_first_3000['day_of_week'] = data_first_3000['Session start'].dt.day_name()
daily_patterns = data_first_3000.groupby(['day_of_week', 'End of charge reason']).size().reset_index(name='count')
fig7 = px.bar(daily_patterns, x='day_of_week', y='count', color='End of charge reason', title='Patrones Diarios de Razones de Finalización')
fig7.show()

# 7. Detección de Razones Anómalas
anomalous_reasons = filtered_data[filtered_data['End of charge reason'].isin(['Error', 'Desconexión Manual'])]
fig8 = px.scatter(anomalous_reasons, x='Session duration', y='Energy consumption', color='End of charge reason', title='Detección de Razones Anómalas')
fig8.show()

# 8. Relación con Estado de la Sesión
state_vs_end_reason = data_first_3000.groupby(['Status', 'End of charge reason']).size().reset_index(name='count')
fig9 = px.bar(state_vs_end_reason, x='Status', y='count', color='End of charge reason', title='Relación entre Estado de la Sesión y Razón de Finalización')
fig9.show()

# 9. Impacto en el Retorno de Usuarios
user_return = data_first_3000.groupby(['UID', 'End of charge reason']).size().reset_index(name='count')
fig10 = px.histogram(user_return, x='count', color='End of charge reason', title='Impacto de la Razón de Finalización en el Retorno de Usuarios', nbins=20)
fig10.show()

# 10. Eficiencia Energética Media por Razón
mean_efficiency = filtered_data.groupby('End of charge reason')['Efficiency'].mean().reset_index()
fig11 = px.bar(mean_efficiency, x='End of charge reason', y='Efficiency', title='Eficiencia Energética Media por Razón de Finalización')
fig11.show()


# In[ ]:


# 1. Análisis de Cohortes (Opción 25)
# Supongamos que "Serial number" representa a los dispositivos y "Session start" contiene la fecha de la sesión.
data=df
data['Session start'] = pd.to_datetime(data['Session start'], format='%d/%m/%Y %H:%M:%S')

# 1. Análisis de Cohortes (Opción 25)
# Supongamos que "Serial number" representa a los dispositivos y "Session start" contiene la fecha de la sesión.
data['Session start'] = pd.to_datetime(data['Session start'], format='%d/%m/%Y %H:%M:%S')
data['Cohort'] = data['Session start'].dt.to_period('M').astype(str)  # Convertir a string para que sea serializable
cohort_analysis = data.groupby(['Cohort', 'Serial number']).size().reset_index(name='Session Count')

# 2. Análisis Comparativo entre Regiones o Dispositivos (Opción 28)
# Si la columna "Site" representa diferentes regiones, podríamos hacer una comparación.
if 'Site' in data.columns:
    region_comparison = data.groupby('Site')['Energy consumption'].sum().reset_index()

# 3. Análisis de Satisfacción del Usuario (Opción 32)
# Mediremos la frecuencia de sesiones por dispositivo y el tiempo promedio entre sesiones.
user_satisfaction = data.groupby('Serial number').agg(
    Total_Sessions=('Session ID', 'count'),
    Avg_Time_Between_Sessions=('Session start', lambda x: (x.max() - x.min()).days / x.count())
).reset_index()

# Visualizaciones

# Visualización de Cohortes
fig_cohort = px.line(cohort_analysis, x='Cohort', y='Session Count', color='Serial number',
                     title='Cohort Analysis: Sessions over Time by Serial Number')

# Visualización de Comparación entre Regiones
if 'Site' in data.columns:
    fig_region = px.bar(region_comparison, x='Site', y='Energy consumption',
                        title='Energy Consumption by Region')

# Visualización de Satisfacción del Usuario
fig_satisfaction = px.scatter(user_satisfaction, x='Total_Sessions', y='Avg_Time_Between_Sessions',
                              title='User Satisfaction: Frequency of Sessions vs. Average Time Between Sessions')

# Mostrar las visualizaciones
fig_cohort.show()
if 'Site' in data.columns:
    fig_region.show()
fig_satisfaction.show()

# Filtrar y limpiar los datos de la columna 'Session duration'
# Reemplazar valores no válidos con NaT
def safe_to_timedelta(val):
    try:
        return pd.to_timedelta(val)
    except ValueError:
        return np.nan

data['Session duration'] = data['Session duration'].apply(safe_to_timedelta)
data['Session duration'] = data['Session duration'].fillna(pd.Timedelta(seconds=0))

# Convertir la columna 'Energy consumption' a numérico
data['Energy consumption'] = pd.to_numeric(data['Energy consumption'], errors='coerce')

# 1. Medir la Satisfacción Estimada
# Calcular la cantidad total de sesiones y el tiempo promedio entre sesiones por dispositivo
user_satisfaction = data.groupby('Serial number').agg(
    Total_Sessions=('Session ID', 'count'),
    Avg_Time_Between_Sessions=('Session start', lambda x: (x.max() - x.min()).days / max(1, x.count() - 1))
).reset_index()

# 2. Impacto de la Experiencia de Carga
# Analizar el impacto de la duración y consumo de energía en la satisfacción del usuario
impact_analysis = data.groupby('Serial number').agg(
    Avg_Session_Duration=('Session duration', 'mean'),
    Total_Energy_Consumption=('Energy consumption', 'sum'),
    Total_Sessions=('Session ID', 'count'),
    Avg_Time_Between_Sessions=('Session start', lambda x: (x.max() - x.min()).days / max(1, x.count() - 1))
).reset_index()

# Visualizaciones

# Visualizar la relación entre la cantidad de sesiones y el tiempo promedio entre sesiones
fig_satisfaction = px.scatter(user_satisfaction, x='Total_Sessions', y='Avg_Time_Between_Sessions',
                              title='User Satisfaction: Frequency of Sessions vs. Average Time Between Sessions')

# Visualizar el impacto de la duración de la sesión y el consumo de energía en la satisfacción
fig_impact = px.scatter(impact_analysis, x='Avg_Session_Duration', y='Avg_Time_Between_Sessions',
                        size='Total_Energy_Consumption', color='Total_Sessions',
                        title='Impact of Session Duration and Energy Consumption on User Satisfaction',
                        labels={'Avg_Session_Duration': 'Average Session Duration',
                                'Avg_Time_Between_Sessions': 'Average Time Between Sessions'})

# Mostrar las visualizaciones
fig_satisfaction.show()
fig_impact.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
import numpy as np
df=dfbk
# Cargar los datos principales y la tabla con la latitud y longitud
df_sessions = df
df_locations = dfst

# Unir las tablas basadas en el número de serie (ajustando nombres de columnas)
df = pd.merge(df_sessions, df_locations, left_on='Serial number', right_on='Station serial number', how='left')

# Eliminar filas con valores NaN en las columnas de latitud o longitud
df = df.dropna(subset=['Latitude', 'Longitude'])

# Crear un GeoDataFrame para visualizaciones geográficas
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# 1. Mapa de Calor de Ubicaciones de Cargadores (usando un histograma bidimensional)
#plt.figure(figsize=(10, 8))
#plt.hist2d(df['Longitude'], df['Latitude'], bins=(50, 50), cmap=plt.cm.jet)
#plt.colorbar(label='Número de sesiones')
#plt.xlabel('Longitud')
#plt.ylabel('Latitud')
#plt.title('Mapa de Calor de Ubicaciones de Cargadores')
#plt.show()
# Crear un mapa de calor
df = df.dropna(subset=['Energy consumption'])

fig = go.Figure(go.Scattergeo(
    lon=df['Longitude'],
    lat=df['Latitude'],
    text=df['Energy consumption'],
    marker=dict(
        size=10,
        color=df['Energy consumption'],
        colorscale='rdylgn_r',
        showscale=True,
        colorbar=dict(title="Consumo Energético (kWh)")
    )
))

fig.update_layout(
    title="Mapa de Calor de Consumo Energético en Ubicaciones de Cargadores",
    geo=dict(
        scope='world',
        projection_type='natural earth',
        showland=True,
        landcolor='rgb(217, 217, 217)',
    )
)

# Mostrar el gráfico
fig.write_html("Mapa de Calor Consumo.html")


# In[ ]:


# 2. Distribución Geográfica de Cargadores
gdf.plot(marker='o', color='blue', markersize=5, figsize=(10, 8))
plt.title('Distribución Geográfica de Cargadores')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()

# 3. Clusters Geográficos de Cargadores
coords = df[['Latitude', 'Longitude']].dropna()
db = DBSCAN(eps=0.01, min_samples=5).fit(coords)
df['Cluster'] = db.labels_

# Visualización de clusters
plt.figure(figsize=(10, 8))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='rainbow', s=10)
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.title('Clusters Geográficos de Cargadores')
plt.show()

# 4. Análisis de Accesibilidad - Distancia al Cargador más Cercano
dist_matrix = distance_matrix(coords.values, coords.values)
np.fill_diagonal(dist_matrix, np.inf)
df['Min_Distance'] = dist_matrix.min(axis=1)

plt.figure(figsize=(10, 6))
plt.hist(df['Min_Distance'], bins=50, color='green')
plt.title("Distancia al Cargador Más Cercano")
plt.xlabel('Distancia (grados)')
plt.ylabel('Frecuencia')
plt.show()

# 5. Comparación entre Regiones (si existe una columna 'Region' en df)
if 'Region' in df.columns:
    df.boxplot(column='Energy consumption', by='Region', grid=False, figsize=(10, 6))
    plt.title('Consumo de Energía por Región')
    plt.suptitle('')
    plt.xlabel('Región')
    plt.ylabel('Consumo de Energía')
    plt.show()

# 6. Mapa de Clusters y Demanda
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='rainbow', s=df['Session ID'].apply(lambda x: len(str(x))), alpha=0.5)
plt.colorbar(scatter)
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.title('Clusters de Cargadores con Demanda')
plt.show()


# In[ ]:


df.columns.tolist()


# In[ ]:


df['Latitude']


# In[ ]:




