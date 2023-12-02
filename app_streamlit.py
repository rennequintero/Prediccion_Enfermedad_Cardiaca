# Importamos las bibliotecas necesarias
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder

# Configuramos la página de Streamlit
st.set_page_config(page_title="App de predicción",
                   page_icon='https://cdn-icons-png.flaticon.com/512/5935/5935638.png',
                   layout="centered",
                   initial_sidebar_state="auto")

# Definimos el título y la descripción de la aplicación
st.title("App de predicción de enfermedades cardiacas")
st.markdown("""Esta aplicación predice si tienes una enfermedad cardiaca basándose en datos ingresados.""")
st.markdown("""---""")

# Cargamos y mostramos el logo en la barra lateral
logo = "Corazon.png"
st.sidebar.image(logo, width=150)

# Añadimos un encabezado para la sección de datos del usuario en la barra lateral
st.sidebar.header('Datos ingresados por el usuario')

# Permitimos al usuario cargar un archivo csv o ingresar datos manualmente
uploaded_file = st.sidebar.file_uploader("Cargue su archivo csv", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        # Creamos controles deslizantes y cuadros de selección para que el usuario ingrese los datos
        sbp = st.sidebar.slider('Presión Arterial Sistólica', 101, 218, 150)
        Tabaco = st.sidebar.slider('Tabaco Acumulado (Kg)', 0.00, 31.20, 2.00)
        ldl = st.sidebar.slider('Colesterol de Lipoproteinas de Baja Densidad', 0.98, 15.33, 4.34)
        Adiposidad = st.sidebar.slider('Adiposidad', 6.74, 42.49, 26.12)
        Familia = st.sidebar.selectbox('Antecedentes Familiares de Enfermedad Cardiaca', ('Presente', 'Ausente'))
        Tipo = st.sidebar.slider('Tipo', 13, 78, 53)
        Obesidad = st.sidebar.slider('Obesidad', 14.70, 46.58, 25.80)
        Alcohol = st.sidebar.slider('Consumo Actual de Alcohol', 0.00, 147.19, 7.51)
        Edad = st.sidebar.slider('Edad', 15, 64, 45)

        # Creamos un diccionario con los datos ingresados por el usuario
        data = {'sbp': sbp,
                'Tabaco': Tabaco,
                'ldl': ldl,
                'Adiposidad': Adiposidad,
                'Familia': Familia,
                'Tipo': Tipo,
                'Obesidad': Obesidad,
                'Alcohol': Alcohol,
                'Edad': Edad,
                
                }
        
        # Convertimos el diccionario en un DataFrame
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

# Aplicamos el LabelEncoder para convertir la columna 'Familia' en valores numéricos
encoder = LabelEncoder()
input_df['Familia'] = encoder.fit_transform(input_df['Familia'])

# Seleccionamos solo la primera fila
input_df = input_df[:1]

st.subheader('Datos ingresados por el usuario')

# Mostramos los datos ingresados por el usuario en la página principal
if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('A la espera de que se cargue el archivo CSV. Actualmente usando parámetros de entrada de ejemplo (que se muestran a continuación).')
    st.write(input_df)

# Cargamos el modelo de clasificación previamente entrenado
load_clf = pickle.load(open('Heart.pkl', 'rb'))

# Aplicamos el modelo para realizar predicción con base a los datos ingresados
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

col1, col2 = st.columns(2)

with col1:
    st.subheader('Predicción')
    st.write(prediction)

with col2:
    st.subheader('Probabilidad de predicción')
    st.write(prediction_proba)

if prediction == 0:
    st.subheader('La persona no tiene problemas cardiacos')
else:
    st.subheader('La persona tiene problemas cardiacos')

st.markdown("""---""")
