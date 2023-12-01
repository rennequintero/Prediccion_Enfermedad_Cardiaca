# Importamos la biblioteca pandas
import pandas as pd

# Leemos el dataset
data = pd.read_csv('Heart1.csv')

# Importamos la clase LabelEncoder de la biblioteca scikit-learn
from sklearn.preprocessing import LabelEncoder

# Creamos una instancia de LabelEncoder
encoder = LabelEncoder()

# Convertimos la columna 'FamHist' en valores numéricos utilizando el LabelEncoder
data['Familia'] = encoder.fit_transform(data['Familia'])

# Separamos las características (x) y la variable objetivo (y)
x = data.drop('chd', axis=1)
y = data['chd']

# Importamos la función train_test_split de la biblioteca scikit-learn
from sklearn.model_selection import train_test_split

# Dividimos el conjunto de datos en conjuntos de Entrenamiento y Prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Importamos la clase RandomForestClassifier de la biblioteca scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Creamos una instancia de RandomForestClassifier
clf = RandomForestClassifier()

# Entrenamos el modelo de clasificación RandomForest
clf.fit(x_train, y_train)

# Importamos la biblioteca pickle para guardar y cargar modelos entrenados
import pickle

# Guardamos el modelo entrenado en un archivo
pickle.dump(clf, open('Heart.pkl', 'wb'))
