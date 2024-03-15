import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cargamos el conjunto de datos de flores Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos un clasificador KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenamos el clasificador
knn.fit(X_train, y_train)

# Hacemos predicciones en el conjunto de prueba
predictions = knn.predict(X_test)

# Calculamos la precisión
accuracy = accuracy_score(y_test, predictions)
print("Precisión del clasificador KNN:", accuracy)

 
 

