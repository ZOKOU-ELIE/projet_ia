import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Chargement des données Iris (on filtre pour une classification binaire)
iris = load_iris()
X = iris.data
y = iris.target

# Garder uniquement les classes 0 et 1 (binaire)
X = X[y != 2]
y = y[y != 2]

# 2. Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Séparation en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Création du modèle avec une fonction sigmoïde en sortie
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(X.shape[1],)),  # Couche cachée
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sortie avec sigmoïde
])

# 5. Compilation du modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. Entraînement du modèle
model.fit(X_train, y_train, epochs=100, verbose=0)

# 7. Évaluation sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Précision sur les données de test : {accuracy:.2f}")

# 8. Prédiction d’un exemple
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)
print(f"Sortie sigmoïde : {prediction[0][0]:.4f}")
print("Classe prédite :", "Classe 1" if prediction[0][0] >= 0.5 else "Classe 0")