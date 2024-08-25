from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, \
    accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv1D, MaxPooling1D
from sklearn.utils.class_weight import compute_class_weight


# fonction qui calcule la matrice de confusion, F1, ACCURACY, RECALL ET PRECISION
def test_metrics(model, name):
    #Test Performance
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}:")
    print("Matrice de confusion:")
    print(confusion_matrix(y_test, y_pred))
    f1 = f1_score(y_pred=y_pred, y_true=y_test)
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    recall = recall_score(y_pred=y_pred, y_true=y_test)
    precision = precision_score(y_pred=y_pred, y_true=y_test)

    return {"Classifier": name, "Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1": f1}

# Préparation des données
data = pd.read_csv('C:/Users/mohel/Desktop/TELUQ/ProjetFinal/alzheimers_disease_data.csv')
data.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)
# séparation des jeu de données
X = data.drop('Diagnosis', axis=1)
# variable à prédire
y = data['Diagnosis']
# données d'entrainement et données de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
len(X_train), len(X_test)

# Le jeu de données étant petit, nous allons utiliser SMOTE
smote = SMOTE(sampling_strategy="minority")
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_smote)
y_train = y_train_smote
X_test = scaler.transform(X_test)

# Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(),
    "SVC": SVC(kernel='linear', probability=True),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}
# Affichage des résultats pour chaque modèle
results = []
for name, model in models.items():
    results.append(test_metrics(model, name))

results_df = pd.DataFrame(results, columns=["Classifier","Accuracy","Recall","Precision","F1"])
print(results_df)


# Suite au metrique précédante, nous nous focaliserons sur le model Gradient Boosting
# Choix de l'ajustement des parametres

params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_gb_model = grid_search.best_estimator_
print("Les meilleurs parametres pour ce modele:", grid_search.best_params_)

# Modele avec les meilleurs parametres
model_ideal = GradientBoostingClassifier(learning_rate=0.01, max_depth=4, n_estimators=300, subsample=0.8, random_state=42)

# Entrainement
model_ideal.fit(X_train, y_train)

# Prédication
y_test_pred = model_ideal.predict(X_test)

# évaluation du modele
print("Évaluation du modele choisi:")
print(f"Accuracy = {accuracy_score(y_test, y_test_pred):.4f}")
print("Matrice de confusion:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print("Statistiques:")
print(classification_report(y_test, y_test_pred))
plt.figure(figsize=(10, 7))
plt.title("valeurs réelles vs valeurs prédites")
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("valeurs prédites")
plt.ylabel("valeurs réelles")



# Réseau de neuronnes

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

model.add(Flatten())


model.add(Dense(40, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu')),
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
resultat = model.fit(X_train, y_train, epochs=100)

print(f"Évaluation : {model.evaluate(X_test, y_test)}")
y_pred = model.predict(X_test)
y_pred = [1 if i > 0.5 else 0 for i in y_pred]
print(f"Accuracy : {accuracy_score(y_test, y_pred)}") # 84%

# Le modele de classification Gradient bossting reste plus performent de tous les autres modeles avec Accuracy de 95%