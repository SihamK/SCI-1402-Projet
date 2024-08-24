from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, \
    accuracy_score



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
# variable à prédir
y = data['Diagnosis']
# données d'entrainement et données de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(),
    "SVC": SVC(kernel='linear', probability=True),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}
# Affichage des résultats pour chaque modèle
results = []
for name, model in models.items():
    results.append(test_metrics(model, name))

results_df = pd.DataFrame(results, columns=["Classifier","Accuracy","Recall","Precision","F1"])
print(results_df)



