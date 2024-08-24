import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('C:/Users/mohel/Desktop/TELUQ/ProjetFinal/alzheimers_disease_data.csv')
# Exploration des données
print('Aperçu des données: ')
print(data)
print('Dimensions des données: ')
print(data.shape)
print('Type des données et vérifications des valeurs null ')
print(data.info())


# Vérification des valeurs dupliqués et les valeurs non null pour toutes les colonnes

print(data.duplicated().sum())
print(data.isna().sum())

# Suppression des deux colonnes PatientID, DoctorInCharge
# l'identifiant du patient ne seriva à rien dans nos analyses
# la colonne DoctorInCharge est la même pour tous les patients, valeur non utile pour les analyses


data.drop(columns=['PatientID', 'DoctorInCharge'], axis=1, inplace=True)
print(data.info()) # 33 colonnes au lieu de 35


# Exploration de la variable à prédire (diagnostis)
print(data['Diagnosis'].value_counts())
sns.histplot(x='Diagnosis', data=data)
plt.show()
# données des 760 patients qui ont le diagnostics de la maladie
diagnosis_data = data[data['Diagnosis'] == 1]

# définition de la fontion pour la visualitions des données
def plot(column, dataframe=diagnosis_data):
    plt.figure(figsize=(8, 4))
    sns.histplot(dataframe[column], kde=True)
    plt.title(f'Distribution de {column}')
    plt.show()






# Visualisatrion de la distribution colonne Age
plot("Age")
# Visualisatrion de la distribution colonne Gender
plot("Gender")
# Visualisatrion de la distribution colonne Ethnicity
plot("Ethnicity")
# Visualisatrion de la distribution colonne Smoking
plot("Smoking")

# Visualisation de l'enseble des colonnes à la fois
data.hist(figsize=(20,20))
plt.show()

# identification des colonnes numérique et celle catégorique pour fins de séparation et analyse

numerical_columns = [col for col in data.columns if data[col].nunique() > 5]
categorical_columns = data.columns.difference(numerical_columns).difference(["Diagnosis"]).to_list()

print("Numerical cols:", len(numerical_columns)) # 15 colonnes
print("Categorical cols:", len(categorical_columns)) # 17 colonnes



# génération d'une palettes de 5 couleurs pour la visualisation
palette = sns.color_palette('husl', 5)

# mettre des valeurs significatifs au lieu de 1 et 0 pour les colonnes catégories
custom_labels = {
    'Gender': ['Male', 'Female'],
    'Ethnicity': ['Caucasian', 'African American', 'Asian', 'Other'],
    'EducationLevel': ['None', 'High School', 'Bachelor\'s', 'Higher'],
    'Smoking': ['No', 'Yes'],
    'FamilyHistoryAlzheimers': ['No', 'Yes'],
    'CardiovascularDisease': ['No', 'Yes'],
    'Diabetes': ['No', 'Yes'],
    'Depression': ['No', 'Yes'],
    'HeadInjury': ['No', 'Yes'],
    'Hypertension': ['No', 'Yes'],
    'MemoryComplaints': ['No', 'Yes'],
    'BehavioralProblems': ['No', 'Yes'],
    'Confusion': ['No', 'Yes'],
    'Disorientation': ['No', 'Yes'],
    'PersonalityChanges': ['No', 'Yes'],
    'DifficultyCompletingTasks': ['No', 'Yes'],
    'Forgetfulness': ['No', 'Yes']
}


# Plot colonnes catégoriques
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x=col, palette=palette, legend=False)

    # Ajouter les étiquettes/label
    labels = custom_labels[col]
    ticks = range(len(labels))
    plt.xticks(ticks=ticks, labels=labels)

    plt.show()  # 17 plot


categories = [0, 1]
counts = data.Diagnosis.value_counts().tolist()

# Couleurs
colors = sns.color_palette("muted")

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Distribution du diagnostic')
plt.show()

# Define colors palette
colors = sns.color_palette('pastel')[0:5]

# Create subplots with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(6, 6))

# Pie chart for Age
data['bins'] = pd.cut(data['Age'], bins=[60, 69, 79, 90], labels=["60-69", "70-79", "80-90"])
axs[0, 0].pie(data.groupby('bins').size(), labels=data.groupby('bins').size().index, colors=colors, autopct='%.0f%%',
              radius=0.8)
axs[0, 0].set_title("Age")

# Pie chart for Gender
axs[0, 1].pie(data['Gender'].value_counts(), labels=['Female', 'Male'], colors=colors, autopct='%.0f%%', radius=0.8)
axs[0, 1].set_title("Gender")

# Pie chart for Ethnicity
axs[1, 0].pie(data['Ethnicity'].value_counts(), labels=['Caucasian', 'African-American', 'Other', 'Asian'],
              colors=colors, autopct='%.0f%%', radius=0.8)
axs[1, 0].set_title("Ethnicity")

# Pie chart for Educational Level
axs[1, 1].pie(data['EducationLevel'].value_counts(), startangle=45,
              labels=['High School', "Bachelor's", 'None', 'Higher'], colors=colors, autopct='%.0f%%', radius=0.8)
axs[1, 1].set_title("Educational Level")
# Remove the 'bins' column from the data
data.drop(['bins'], axis=1, inplace=True)

# Adjust layout and display
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5, linecolor="pink")
plt.title("Matrice de corrélation")
plt.show()

