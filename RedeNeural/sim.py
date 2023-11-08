# %%
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
import tensorflow as tf
import pandas as pd

# %%
#Dataset

fifa17 = pd.read_csv("datasets/regression/CLEAN_FIFA17_official_data.csv")
fifa18 = pd.read_csv("datasets/regression/CLEAN_FIFA18_official_data.csv")
fifa19 = pd.read_csv("datasets/regression/CLEAN_FIFA19_official_data.csv")
fifa20 = pd.read_csv("datasets/regression/CLEAN_FIFA20_official_data.csv")
fifa21 = pd.read_csv("datasets/regression/CLEAN_FIFA21_official_data.csv")
fifa22 = pd.read_csv("datasets/regression/CLEAN_FIFA22_official_data.csv")

# Adicionar a Coluna "Year" ao dataset
  
fifa17["Year"] = 2017
fifa18["Year"] = 2018
fifa19["Year"] = 2019
fifa20["Year"] = 2020
fifa21["Year"] = 2021
fifa22["Year"] = 2022

df = pd.concat([fifa17,fifa18,fifa19,fifa20,fifa21,fifa22])
df = df.sample(frac=0.2, random_state=42)
# df = fifa17

df.head()

# %%
X = df.drop(columns=["Overall"], axis=1)
y = df["Overall"]

# %%
#TRATAMENTO DOS DADOS

# Ver a taxa de correlação entre Overall e os outros atributos

corr_overall = df.corr(numeric_only=True)['Overall'].sort_values(ascending=False)
top_20 = corr_overall.nlargest(21).drop(["Overall"])
excluir_features = ["Best Overall Rating","Potential","Value(£)","Wage(£)","Release Clause(£)","International Reputation"]

# Pegar as colunas das features mais correlacionandas ao nosso objetivo

features_ideais = []
for feature in top_20.index:
    if feature not in excluir_features:
        features_ideais.append(feature)

        

# %%
"""
Então fazendos um heatmap para ver se as features estão bem correlacionadas entre sí 
"""
import matplotlib.pyplot as plt
import seaborn as sns


# Colocamos o feature "Overall" devolta na nossas features para podermos fazer o heatmap
selected_attributes_overall=features_ideais
selected_attributes_overall_test = features_ideais+["ID"]
selected_attributes_overall_corr = df[selected_attributes_overall].corr()

# E Por fim plotamos o heatmap para os atributos relacionados com "Overall"
plt.figure(figsize=(8, 6))
sns.heatmap(selected_attributes_overall_corr, annot=True, cmap='coolwarm', linewidth=0.5)
plt.title('Heatmap da Correlação de atributos ao Overall')
plt.show()

# %%
X_ima = X[selected_attributes_overall_test]
X = X[selected_attributes_overall]

# %%
X

# %%
y

# %%
# Load the Boston Housing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_full_train, X_validation, y_full_train, y_validation = train_test_split(X_ima, y, random_state=42)

# %%
X_train

# %%
X_test

# %%
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
# Build the model
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, InputLayer
from keras.initializers import glorot_uniform
# Build the model

model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1],)))
model.add(Dense(64,activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer, since it's a regression task, no activation is needed


# %%

optimizer = keras.optimizers.SGD(learning_rate=3e-3)
model.compile(optimizer=optimizer, loss='mse')


# %%
# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=1)

# %%

# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test)
print(f"Mean Squared Error: {mse}")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
X_new = X_test[5:10]
y_preds = model.predict(X_new)
y_preds.round(2)

# %%
X_new

# %%
X_test_ima[5:10]

# %%
y_test[5:10]



# %%