import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_noise = pd.read_csv("waveforms_noisenonnorm.csv")
df_EQ = pd.read_csv("waveforms_unnorm.csv")
#df_EQ_ex = pd.read_csv("waveforms_extra.csv")
print('done!')

df_noisef = df_noise.iloc[:19235]
df_EQf = df_EQ.iloc[21000:40235]
#subset = df_EQ_ex.iloc[100:2850]
frames = [df_EQf, df_noisef]

df_waves =  pd.concat(frames)
from sklearn.model_selection import train_test_split


df_waves = df_waves.drop(df_waves.columns[0], axis=1)
df_waves.rename(columns={"511": "class"}, inplace=True)
df_waves.describe()



from sklearn.model_selection import train_test_split

df_train_VAL, df_test = train_test_split(df_waves, test_size=0.2, random_state=2023)
df_train_VAL = df_train_VAL.dropna()
df_test = df_test.dropna()
df_train, df_val= train_test_split(df_train_VAL, test_size=0.25, random_state=1234)

y_train_VAL = np.array(df_train_VAL['class'])
y_test = np.array(df_test['class'])
y_train = np.array(df_train['class'])
y_val = np.array(df_val['class'])
df_train = df_train.drop(['class'], axis=1)
df_test = df_test.drop(['class'], axis=1)
df_val = df_val.drop(['class'], axis=1)
df_train_VAL = df_train_VAL.drop(['class'], axis=1)
print(df_train.shape)
print(df_test.shape)

X_train = np.array(df_train)
X_test = np.array(df_test)
X_train_VAL = np.array(df_train_VAL)
X_val = np.array(df_val)
print(y_train[1])

X_train = np.array(df_train)
X_test = np.array(df_test)
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
print(y_train[1])
y_train_onehot    = onehot.fit_transform(y_train.reshape(-1,1)).toarray()
y_test_onehot     = onehot.fit_transform(y_test.reshape(-1,1)).toarray()
y_val_onehot    = onehot.fit_transform(y_val.reshape(-1,1)).toarray()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
print(y_train_onehot[1,:])
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
X_val = scaler.transform(X_val)



#print(np.round( np.cumsum(pca.explained_variance_ratio_[0:20]), 4))

n_components = 80
pca = PCA(n_components=n_components)
pca.fit(X_train)

Xpca_train = pca.transform(X_train)
Xpca_test = pca.transform(X_test)
Xpca_val = pca.transform(X_val)
from sklearn.decomposition import PCA

#pca = PCA(n_components=100)
#pca.fit(X_train)
#plt.figure(figsize=(12,4))
#plt.plot(np.arange(1,101), np.cumsum(pca.explained_variance_ratio_), marker='.')
#plt.ylim([0,1])
#plt.show()

print(np.round( np.cumsum(pca.explained_variance_ratio_[0:50]), 4))
#do a grid search for tree depth and learning rate, number of trees delt with by early stopping function


from sklearn.metrics import accuracy_score


from sklearn.model_selection import cross_val_score
#xgb_classifier = xgb.XGBClassifier(n_estimators=10000, objective='binary:logistic', tree_method='hist', eta=(.01),
 #                                        max_depth=5, early_stopping_rounds=5, enable_categorical=True)
#xgb_classifier.fit(Xpca_train, y_train_onehot, eval_set=[(Xpca_val, y_val_onehot)])
#nfolds = 5
#x = cross_val_score(xgb_classifier, Xpca_train, y_train_onehot, cv=nfolds, verbose=1)
#print(x)

import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#
#Creating an XGBoost classifier
model = xgb.XGBClassifier(n_estimators = 2000, early_stopping_rounds=5, max_depth = 10, eta = 0.05, objective="binary:logistic")
nfolds = 10
#Training the model on the training data
model.fit(Xpca_train, y_train_onehot, eval_set=[(Xpca_val, y_val_onehot)])
print(model.best_iteration)
#val_scores = cross_val_score(model, Xpca_train, y_train_onehot, cv=nfolds)
#tree_val_score = np.mean(val_scores)
#print(tree_val_score)
#Making predictions on the test set
predictions = model.predict(Xpca_test)

#Calculating accuracy
accuracy = accuracy_score(y_test_onehot, predictions)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test_onehot, predictions))
TrueTrue = 0
FalseFalse = 0
Falsepos = 0
Falseneg = 0
from sklearn.metrics import confusion_matrix

for a in np.arange(0,len(y_test_onehot)):
    if y_test_onehot[a,0] == predictions[a,0] and predictions[a,0] == 1:
        TrueTrue = TrueTrue +1
    if y_test_onehot[a,0] != predictions[a,0] and predictions[a,0] == 1:
        Falseneg = Falseneg +1
    if y_test_onehot[a,0] == predictions[a,0] and predictions[a,0] == 0:
        FalseFalse = FalseFalse +1
    if y_test_onehot[a,0] != predictions[a,0] and predictions[a,0] == 0:
        Falsepos = Falsepos +1

conf_matrix = np.array([[TrueTrue, Falseneg], [Falsepos, FalseFalse]])

print(conf_matrix)
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix for XGBoost', fontsize=18)
plt.show()

