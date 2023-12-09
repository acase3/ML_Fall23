import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1 == EQ, 0 == noise

df_ypn = pd.read_csv("EQ_y_prednf.csv") # predicted class for noise
df_ypeq = pd.read_csv("EQ_y_predf.csv") # predicted class for eq
df_yrn = pd.read_csv("EQ_y_realnf.csv") # real class for noise
df_yreq = pd.read_csv("EQ_y_realf.csv") # real class for eq

ypn = np.array(df_ypn)
ypn = ypn[:,1]
ypeq = np.array(df_ypeq)
ypeq = ypeq[:,1]
yrn = np.array(df_yrn)
yrn = yrn[:,1]
yreq = np.array(df_yreq)
yreq = yreq[:,1]

sum_ypn = np.sum(ypn)
sum_ypeq = np.sum(ypeq)
sum_yrn = np.sum(yrn)
sum_yreq = np.sum(yreq)

print('EQs predicted= ' +str(sum_ypeq))
print('actual number of EQs= ' +str(sum_yreq))

print('EQs predicted in noise= ' +str(sum_ypn))
print('actual number of EQs= ' +str(sum_yrn))

TrueTrue = sum_ypeq
FalseFalse = int(len(ypn) - sum_ypn)
Falsepos = sum_ypn
Falseneg = sum_yreq - sum_ypeq
cf_matrix = [[TrueTrue,Falsepos],[Falseneg, FalseFalse]]
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
plt.title('Confusion Matrix for STA/LTA', fontsize=18)
plt.show()