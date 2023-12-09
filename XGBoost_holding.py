depth = np.arange(1,7,1)
learning_rate = np.linspace(.001, 0.8, 10)
lr_iter = np.arange(1,10,1)
val_scores = np.zeros((depth.shape[0], learning_rate.shape[0]))
import xgboost as xgb
for i, d in enumerate(depth):
  for j, lr in enumerate(lr_iter):
      xgb_classifier = xgb.XGBClassifier(n_estimators=2000, tree_method='hist', eta=(learning_rate[lr]),
                                         max_depth=d, early_stopping_rounds=5, enable_categorical=True)
      xgb_classifier.fit(Xpca_train, y_train_onehot, eval_set=[(Xpca_val, y_val_onehot)])
      val_scores[i,j] = accuracy_score(y_val_onehot, xgb_classifier.predict(Xpca_val))
#objective='binary:logistic'

tmpx, tmpy = np.meshgrid(depth, learning_rate)
print(tmpy)
plt.figure(figsize=(12,6))
plt.pcolor(tmpx, tmpy, val_scores.transpose())
plt.title('XGboost tree depth and learning rate w/ early stopping')
plt.xlabel('depth')
plt.ylabel('learning_rate')
cbar = plt.colorbar()
cbar.set_label('validation accuracy')
plt.show()