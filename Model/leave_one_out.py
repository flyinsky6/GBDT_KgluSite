from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

X = np.load('x_trainElastic_new.npy')
y = np.load('y_train.npy')

model = GradientBoostingClassifier()
parameters = {'max_depth': [3], 'n_estimators': [100]}

# Define LOO
loo = LeaveOneOut()

# Define grid search
grid_search = GridSearchCV(model, parameters, scoring='accuracy', cv=loo)

# Fit model
grid_search.fit(X, y)

rf = grid_search.best_estimator_

x_test = np.load('x_testElastic_new.npy')
y_test = np.load('y_test.npy')
y_score = rf.predict_proba(x_test)
fpr,tpr,threshold=roc_curve(y_test,y_score[:, 1])
roc_auc=auc(fpr,tpr)

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
 lw=2, label='ROC curve (AUC = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
#
y_pred = rf.predict(x_test)
print('testscore:'+str(rf.score(x_test,y_test)))
Acc = accuracy_score(y_test,y_pred)
Re = recall_score(y_test,y_pred)
Pre = precision_score(y_test,y_pred)
F1 = f1_score(y_test,y_pred)
MCC = matthews_corrcoef(y_test,y_pred)
Metrics = [Acc, Pre, Re, F1, roc_auc]
print('Acc = %.4f' % Acc)
print('Re = %.4f' % Re)
print('Pre = %.4f' % Pre)
print('F1 = %.4f' % F1)
print('MCC = %.4f' % MCC)
print('roc_auc = %.4f' % roc_auc)

