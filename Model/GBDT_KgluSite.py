
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import pylab as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import pandas as pd
from IPython.display import display
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import warnings;warnings.filterwarnings('ignore')
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from xgboost.sklearn import XGBClassifier #sklearn xgboost
from catboost import CatBoostClassifier
from imblearn.under_sampling import TomekLinks
from collections import Counter

Mn_S = MinMaxScaler(feature_range=(0, 1))
# #
EAACN = np.load('./EAAC581N.npy')
print(EAACN.shape)
EAACP = np.load('./EAAC581P.npy')
StrucN = np.load('StructureN.npy')
StrucP = np.load('StructureP.npy')
CKSAAPN = np.load('./CKSAAPN.npy')
CKSAAPP = np.load('./CKSAAPP.npy')
CTDCN = np.load('./CTDCN.npy')
CTDCP = np.load('./CTDCP.npy')
one_hotN = np.load('one_hotN1.npy')
# one_hotN = np.load('one_hotNall.npy')
# [m,n,p] = one_hotN.shape
# one_hotN = one_hotN.reshape(m,n*p)
one_hotP = np.load('one_hotP1.npy')
Blosum62N = np.load('BLOSUM62N_new.npy')
# Blosum62N = np.load('BLOSUM62Nall.npy')
Blosum62P = np.load('BLOSUM62P_new.npy')

PSSMN = np.load('./PSSMN1.npy')
PSSMP = np.load('./PSSMP1.npy')
CKSAAPN = CKSAAPN[:,0:484]
CKSAAPP =CKSAAPP[:,0:484]

Neg = np.hstack((EAACN,StrucN,Blosum62N,CKSAAPN,one_hotN,CTDCN,PSSMN))
Pos = np.hstack((EAACP,StrucP,Blosum62P,CKSAAPP,one_hotP,CTDCP,PSSMP))

(m,n)=Neg.shape
tag1 =(np.zeros(Neg.shape[0]).astype(int)).reshape(Neg.shape[0],1)
tag2 =(np.ones(Pos.shape[0]).astype(int)).reshape(Pos.shape[0],1)
y = np.vstack((tag1,tag2))
X = np.vstack((Neg,Pos))

(x_train,x_test,y_train,y_test) = train_test_split(X, y, test_size=0.25, random_state=666)
print(x_train.shape)
print(x_test.shape)
######imbalance
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
cc = NearMiss(version=3)
x_train, y_train = cc.fit_resample(x_train, y_train)
print(x_train.shape)
print(Counter(y_train))
#################
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
x_train=Mn_S.fit_transform(x_train)
x_test=Mn_S.transform(x_test)

# # # # ##嵌入式特征选择基于Elastic Net
from sklearn.linear_model import Ridge,LassoCV,ElasticNet,Lasso
model = ElasticNet(alpha=0.0000001,normalize=True)   #ACC=86%
model.fit(x_train, y_train)
mask = model.coef_ != 0  #返回一个相关系数是否为零的布尔数组
x_train = x_train[:, mask]  #返回相关系数非零的数据
x_test = x_test[:,mask]

with open('GBDT_KgluSite.pickle', 'rb') as f:
 rf = pickle.load(f)


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
