import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,precision_score, recall_score
)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sktime.classification.interval_based import DrCIF
from sktime.transformations.panel.compose import ColumnConcatenator
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sktime.classification.sklearn import RotationForest
from aeon.classification.feature_based import Catch22Classifier

y_train = pd.read_csv("D:/advanced verification and validation/df_y_train.csv")
# x_train=x_train.drop(['filename'],axis=1)
y_train=y_train.values.reshape(-1,1)
y_train=np.squeeze(y_train)

y_test = pd.read_csv("D:/advanced verification and validation/df_y_test.csv")
# x_test=x_test.drop(x_test.iloc[:,3002],axis=1)
y_test=y_test.values.reshape(-1,1)

x_train=pd.read_csv("D:/advanced verification and validation/df_norm_x_train.csv")
x_train=x_train.values.reshape(4230,-1)

x_test=pd.read_csv("D:/advanced verification and validation/df_norm_x_test.csv")
x_test=x_test.values.reshape(1060,-1)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

########BaggingClassifier###################
bc = BaggingClassifier(base_estimator=None,
                       random_state=0)
bc.fit(x_train, y_train) #doctest:
bc_y_pred = bc.predict(x_test)
bc_y_proba_pred=bc.predict_proba(x_test)[:,1]
# bc_y_proba_pred.reshape(-1,1)
bc_Accuracy=balanced_accuracy_score(y_test, bc_y_pred)
bc_f1_score=f1_score(y_test,bc_y_pred)
auc_score_bc=roc_auc_score(y_test,bc_y_proba_pred)
print("bc_Accuracy:",bc_Accuracy)
print("bc_f1_score:", bc_f1_score)
print("bc_roc_auc:",auc_score_bc)

bc_res=[]
bc_res.append("BC")
bc_res.append(bc_Accuracy)
bc_res.append(bc_f1_score)
bc_res.append(auc_score_bc)
print("bc_res = ", bc_res)
# labels = ["No Steady state", "steady state"]
bc_cm = confusion_matrix(y_test, bc_y_pred)
print("bc_cm = ", bc_cm)
bc_disp=ConfusionMatrixDisplay(confusion_matrix=bc_cm, display_labels=None)
bc_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/bc_norm_cm.png")
# plt.show()
plt.close()

########BaggingClassifier Decision Tree###################
bc_DTree = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                       random_state=0)
bc_DTree.fit(x_train, y_train) #doctest:
bc_DTree_y_pred = bc_DTree.predict(x_test)
bc_DTree_y_pred_proba = bc_DTree.predict_proba(x_test)[:,1]
# bc_DTree_y_pred_proba.reshape(-1,1)
bc_DTree_Accuracy=balanced_accuracy_score(y_test, bc_DTree_y_pred)
bc_DTree_f1_score=f1_score(y_test,bc_DTree_y_pred)
auc_score_bc_DTree=roc_auc_score(y_test,bc_DTree_y_pred_proba)
print("bc_DTree_Accuracy:",bc_DTree_Accuracy)
print("bc_DTree_f1_score:", bc_DTree_f1_score)
print("bc_DTree_roc_auc:",auc_score_bc_DTree)

bc_DTree_res=[]
bc_DTree_res.append("BC_DTree")
bc_DTree_res.append(bc_DTree_Accuracy)
bc_DTree_res.append(bc_DTree_f1_score)
bc_DTree_res.append(auc_score_bc_DTree)
print("bc_DTree_res = ", bc_DTree_res)
# labels = ["No Steady state", "steady state"]
bc_DTree_cm = confusion_matrix(y_test, bc_DTree_y_pred)
print("bc_DTree_cm = ", bc_DTree_cm)
bc_DTree_disp=ConfusionMatrixDisplay(confusion_matrix=bc_DTree_cm, display_labels=None)
bc_DTree_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/bc_DTree_norm_cm.png")
# plt.show()
plt.close()

#############BalancedBaggingClassifier#####################
from imblearn.ensemble import BalancedBaggingClassifier
bbc = BalancedBaggingClassifier(n_estimators=50, random_state=0)
bbc.fit(x_train, y_train)
bbc_y_pred = bbc.predict(x_test)
bbc_y_pred_proba = bbc.predict_proba(x_test)[:,1]
# bbc_y_pred_proba.reshape(-1,1)
bbc_Accuracy=balanced_accuracy_score(y_test, bbc_y_pred)
bbc_f1_score=f1_score(y_test,bbc_y_pred)
auc_score_bbc=roc_auc_score(y_test,bbc_y_pred_proba)
print("bbc_Accuracy:",bbc_Accuracy)
print("bbc_f1_score:", bbc_f1_score)
print("bbc_roc_auc:",auc_score_bbc)

bbc_res=[]
bbc_res.append("BBC")
bbc_res.append(bbc_Accuracy)
bbc_res.append(bbc_f1_score)
bbc_res.append(auc_score_bbc)
print("bbc_resampled_res = ", bbc_res)
# labels = ["No Steady state", "steady state"]
bbc_cm = confusion_matrix(y_test, bbc_y_pred)
print("bbc_cm = ", bbc_cm)
bbc_disp=ConfusionMatrixDisplay(confusion_matrix=bbc_cm, display_labels=None)
bbc_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/bbc_norm_cm.png")
# plt.show()
plt.close()


#############BalancedBaggingClassifier Decision Tree#####################
from imblearn.ensemble import BalancedBaggingClassifier
bbc_Dec_Tree = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)
bbc_Dec_Tree.fit(x_train, y_train)
bbc_Dec_Tree_y_pred = bbc_Dec_Tree.predict(x_test)
bbc_Dec_Tree_y_pred_proba = bbc_Dec_Tree.predict_proba(x_test)[:,1]
# bbc_Dec_Tree_y_pred_proba.reshape(-1,1)
bbc_Dec_Tree_Accuracy=balanced_accuracy_score(y_test, bbc_Dec_Tree_y_pred)
bbc_Dec_Tree_f1_score=f1_score(y_test,bbc_Dec_Tree_y_pred)
auc_score_bbc_Dec_Tree=roc_auc_score(y_test,bbc_Dec_Tree_y_pred_proba)
print("bbc_Dec_Tree_Accuracy:",bbc_Dec_Tree_Accuracy)
print("bbc_Dec_Tree_f1_score:", bbc_Dec_Tree_f1_score)
print("bbc_Dec_Tree_roc_auc:",auc_score_bbc_Dec_Tree)

bbc_Dec_Tree_res=[]
bbc_Dec_Tree_res.append("BBC_Dec_Tree")
bbc_Dec_Tree_res.append(bbc_Dec_Tree_Accuracy)
bbc_Dec_Tree_res.append(bbc_Dec_Tree_f1_score)
bbc_Dec_Tree_res.append(auc_score_bbc_Dec_Tree)
print("bbc_Dec_Tree_res = ", bbc_Dec_Tree_res)
# labels = ["No Steady state", "steady state"]
bbc_Dec_Tree_cm = confusion_matrix(y_test, bbc_Dec_Tree_y_pred)
print("bbc_Dec_Tree_cm = ", bbc_Dec_Tree_cm)
bbc_Dec_Tree_disp=ConfusionMatrixDisplay(confusion_matrix=bbc_Dec_Tree_cm, display_labels=None)
bbc_Dec_Tree_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/bbc_Dec_Tree_norm_cm.png")
# plt.show()
plt.close()

##########EasyEnsembleClassifier###############
from imblearn.ensemble import EasyEnsembleClassifier
eec = EasyEnsembleClassifier(random_state=0)
eec.fit(x_train, y_train)
EEC_y_pred = eec.predict(x_test)
EEC_y_pred_proba = eec.predict_proba(x_test)[:,1]
# EEC_y_pred_proba.reshape(-1,1)
EEC_accuracy=balanced_accuracy_score(y_test, EEC_y_pred)
EEC_f1_score=f1_score(y_test,EEC_y_pred)
auc_score_EEC=roc_auc_score(y_test,EEC_y_pred_proba)
print("EE_Accuracy:",EEC_accuracy)
print("EE_f1_score:", EEC_f1_score)
print("EE_roc_auc:",auc_score_EEC)

EEC_res=[]
EEC_res.append("EEC")
EEC_res.append(EEC_accuracy)
EEC_res.append(EEC_f1_score)
EEC_res.append(auc_score_EEC)
print("DrCIF_res = ", EEC_res)
# labels = ["No Steady state", "steady state"]
EEC_cm = confusion_matrix(y_test, EEC_y_pred)
print("EEC_cm = ", EEC_cm)
EEC_disp=ConfusionMatrixDisplay(confusion_matrix=EEC_cm, display_labels=None)
EEC_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/EEC_norm_cm.png")
# plt.show()
plt.close()

#############RUSBoostClassifier##########
from imblearn.ensemble import RUSBoostClassifier
rusboost = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R',
                              random_state=0)
rusboost.fit(x_train, y_train)
rusboost_y_pred = rusboost.predict(x_test)
rusboost_y_pred_proba = rusboost.predict_proba(x_test)[:,1]
# rusboost_y_pred_proba.reshape(-1,1)
rusboost_f1_score=f1_score(y_test,rusboost_y_pred)
auc_score_rusboost=roc_auc_score(y_test,rusboost_y_pred_proba)
rusboost_accuracy=balanced_accuracy_score(y_test, rusboost_y_pred)
print("rusboost_Accuracy:",rusboost_accuracy)
print("rusboost_f1_score:", rusboost_f1_score)
print("rusboost_roc_auc:",auc_score_rusboost)

rusboost_res=[]
rusboost_res.append("rusboost")
rusboost_res.append(rusboost_accuracy)
rusboost_res.append(rusboost_f1_score)
rusboost_res.append(auc_score_rusboost)
print("rusboost_res = ", rusboost_res)
# labels = ["No Steady state", "steady state"]
rusboost_cm = confusion_matrix(y_test, rusboost_y_pred)
print("rusboost_cm = ", rusboost_cm)
rusboost_disp=ConfusionMatrixDisplay(confusion_matrix=rusboost_cm, display_labels=None)
rusboost_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/rusboost_norm_cm.png")
# plt.show()
plt.close()

############BalancedRandomForestClassifier############
from imblearn.ensemble import BalancedRandomForestClassifier
print("############BalancedRandomForestClassifier############")
BRF = BalancedRandomForestClassifier(sampling_strategy="all", replacement=True, max_depth=2, random_state=0)
BRF.fit(x_train,y_train)
BRF_y_pred=BRF.predict(x_test)
BRF_y_pred_proba=BRF.predict_proba(x_test)[:,1]
# BRF_y_pred_proba.reshape(-1,1)
BRF_ac=balanced_accuracy_score(y_test,BRF_y_pred)
BRF_f1_score=f1_score(y_test,BRF_y_pred)
auc_score_BRF=roc_auc_score(y_test,BRF_y_pred_proba)

print("BRF_y_pred", BRF_y_pred)
print("BRF accuracy : ", BRF_ac)
print("BRF F1 score : ", BRF_f1_score)
print("BRF AUC score : ", auc_score_BRF)

BRF_res=[]
BRF_res.append("BRF")
BRF_res.append(BRF_ac)
BRF_res.append(BRF_f1_score)
BRF_res.append(auc_score_BRF)
print("BRF result = ", BRF_res)

# labels = ["No Steady state", "steady state"]
BRF_cm = confusion_matrix(y_test, BRF_y_pred)
print("BRF_cm = ", BRF_cm)
BRF_disp=ConfusionMatrixDisplay(confusion_matrix=BRF_cm, display_labels=None)
BRF_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/BRF_norm_cm.png")
# plt.show()
plt.close()

#######rotation forest########
print("########### RotationForest ########")
rot_for_clf=RotationForest(n_estimators=10)
rot_for_clf.fit(x_train,y_train)
rot_for_y_pred=rot_for_clf.predict(x_test)
rot_for_y_pred_proba=rot_for_clf.predict_proba(x_test)[:,1]
# rot_for_y_pred_proba.reshape(-1,1)
rot_for_ac=balanced_accuracy_score(y_test,rot_for_y_pred)
rot_for_f1_score=f1_score(y_test,rot_for_y_pred)
auc_score_rot_for=roc_auc_score(y_test,rot_for_y_pred_proba)

print("rot_for_y_pred", rot_for_y_pred)
print("Rot_For accuracy : ", rot_for_ac)
print("Rot_For F1 score : ", rot_for_f1_score)
print("Rot_For AUC score : ", auc_score_rot_for)

rot_for_res=[]
rot_for_res.append("Rot_For")
rot_for_res.append(rot_for_ac)
rot_for_res.append(rot_for_f1_score)
rot_for_res.append(auc_score_rot_for)
print("Rot_For result = ", rot_for_res)

# labels = ["No Steady state", "steady state"]
rot_for_cm = confusion_matrix(y_test, rot_for_y_pred)
print("rot_for_cm = ", rot_for_cm)
rot_for_disp=ConfusionMatrixDisplay(confusion_matrix=rot_for_cm, display_labels=None)
rot_for_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/rot_for_norm_cm.png")
# plt.show()
plt.close()

#########DrCIF###########
print("###########DrCIF############")
# clf = ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
DrCIF=ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
DrCIF.fit(x_train, y_train)
DrCIF_y_pred = DrCIF.predict(x_test)
DrCIF_y_pred_proba = DrCIF.predict_proba(x_test)[:,1]
# DrCIF_y_pred_proba.reshape(-1,1)
DrCIF_accuracy=balanced_accuracy_score(y_test, DrCIF_y_pred)
DrCIF_f1_score=f1_score(y_test,DrCIF_y_pred)
auc_score_DrCIF=roc_auc_score(y_test,DrCIF_y_pred_proba)

print("DrCIF_y_pred", DrCIF_y_pred)
print("DrCIF_accuracy : ", DrCIF_accuracy)
print("DrCIF_f1_score : ", DrCIF_f1_score)
print("auc_score_DrCIF : ", auc_score_DrCIF)

DrCIF_res=[]
DrCIF_res.append("DrCIF")
DrCIF_res.append(DrCIF_accuracy)
DrCIF_res.append(DrCIF_f1_score)
DrCIF_res.append(auc_score_DrCIF)
print("DrCIF_res = ", DrCIF_res)
# labels = ["No Steady state", "steady state"]
DrCIF_cm = confusion_matrix(y_test, DrCIF_y_pred)
print("DrCIF_cm = ", DrCIF_cm)
DrCIF_disp=ConfusionMatrixDisplay(confusion_matrix=DrCIF_cm, display_labels=None)
DrCIF_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/DrCIF_norm_cm.png")
# plt.show()
plt.close()


##################Catch22Classifier##################
from aeon.classification.feature_based import Catch22Classifier
c22cls = Catch22Classifier()
c22cls.fit(x_train, y_train)
c22_preds = c22cls.predict(x_test)
c22_preds_proba = c22cls.predict_proba(x_test)[:,1]
# c22_preds_proba.reshape(-1,1)
c22_accuracy=balanced_accuracy_score(y_test, c22_preds)
c22_f1_score=f1_score(y_test,c22_preds)
auc_score_c22=roc_auc_score(y_test,c22_preds_proba)

print("c22_preds", c22_preds)
print("c22_accuracy : ", c22_accuracy)
print("c22_f1_score : ", c22_f1_score)
print("auc_score_c22 : ", auc_score_c22)

c22_res=[]
c22_res.append("c22")
c22_res.append(c22_accuracy)
c22_res.append(c22_f1_score)
c22_res.append(auc_score_c22)
print("c22_res = ", c22_res)
# labels = ["No Steady state", "steady state"]
c22_cm = confusion_matrix(y_test, c22_preds)
print("c22_cm = ", c22_cm)
c22_disp=ConfusionMatrixDisplay(confusion_matrix=c22_cm, display_labels=None)
c22_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/c22_norm_cm.png")
# plt.show()
plt.close()


####################RandomOverSampler###################

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(x_train, y_train)

###############bc_resampled###########
bc.fit(X_resampled, y_resampled) #doctest:
bc_resa_y_pred = bc.predict(x_test)
bc_resa_y_pred_proba = bc.predict_proba(x_test)[:,1]
# bc_resa_y_pred_proba.reshape(-1,1)
bc_resa_Accuracy=balanced_accuracy_score(y_test, bc_resa_y_pred)
bc_resa_f1_score=f1_score(y_test,bc_resa_y_pred)
auc_score_bc_resa=roc_auc_score(y_test,bc_resa_y_pred_proba)
print("bc_resa_Accuracy:",bc_resa_Accuracy)
print("bc_resa_f1_score:", bc_resa_f1_score)
print("auc_score_bc_resa:",auc_score_bc_resa)

bc_resa_res=[]
bc_resa_res.append("BC_resa")
bc_resa_res.append(bc_resa_Accuracy)
bc_resa_res.append(bc_resa_f1_score)
bc_resa_res.append(auc_score_bc_resa)
print("bc_resa_res = ", bc_resa_res)
# labels = ["No Steady state", "steady state"]
bc_resa_cm = confusion_matrix(y_test, bc_resa_y_pred)
print("bc_resa_cm = ", bc_resa_cm)
bc_resa_disp=ConfusionMatrixDisplay(confusion_matrix=bc_resa_cm, display_labels=None)
bc_resa_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/bc_resa_cm.png")
# plt.show()
plt.close()

###############bc_DTree_resampled###########
bc_DTree.fit(X_resampled, y_resampled) #doctest:
bc_DTree_resa_y_pred = bc_DTree.predict(x_test)
bc_DTree_resa_y_pred_proba = bc_DTree.predict_proba(x_test)[:,1]
# bc_DTree_resa_y_pred_proba.reshape(-1,1)
bc_DTree_resa_Accuracy=balanced_accuracy_score(y_test, bc_DTree_resa_y_pred)
bc_DTree_resa_f1_score=f1_score(y_test,bc_DTree_resa_y_pred)
auc_score_bc_DTree_resa=roc_auc_score(y_test,bc_DTree_resa_y_pred_proba)
print("bc_DTree_resa_Accuracy:",bc_DTree_resa_Accuracy)
print("bc_DTree_resa_f1_score:", bc_DTree_resa_f1_score)
print("auc_score_bc_DTree_resa:",auc_score_bc_DTree_resa)

bc_DTree_resa_res=[]
bc_DTree_resa_res.append("BC_DTree_resa")
bc_DTree_resa_res.append(bc_DTree_resa_Accuracy)
bc_DTree_resa_res.append(bc_DTree_resa_f1_score)
bc_DTree_resa_res.append(auc_score_bc_DTree_resa)
print("bc_DTree_resa_res = ", bc_DTree_resa_res)
# labels = ["No Steady state", "steady state"]
bc_DTree_resa_cm = confusion_matrix(y_test, bc_DTree_resa_y_pred)
print("bc_DTree_resa_cm = ", bc_DTree_resa_cm)
bc_DTree_resa_disp=ConfusionMatrixDisplay(confusion_matrix=bc_DTree_resa_cm, display_labels=None)
bc_DTree_resa_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/bc_DTree_resa_cm.png")
# plt.show()
plt.close()

#############BalancedBaggingClassifier#####################
from imblearn.ensemble import BalancedBaggingClassifier
# bbc = BalancedBaggingClassifier(random_state=42)
bbc.fit(X_resampled, y_resampled)
bbc_resampled_y_pred = bbc.predict(x_test)
bbc_resampled_y_pred_proba = bbc.predict_proba(x_test)[:,1]
# bbc_resampled_y_pred_proba.reshape(-1,1)
bbc_resampled_Accuracy=balanced_accuracy_score(y_test, bbc_resampled_y_pred)
bbc_resampled_f1_score=f1_score(y_test,bbc_resampled_y_pred)
auc_score_bbc_resampled=roc_auc_score(y_test,bbc_resampled_y_pred_proba)
print("bbc_Accuracy:",bbc_resampled_Accuracy)
print("bbc_f1_score:", bbc_resampled_f1_score)
print("bbc_roc_auc:",auc_score_bbc_resampled)

bbc_resampled_res=[]
bbc_resampled_res.append("BBC_resampled")
bbc_resampled_res.append(bbc_resampled_Accuracy)
bbc_resampled_res.append(bbc_resampled_f1_score)
bbc_resampled_res.append(auc_score_bbc_resampled)
print("bbc_resampled_res = ", bbc_resampled_res)
# labels = ["No Steady state", "steady state"]
bbc_resampled_cm = confusion_matrix(y_test, bbc_resampled_y_pred)
print("bbc_resampled_cm = ", bbc_resampled_cm)
bbc_resampled_disp=ConfusionMatrixDisplay(confusion_matrix=bbc_resampled_cm, display_labels=None)
bbc_resampled_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/bbc_resampled_norm_cm.png")
# plt.show()
plt.close()

#############BalancedBaggingClassifier Decision tree#####################
from imblearn.ensemble import BalancedBaggingClassifier
# bbc_DTree = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),sampling_strategy='auto',replacement=False,random_state=0)
bbc_Dec_Tree.fit(X_resampled, y_resampled)
bbc_Dec_Tree_resampled_y_pred = bbc_Dec_Tree.predict(x_test)
bbc_Dec_Tree_resampled_y_pred_proba = bbc_Dec_Tree.predict_proba(x_test)[:,1]
# bbc_Dec_Tree_resampled_y_pred_proba.reshape(-1,1)
bbc_Dec_Tree_resampled_Accuracy=balanced_accuracy_score(y_test, bbc_Dec_Tree_resampled_y_pred)
bbc_Dec_Tree_resampled_f1_score=f1_score(y_test,bbc_Dec_Tree_resampled_y_pred)
auc_score_bbc_Dec_Tree_resampled=roc_auc_score(y_test,bbc_Dec_Tree_resampled_y_pred_proba)
print("bbc_Dec_Tree_Accuracy:",bbc_Dec_Tree_resampled_Accuracy)
print("bbc_Dec_Tree_f1_score:", bbc_Dec_Tree_resampled_f1_score)
print("bbc_Dec_Tree_roc_auc:",auc_score_bbc_Dec_Tree_resampled)

bbc_Dec_Tree_resampled_res=[]
bbc_Dec_Tree_resampled_res.append("BBC_Dec_Tree_resampled")
bbc_Dec_Tree_resampled_res.append(bbc_Dec_Tree_resampled_Accuracy)
bbc_Dec_Tree_resampled_res.append(bbc_Dec_Tree_resampled_f1_score)
bbc_Dec_Tree_resampled_res.append(auc_score_bbc_Dec_Tree_resampled)
print("bbc_Dec_Tree_resampled_res = ", bbc_Dec_Tree_resampled_res)
# labels = ["No Steady state", "steady state"]
bbc_Dec_Tree_resampled_cm = confusion_matrix(y_test, bbc_Dec_Tree_resampled_y_pred)
print("bbc_Dec_Tree_resampled_cm = ", bbc_Dec_Tree_resampled_cm)
bbc_Dec_Tree_resampled_disp=ConfusionMatrixDisplay(confusion_matrix=bbc_Dec_Tree_resampled_cm, display_labels=None)
bbc_Dec_Tree_resampled_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/bbc_Dec_Tree_resampled_norm_cm.png")
# plt.show()
plt.close()

#############EasyEnsembleClassifier################
from imblearn.ensemble import EasyEnsembleClassifier
eec = EasyEnsembleClassifier(random_state=0)
eec.fit(X_resampled, y_resampled)
eec_resampled_y_pred = eec.predict(x_test)
eec_resampled_y_pred_proba = eec.predict_proba(x_test)[:,1]
# eec_resampled_y_pred_proba.reshape(-1,1)
EE_Accuracy=balanced_accuracy_score(y_test, eec_resampled_y_pred)
EE_f1_score=f1_score(y_test,eec_resampled_y_pred)
EE_roc_auc=roc_auc_score(y_test,eec_resampled_y_pred_proba)
print("EE_Accuracy:",EE_Accuracy)
print("EE_f1_score:", EE_f1_score)
print("EE_roc_auc:",EE_roc_auc)

eec_resampled_res=[]
eec_resampled_res.append("eec_resampled_res")
eec_resampled_res.append(EE_Accuracy)
eec_resampled_res.append(EE_f1_score)
eec_resampled_res.append(EE_roc_auc)
print("eec_resampled result = ", eec_resampled_res)

# labels = ["No Steady state", "steady state"]
eec_resampled_cm = confusion_matrix(y_test, eec_resampled_y_pred)
print("eec_resampled_cm = ", eec_resampled_cm)
eec_resampled_disp=ConfusionMatrixDisplay(confusion_matrix=eec_resampled_cm, display_labels=None)
eec_resampled_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/eec_resampled_norm_cm.png")
# plt.show()
plt.close()

#############RUSBoostClassifier##########
from imblearn.ensemble import RUSBoostClassifier
rusboost = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R',
                              random_state=0)
rusboost.fit(X_resampled, y_resampled)
rusboost_resampled_y_pred = rusboost.predict(x_test)
rusboost_resampled_y_pred_proba = rusboost.predict_proba(x_test)[:,1]
# rusboost_resampled_y_pred_proba.reshape(-1,1)
rusboost_resampled_f1_score=f1_score(y_test,rusboost_resampled_y_pred)
auc_score_rusboost_resampled=roc_auc_score(y_test,rusboost_resampled_y_pred_proba)
rusboost_resampled_accuracy=balanced_accuracy_score(y_test, rusboost_resampled_y_pred)
print("rusboost_resampled_Accuracy:",rusboost_resampled_accuracy)
print("rusboost_resampled_f1_score:", rusboost_resampled_f1_score)
print("rusboost_resampled_roc_auc:",auc_score_rusboost_resampled)

rusboost_resampled_res=[]
rusboost_resampled_res.append("rusboost")
rusboost_resampled_res.append(rusboost_resampled_accuracy)
rusboost_resampled_res.append(rusboost_resampled_f1_score)
rusboost_resampled_res.append(auc_score_rusboost_resampled)
print("rusboost_resampled_res = ", rusboost_resampled_res)
# labels = ["No Steady state", "steady state"]
rusboost_resampled_cm = confusion_matrix(y_test, rusboost_resampled_y_pred)
print("rusboost_resampled_cm = ", rusboost_resampled_cm)
rusboost_resampled_disp=ConfusionMatrixDisplay(confusion_matrix=rusboost_resampled_cm, display_labels=None)
rusboost_resampled_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/rusboost_resampled_norm_cm.png")
# plt.show()
plt.close()

############BalancedRandomForestClassifier############

BRF.fit(X_resampled, y_resampled)
BRF_resampled_y_pred=BRF.predict(x_test)
BRF_resampled_y_pred_proba=BRF.predict_proba(x_test)[:,1]
# BRF_resampled_y_pred_proba.reshape(-1,1)
BRF_resampled_ac=balanced_accuracy_score(y_test,BRF_resampled_y_pred)
BRF_resampled_f1_score=f1_score(y_test,BRF_resampled_y_pred)
auc_score_BRF_resampled=roc_auc_score(y_test,BRF_resampled_y_pred_proba)

print("BRF_resampled_y_pred", BRF_resampled_y_pred)
print("BRF_resampled accuracy : ", BRF_resampled_ac)
print("BRF_resampled F1 score : ", BRF_resampled_f1_score)
print("BRF_resampled AUC score : ", auc_score_BRF_resampled)

BRF_resampled_res=[]
BRF_resampled_res.append("BRF_resampled")
BRF_resampled_res.append(BRF_resampled_ac)
BRF_resampled_res.append(BRF_resampled_f1_score)
BRF_resampled_res.append(auc_score_BRF_resampled)
print("BRF_resampled result = ", BRF_resampled_res)

# labels = ["No Steady state", "steady state"]
BRF_resampled_cm = confusion_matrix(y_test, BRF_resampled_y_pred)
print("BRF_resampled_cm = ", BRF_resampled_cm)
BRF_resampled_disp=ConfusionMatrixDisplay(confusion_matrix=BRF_resampled_cm, display_labels=None)
BRF_resampled_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/BRF_resampled_norm_cm.png")
# plt.show()
plt.close()

#######rotation forest########
print("########### RotationForest ########")
rot_for_clf=RotationForest(n_estimators=10)
rot_for_clf.fit(X_resampled, y_resampled)
rot_for_resampled_y_pred=rot_for_clf.predict(x_test)
rot_for_resampled_y_pred_proba=rot_for_clf.predict_proba(x_test)[:,1]
# rot_for_resampled_y_pred_proba.reshape(-1,1)
rot_for_resampled_ac=balanced_accuracy_score(y_test,rot_for_resampled_y_pred)
rot_for_resampled_f1_score=f1_score(y_test,rot_for_resampled_y_pred)
auc_score_rot_for_resampled=roc_auc_score(y_test,rot_for_resampled_y_pred_proba)

print("rot_for_resampled_y_pred", rot_for_resampled_y_pred)
print("Rot_For_resampled accuracy : ", rot_for_resampled_ac)
print("Rot_For_resampled F1 score : ", rot_for_resampled_f1_score)
print("Rot_For_resampled AUC score : ", auc_score_rot_for_resampled)

rot_for_resampled_res=[]
rot_for_resampled_res.append("Rot_For_resampled")
rot_for_resampled_res.append(rot_for_resampled_ac)
rot_for_resampled_res.append(rot_for_resampled_f1_score)
rot_for_resampled_res.append(auc_score_rot_for_resampled)
print("Rot_For_resampled result = ", rot_for_resampled_res)

# labels = ["No Steady state", "steady state"]
rot_for_resampled_cm = confusion_matrix(y_test, rot_for_y_pred)
print("rot_for_resampled_cm = ", rot_for_resampled_cm)
rot_for_resampled_disp=ConfusionMatrixDisplay(confusion_matrix=rot_for_resampled_cm, display_labels=None)
rot_for_resampled_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/rot_for_resampled_norm_cm.png")
# plt.show()
plt.close()

#########DrCIF###########
print("###########DrCIF############")
# clf = ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
# DrCIF=ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
DrCIF.fit(X_resampled, y_resampled)
DrCIF_resampled_y_pred = DrCIF.predict(x_test)
DrCIF_resampled_y_pred_proba = DrCIF.predict_proba(x_test)[:,1]
# DrCIF_resampled_y_pred_proba.reshape(-1,1)
DrCIF_resampled_accuracy=balanced_accuracy_score(y_test, DrCIF_resampled_y_pred)
DrCIF_resampled_f1_score=f1_score(y_test,DrCIF_resampled_y_pred)
auc_score_DrCIF_resampled=roc_auc_score(y_test,DrCIF_resampled_y_pred_proba)

print("DrCIF_resampled_y_pred", DrCIF_resampled_y_pred)
print("DrCIF_resampled_accuracy : ", DrCIF_resampled_accuracy)
print("DrCIF_resampled_f1_score : ", DrCIF_resampled_f1_score)
print("auc_score_DrCIF_resampled : ", auc_score_DrCIF_resampled)

DrCIF_resampled_res=[]
DrCIF_resampled_res.append("DrCIF_resampled")
DrCIF_resampled_res.append(DrCIF_resampled_accuracy)
DrCIF_resampled_res.append(DrCIF_resampled_f1_score)
DrCIF_resampled_res.append(auc_score_DrCIF_resampled)
print("DrCIF_res = ", DrCIF_resampled_res)
# labels = ["No Steady state", "steady state"]
DrCIF_resampled_cm = confusion_matrix(y_test, DrCIF_resampled_y_pred)
print("DrCIF_resampled_cm = ", DrCIF_resampled_cm)
DrCIF_resampled_disp=ConfusionMatrixDisplay(confusion_matrix=DrCIF_resampled_cm, display_labels=None)
DrCIF_resampled_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/DrCIF_resampled_norm_cm.png")
# plt.show()
plt.close()

##################Catch22Classifier##################
from aeon.classification.feature_based import Catch22Classifier
c22cls = Catch22Classifier()
c22cls.fit(X_resampled, y_resampled)
c22_resampled_preds = c22cls.predict(x_test)
c22_resampled_preds_proba = c22cls.predict_proba(x_test)[:,1]
# c22_resampled_preds_proba.reshape(-1,1)
c22_resampled_accuracy=balanced_accuracy_score(y_test, c22_resampled_preds)
c22_resampled_f1_score=f1_score(y_test,c22_resampled_preds)
auc_score_c22_resampled=roc_auc_score(y_test,c22_resampled_preds_proba)

print("c22_resampled_preds", c22_resampled_preds)
print("c22_resampled_accuracy : ", c22_resampled_accuracy)
print("c22_resampled_f1_score : ", c22_resampled_f1_score)
print("auc_score_c22_resampled : ", auc_score_c22_resampled)

c22_resampled_res=[]
c22_resampled_res.append("c22_resampled")
c22_resampled_res.append(c22_resampled_accuracy)
c22_resampled_res.append(c22_resampled_f1_score)
c22_resampled_res.append(auc_score_c22_resampled)
print("c22_res = ", c22_resampled_res)
# labels = ["No Steady state", "steady state"]
c22_resampled_cm = confusion_matrix(y_test, c22_resampled_preds)
print("c22_cm = ", c22_resampled_cm)
c22_resampled_disp=ConfusionMatrixDisplay(confusion_matrix=c22_resampled_cm, display_labels=None)
c22_resampled_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/c22_resampled_norm_cm.png")
# plt.show()
plt.close()


result=[]
result.append(DrCIF_res)
result.append(rot_for_res)
result.append(rusboost_res)
result.append(EEC_res)
result.append(bc_res)
result.append(bbc_res)
result.append(bc_DTree_res)
result.append(bbc_Dec_Tree_res)
result.append(BRF_res)
result.append(c22_res)
result.append(DrCIF_resampled_res)
result.append(rot_for_resampled_res)
result.append(rusboost_resampled_res)
result.append(eec_resampled_res)
result.append(bc_resa_res)
result.append(bbc_resampled_res)
result.append(bc_DTree_resa_res)
result.append(bbc_Dec_Tree_resampled_res)
result.append(BRF_resampled_res)
result.append(c22_resampled_res)
print("#####result#####", result)
result_df=pd.DataFrame(result,columns=['Clf Name','Accuracy','F1 Score','ROC score'])
print(result_df)

result_df.to_csv("D:/master thesis/result_norm_df.csv",index=False)

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, DrCIF_y_pred, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, rot_for_y_pred, pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, rusboost_y_pred, pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(y_test, EEC_y_pred, pos_label=1)
fpr5, tpr5, thresh5 = roc_curve(y_test, bc_y_pred, pos_label=1)
fpr6, tpr6, thresh6 = roc_curve(y_test, bbc_y_pred, pos_label=1)

fpr7, tpr7, thresh7 = roc_curve(y_test, c22_preds, pos_label=1)
fpr8, tpr8, thresh8 = roc_curve(y_test, DrCIF_resampled_y_pred, pos_label=1)
fpr9, tpr9, thresh9 = roc_curve(y_test, rot_for_resampled_y_pred, pos_label=1)
fpr10, tpr10, thresh10 = roc_curve(y_test, rusboost_resampled_y_pred, pos_label=1)
fpr11, tpr11, thresh11 = roc_curve(y_test, eec_resampled_y_pred, pos_label=1)
fpr12, tpr12, thresh12 = roc_curve(y_test, bc_resa_y_pred, pos_label=1)
fpr13, tpr13, thresh13 = roc_curve(y_test, bbc_resampled_y_pred, pos_label=1)
fpr14, tpr14, thresh14 = roc_curve(y_test, c22_resampled_preds, pos_label=1)
fpr15, tpr15, thresh15 = roc_curve(y_test, bc_DTree_y_pred, pos_label=1)
fpr16, tpr16, thresh16 = roc_curve(y_test, bbc_Dec_Tree_y_pred, pos_label=1)
fpr17, tpr17, thresh17 = roc_curve(y_test, bc_DTree_resa_y_pred, pos_label=1)
fpr18, tpr18, thresh18 = roc_curve(y_test, bbc_Dec_Tree_resampled_y_pred, pos_label=1)
fpr19, tpr19, thresh19 = roc_curve(y_test, BRF_y_pred, pos_label=1)
fpr20, tpr20, thresh20 = roc_curve(y_test, BRF_resampled_y_pred, pos_label=1)


# roc curve for tpr = fpr
random_probs = [0 for n in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='DrCIF')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='rot_for')
plt.plot(fpr3, tpr3, linestyle='--',color='red', label='rusboost')
plt.plot(fpr4, tpr4, linestyle='--',color='yellow', label='EEC')
plt.plot(fpr5, tpr5, linestyle='--',color='violet', label='bc')
plt.plot(fpr6, tpr6, linestyle='--',color='brown', label='bbc')
plt.plot(fpr7, tpr7, linestyle='--',color='pink', label='c22')
plt.plot(fpr8, tpr8, linestyle='--',color='black', label='DrCIF_resampled')
plt.plot(fpr9, tpr9, linestyle='--',color='purple', label='rot_for_resampled')
plt.plot(fpr10, tpr10, linestyle='--',color='maroon', label='rusboost_resampled')
plt.plot(fpr11, tpr11, linestyle='--',color='cyan', label='EEC_resampled')
plt.plot(fpr12, tpr12, linestyle='--',color='blue', label='bc_resampled')
plt.plot(fpr13, tpr13, linestyle='--',color='red', label='bbc_resampled')
plt.plot(fpr14, tpr14, linestyle='--',color='magenta', label='c22_resampled')
plt.plot(fpr15, tpr15, linestyle='--',color='purple', label='bc_DTree')
plt.plot(fpr16, tpr16, linestyle='--',color='maroon', label='bbc_Dtree')
plt.plot(fpr17, tpr17, linestyle='--',color='cyan', label='BC_Dtree_resampled')
plt.plot(fpr18, tpr18, linestyle='--',color='blue', label='bbc_Dtree_resampled')
plt.plot(fpr19, tpr19, linestyle='--',color='red', label='BRF')
plt.plot(fpr20, tpr20, linestyle='--',color='magenta', label='BRF_resampled')

plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
# plt.legend()
# plt.legend(loc='best')
plt.savefig('D:/master thesis/nor_ROC.png',dpi=300)
# plt.show()
plt.close()
