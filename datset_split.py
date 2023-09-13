import pandas as pd
from sklearn.model_selection import train_test_split

'''''
####dataset concatenation########
####used code######

count=0
for n in range(0,529):
    df=pd.read_csv(f"D:/advanced verification and validation/df_{count}.csv",index_col=False)
    df1=pd.read_csv(f"D:/advanced verification and validation/df1_{count}.csv")
    df_con=pd.concat([df,df1],axis=1,ignore_index=False)
    df_con.to_csv(f"D:/advanced verification and validation/dataset/df_{count}.csv",index=False)
    count=count+1
'''
'''''
########used code########
########test train split#########
###########80:20 for each benchmark#############
count=0

x_con_train=[]
x_con_test=[]
y_con_train=[]
y_con_test=[]
X_train_all=pd.DataFrame()
X_test_all=pd.DataFrame()
y_train_all=pd.DataFrame()
y_test_all=pd.DataFrame()

for n in range(0,529):
    ds=pd.read_csv(f"D:/advanced verification and validation/dataset/df_{count}.csv")
    ds['forks'] = [1 if i == "steady state" else 0 for i in ds['forks']]
    y=ds['forks']
    x=ds.drop(['forks','run','steady_state_starts'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    print("*********X_train*********",X_train)
    print("*********X_test*********",X_test)
    print("*********y_train**********",y_train)
    print("**********y_test********",y_test)
    count=count+1

    X_train_all=X_train_all.append(X_train,ignore_index=True)
    X_test_all=X_test_all.append(X_test,ignore_index=True)
    y_train_all=y_train_all.append(y_train,ignore_index=True)
    y_test_all=y_test_all.append(y_test,ignore_index=True)

    print("*********X_train_all********",X_train_all)
    print("*********X_test_all********",X_test_all)
    print("*********y_train_all********",y_train_all)
    print("*********y_test_all********",y_test_all)

X_train_all.to_csv(f"D:/master thesis/dataset/train/x_con_train.csv",index=False)
X_test_all.to_csv(f"D:/master thesis/dataset/test/x_con_test.csv",index=False)
y_train_all.to_csv(f"D:/master thesis/dataset/train/y_con_train.csv",index=False)
y_test_all.to_csv(f"D:/master thesis/dataset/test/y_con_test.csv",index=False)
'''
'''''
X_train=pd.read_csv(f"D:/master thesis/dataset/train/x_con_train.csv")
X_test=pd.read_csv(f"D:/master thesis/dataset/test/x_con_test.csv")
y_train=pd.read_csv(f"D:/master thesis/dataset/train/y_con_train.csv")
y_test=pd.read_csv(f"D:/master thesis/dataset/test/y_con_test.csv")
import numpy as np
nor_x_train=X_train/np.linalg.norm(X_train, axis=1, keepdims=True)
print(nor_x_train)
nor_x_train.to_csv(f"D:/master thesis/dataset/train/df_norm_x_con_train.csv",index=False)

nor_x_test=X_test/np.linalg.norm(X_test, axis=1, keepdims=True)
print(nor_x_test)
nor_x_test.to_csv(f"D:/master thesis/dataset/test/df_norm_x_con_test.csv",index=False)
'''

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
y_train=pd.read_csv(f"D:/master thesis/dataset/train/y_con_train.csv")
y_test=pd.read_csv(f"D:/master thesis/dataset/test/y_con_test.csv")

y_train=y_train.values.reshape(-1,1)
y_train=np.squeeze(y_train)

y_test=y_test.values.reshape(-1,1)

nor_x_train=pd.read_csv(f"D:/master thesis/dataset/train/df_norm_x_con_train.csv")
nor_x_train=nor_x_train.values.reshape(4232,-1)

nor_x_test=pd.read_csv(f"D:/master thesis/dataset/test/df_norm_x_con_test.csv")
nor_x_test=nor_x_test.values.reshape(1058,-1)

print(nor_x_train.shape,y_train.shape,nor_x_test.shape,y_test.shape)

########BaggingClassifier###################
bc = BaggingClassifier(base_estimator=None,
                       random_state=0)
bc.fit(nor_x_train, y_train) #doctest:
bc_y_pred = bc.predict(nor_x_test)
bc_y_pred_proba = bc.predict_proba(nor_x_test)[:,1]
bc_Accuracy=balanced_accuracy_score(y_test, bc_y_pred)
bc_f1_score=f1_score(y_test,bc_y_pred)
auc_score_bc=roc_auc_score(y_test,bc_y_pred)
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
plt.savefig("D:/master thesis/dataset/dataset_split/bc_norm_cm.png")
# plt.show()
plt.close()

########BaggingClassifier Decision Tree###################
bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                       random_state=0)
bc.fit(nor_x_train, y_train) #doctest:
bc_y_pred = bc.predict(nor_x_test)
bc_y_pred_proba = bc.predict_proba(nor_x_test)[:,1]
bc_Accuracy=balanced_accuracy_score(y_test, bc_y_pred)
bc_f1_score=f1_score(y_test,bc_y_pred)
auc_score_bc=roc_auc_score(y_test,bc_y_pred_proba)
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
plt.savefig("D:/master thesis/dataset/dataset_split/bc_norm_cm.png")
# plt.show()
plt.close()

########BaggingClassifier Decision Tree###################
bc_DTree = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                       random_state=0)
bc_DTree.fit(nor_x_train, y_train) #doctest:
bc_DTree_y_pred = bc_DTree.predict(nor_x_test)
bc_DTree_y_pred_proba = bc_DTree.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/bc_DTree_norm_cm.png")
# plt.show()
plt.close()

#############BalancedBaggingClassifier#####################
from imblearn.ensemble import BalancedBaggingClassifier
bbc = BalancedBaggingClassifier(n_estimators=50, random_state=0)
bbc.fit(nor_x_train, y_train)
bbc_y_pred = bbc.predict(nor_x_test)
bbc_y_pred_proba = bbc.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/bbc_norm_cm.png")
# plt.show()
plt.close()

#############BalancedBaggingClassifier Decision Tree#####################
from imblearn.ensemble import BalancedBaggingClassifier
bbc_DTree = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)
bbc_DTree.fit(nor_x_train, y_train)
bbc_DTree_y_pred = bbc_DTree.predict(nor_x_test)
bbc_DTree_y_pred_proba = bbc_DTree.predict_proba(nor_x_test)[:,1]
bbc_DTree_Accuracy=balanced_accuracy_score(y_test, bbc_DTree_y_pred)
bbc_DTree_f1_score=f1_score(y_test,bbc_DTree_y_pred)
auc_score_bbc_DTree=roc_auc_score(y_test,bbc_DTree_y_pred_proba)
print("bbc_DTree_Accuracy:",bbc_DTree_Accuracy)
print("bbc_DTree_f1_score:", bbc_DTree_f1_score)
print("bbc_DTree_roc_auc:",auc_score_bbc_DTree)

bbc_DTree_res=[]
bbc_DTree_res.append("BBC_DTree")
bbc_DTree_res.append(bbc_DTree_Accuracy)
bbc_DTree_res.append(bbc_DTree_f1_score)
bbc_DTree_res.append(auc_score_bbc_DTree)
print("bbc_DTree_res = ", bbc_DTree_res)
# labels = ["No Steady state", "steady state"]
bbc_DTree_cm = confusion_matrix(y_test, bbc_DTree_y_pred)
print("bbc_DTree_cm = ", bbc_DTree_cm)
bbc_DTree_disp=ConfusionMatrixDisplay(confusion_matrix=bbc_DTree_cm, display_labels=None)
bbc_DTree_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/dataset/dataset_split/bbc_DTree_norm_cm.png")
# plt.show()
plt.close()

##########EasyEnsembleClassifier###############
from imblearn.ensemble import EasyEnsembleClassifier
eec = EasyEnsembleClassifier(random_state=0)
eec.fit(nor_x_train, y_train)
EEC_y_pred = eec.predict(nor_x_test)
EEC_y_pred_proba = eec.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/EEC_norm_cm.png")
# plt.show()
plt.close()

#############RUSBoostClassifier##########
from imblearn.ensemble import RUSBoostClassifier
rusboost = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R',
                              random_state=0)
rusboost.fit(nor_x_train, y_train)
rusboost_y_pred = rusboost.predict(nor_x_test)
rusboost_y_pred_proba = rusboost.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/rusboost_norm_cm.png")
# plt.show()
plt.close()

############BalancedRandomForestClassifier############
from imblearn.ensemble import BalancedRandomForestClassifier
print("############BalancedRandomForestClassifier############")
BRF = BalancedRandomForestClassifier(sampling_strategy="all", replacement=True, max_depth=2, random_state=0)
BRF.fit(nor_x_train,y_train)
BRF_y_pred=BRF.predict(nor_x_test)
BRF_y_pred_proba=BRF.predict_proba(nor_x_test)[:,1]
BRF_ac=balanced_accuracy_score(y_test,BRF_y_pred)
BRF_f1_score=f1_score(y_test,BRF_y_pred)
auc_score_BRF=roc_auc_score(y_test,BRF_y_pred)

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
plt.savefig("D:/master thesis/dataset/dataset_split/BRF_norm_cm.png")
# plt.show()
plt.close()

#######rotation forest########
print("########### RotationForest ########")
rot_for_clf=RotationForest(n_estimators=10)
rot_for_clf.fit(nor_x_train,y_train)
rot_for_y_pred=rot_for_clf.predict(nor_x_test)
rot_for_y_pred_proba=rot_for_clf.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/rot_for_norm_cm.png")
# plt.show()
plt.close()

#########DrCIF###########
print("###########DrCIF############")
# clf = ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
DrCIF=ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
DrCIF.fit(nor_x_train, y_train)
DrCIF_y_pred = DrCIF.predict(nor_x_test)
DrCIF_y_pred_proba = DrCIF.predict_proba(nor_x_test)[:,1]

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
plt.savefig("D:/master thesis/dataset/dataset_split/DrCIF_norm_cm.png")
# plt.show()
plt.close()

##################Catch22Classifier##################
from aeon.classification.feature_based import Catch22Classifier
c22cls = Catch22Classifier()
c22cls.fit(nor_x_train, y_train)
c22_preds = c22cls.predict(nor_x_test)
c22_preds_proba = c22cls.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/c22_norm_cm.png")
# plt.show()
plt.close()


####################RandomOverSampler###################

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(nor_x_train, y_train)


###############bc_resampled###########
bc.fit(X_resampled, y_resampled) #doctest:
bc_resa_y_pred = bc.predict(nor_x_test)
bc_resa_y_pred_proba = bc.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/bc_resa_cm.png")
# plt.show()
plt.close()

########BaggingClassifier Decision Tree###################
# bc_DTree = BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0)
bc_DTree.fit(X_resampled, y_resampled) #doctest:
bc_DTree_resampled_y_pred = bc_DTree.predict(nor_x_test)
bc_DTree_resampled_y_pred_proba = bc_DTree.predict_proba(nor_x_test)[:,1]
bc_DTree_resampled_Accuracy=balanced_accuracy_score(y_test, bc_DTree_resampled_y_pred)
bc_DTree_resampled_f1_score=f1_score(y_test,bc_DTree_resampled_y_pred)
auc_score_bc_DTree_resampled=roc_auc_score(y_test,bc_DTree_resampled_y_pred_proba)
print("bc_DTree_resample_dAccuracy:",bc_DTree_resampled_Accuracy)
print("bc_DTree_resampled_f1_score:", bc_DTree_resampled_f1_score)
print("bc_DTree_resampled_roc_auc:",auc_score_bc_DTree_resampled)

bc_DTree_resampled_res=[]
bc_DTree_resampled_res.append("BC_DTree_resampled")
bc_DTree_resampled_res.append(bc_DTree_resampled_Accuracy)
bc_DTree_resampled_res.append(bc_DTree_resampled_f1_score)
bc_DTree_resampled_res.append(auc_score_bc_DTree_resampled)
print("bc_DTree_resampled_res = ", bc_DTree_resampled_res)
# labels = ["No Steady state", "steady state"]
bc_DTree_resampled_cm = confusion_matrix(y_test, bc_DTree_resampled_y_pred)
print("bc_resampled_cm = ", bc_DTree_resampled_cm)
bc_DTree_resample_disp=ConfusionMatrixDisplay(confusion_matrix=bc_DTree_resampled_cm, display_labels=None)
bc_DTree_resample_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/dataset/dataset_split/bc_DTree_resampled_norm_cm.png")
# plt.show()
plt.close()

#############BalancedBaggingClassifier#####################
from imblearn.ensemble import BalancedBaggingClassifier
# bbc = BalancedBaggingClassifier(random_state=42)
bbc.fit(X_resampled, y_resampled)
bbc_resampled_y_pred = bbc.predict(nor_x_test)
bbc_resampled_y_pred_proba = bbc.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/bbc_resampled_norm_cm.png")
# plt.show()
plt.close()

#############BalancedBaggingClassifier Decision tree#####################
from imblearn.ensemble import BalancedBaggingClassifier
# bbc_DTree = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),sampling_strategy='auto',replacement=False,random_state=0)
bbc_DTree.fit(X_resampled, y_resampled)
bbc_Dec_Tree_resampled_y_pred = bbc_DTree.predict(nor_x_test)
bbc_Dec_Tree_resampled_y_pred_proba = bbc_DTree.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/bbc_Dec_Tree_resampled_norm_cm.png")
# plt.show()
plt.close()

#############EasyEnsembleClassifier################
from imblearn.ensemble import EasyEnsembleClassifier
eec = EasyEnsembleClassifier(random_state=0)
eec.fit(X_resampled, y_resampled)
eec_resampled_y_pred = eec.predict(nor_x_test)
eec_resampled_y_pred_proba = eec.predict_proba(nor_x_test)[:,1]
EEc_resampled_Accuracy=balanced_accuracy_score(y_test, eec_resampled_y_pred)
EEc_resampled_f1_score=f1_score(y_test,eec_resampled_y_pred)
EEc_resampled_roc_auc=roc_auc_score(y_test,eec_resampled_y_pred_proba)
print("EE_Accuracy:",EEc_resampled_Accuracy)
print("EE_f1_score:", EEc_resampled_f1_score)
print("EE_roc_auc:",EEc_resampled_roc_auc)


eec_resampled_res=[]
eec_resampled_res.append("eec_resampled_res")
eec_resampled_res.append(EEc_resampled_Accuracy)
eec_resampled_res.append(EEc_resampled_f1_score)
eec_resampled_res.append(EEc_resampled_roc_auc)
print("eec_resampled result = ", eec_resampled_res)

# labels = ["No Steady state", "steady state"]
eec_resampled_cm = confusion_matrix(y_test, eec_resampled_y_pred)
print("eec_resampled_cm = ", eec_resampled_cm)
eec_resampled_disp=ConfusionMatrixDisplay(confusion_matrix=eec_resampled_cm, display_labels=None)
eec_resampled_disp.plot()
# plt.show()
plt.savefig("D:/master thesis/dataset/dataset_split/eec_resampled_norm_cm.png")
# plt.show()
plt.close()

#############RUSBoostClassifier##########
from imblearn.ensemble import RUSBoostClassifier
rusboost = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R',
                              random_state=0)
rusboost.fit(X_resampled, y_resampled)
rusboost_resampled_y_pred = rusboost.predict(nor_x_test)
rusboost_resampled_y_pred_proba = rusboost.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/rusboost_resampled_norm_cm.png")
# plt.show()
plt.close()

############BalancedRandomForestClassifier############

BRF.fit(X_resampled, y_resampled)
BRF_resampled_y_pred=BRF.predict(nor_x_test)
BRF_resampled_y_pred_proba=BRF.predict_proba(nor_x_test)[:,1]
BRF_resampled_ac=balanced_accuracy_score(y_test,BRF_resampled_y_pred)
BRF_resampled_f1_score=f1_score(y_test,BRF_resampled_y_pred)
auc_score_BRF_resampled=roc_auc_score(y_test,BRF_resampled_y_pred_proba)

print("BRF_resampled_y_pred", BRF_resampled_y_pred)
print("BRF_resampled accuracy : ", BRF_resampled_ac)
print("BRF_resampled F1 score : ", BRF_resampled_f1_score)
print("BRF_resampled AUC score : ", auc_score_BRF_resampled)

BRF_resampled_res=[]
BRF_resampled_res.append("BRF")
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
plt.savefig("D:/master thesis/dataset/dataset_split/BRF_resampled_norm_cm.png")
# plt.show()
plt.close()

#######rotation forest########
print("########### RotationForest ########")
rot_for_clf=RotationForest(n_estimators=10)
rot_for_clf.fit(X_resampled, y_resampled)
rot_for_resampled_y_pred=rot_for_clf.predict(nor_x_test)
rot_for_resampled_y_pred_proba=rot_for_clf.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/rot_for_resampled_norm_cm.png")
# plt.show()
plt.close()

#########DrCIF###########
print("###########DrCIF############")
# clf = ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
# DrCIF=ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
DrCIF.fit(X_resampled, y_resampled)
DrCIF_resampled_y_pred = DrCIF.predict(nor_x_test)
DrCIF_resampled_y_pred_proba = DrCIF.predict_proba(nor_x_test)[:,1]

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
plt.savefig("D:/master thesis/dataset/dataset_split/DrCIF_resampled_norm_cm.png")
# plt.show()
plt.close()

##################Catch22Classifier##################
from aeon.classification.feature_based import Catch22Classifier
c22cls = Catch22Classifier()
c22cls.fit(X_resampled, y_resampled)
c22_resampled_preds = c22cls.predict(nor_x_test)
c22_resampled_preds_proba = c22cls.predict_proba(nor_x_test)[:,1]
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
plt.savefig("D:/master thesis/dataset/dataset_split/c22_resampled_norm_cm.png")
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
result.append(bbc_DTree_res)
result.append(BRF_res)
result.append(c22_res)
result.append(DrCIF_resampled_res)
result.append(rot_for_resampled_res)
result.append(rusboost_resampled_res)
result.append(eec_resampled_res)
result.append(bc_resa_res)
result.append(bbc_resampled_res)
result.append(bc_DTree_resampled_res)
result.append(bbc_Dec_Tree_resampled_res)
result.append(BRF_resampled_res)
result.append(c22_resampled_res)
print("#####result#####", result)
result_df=pd.DataFrame(result,columns=['Clf Name','Accuracy','F1 Score','ROC score'])
print(result_df)

result_df.to_csv("D:/master thesis/dataset/dataset_split/result_data_split_norm_df.csv",index=False)

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
fpr16, tpr16, thresh16 = roc_curve(y_test, bbc_DTree_y_pred, pos_label=1)
fpr17, tpr17, thresh17 = roc_curve(y_test, bc_DTree_resampled_y_pred, pos_label=1)
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
plt.savefig('D:/master thesis/dataset/dataset_split/nor_ROC.png',dpi=300)
plt.show()





