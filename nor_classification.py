import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import SGDClassifier

x_train = pd.read_csv("D:/advanced verification and validation/df_x_train.csv")
y_train = pd.read_csv("D:/advanced verification and validation/df_y_train.csv")
# x_train=x_train.drop(['filename'],axis=1)
y_train=y_train.values.reshape(-1,1)

x_test = pd.read_csv("D:/advanced verification and validation/df_x_test.csv")
y_test = pd.read_csv("D:/advanced verification and validation/df_y_test.csv")
# x_test=x_test.drop(x_test.iloc[:,3002],axis=1)
y_test=y_test.values.reshape(-1,1)

print(x_train)
nor_x_train=x_train/np.linalg.norm(x_train, axis=1, keepdims=True)
print(nor_x_train)
nor_x_train.to_csv("D:/advanced verification and validation/df_norm_x_train.csv",index=False)

print(x_test)
nor_x_test=x_test/np.linalg.norm(x_test, axis=1, keepdims=True)
print(nor_x_test)
nor_x_test.to_csv("D:/advanced verification and validation/df_norm_x_test.csv",index=False)

###### SVC ############
print("###### SVC ############")
svc_model = SVC(kernel='linear', random_state=0)

svc_model.fit(nor_x_train, y_train)

svc_y_pred=svc_model.predict(nor_x_test)

print("svc_y_pred = ", svc_y_pred)

# labels = ["No Steady state", "steady state"]
svc_cm = confusion_matrix(y_test, svc_y_pred)
print("svc confusion matrix = ", svc_cm)
svc_disp=ConfusionMatrixDisplay(confusion_matrix=svc_cm, display_labels=None)
svc_disp.plot()
# plt.show()
plt.savefig("D:/advanced verification and validation/svc_norm_cm.png")
plt.show()
plt.close()

svc_score=accuracy_score(y_test,svc_y_pred)
svc_f1 = f1_score(svc_y_pred, y_test, average="weighted")

print("svc accuracy_score : ",svc_score)
print("svc F1 score : ", svc_f1)

# auc scores
auc_score_svc = roc_auc_score(y_test, svc_y_pred)
print("auc_score1", auc_score_svc)

svc_res=[]
svc_res.append("SVC")
svc_res.append(svc_score)
svc_res.append(svc_f1)
svc_res.append(auc_score_svc)
print("svc_res = ", svc_res)

# calculate ROC scores
# ns_auc = roc_auc_score(testy, ns_probs)
# lr_auc = roc_auc_score(testy, lr_probs)

###### Naive Bayes ###########
print("###### Naive Bayes ###########")
# Build a Gaussian Classifier
nb_model = GaussianNB()

# Model training
nb_model.fit(nor_x_train, y_train)

# Predict Output
nb_y_pred = nb_model.predict(nor_x_test)
print("Actual Value:", y_test)
print("Predicted Value:", nb_y_pred)

nb_accuracy = accuracy_score(nb_y_pred, y_test)
nb_f1 = f1_score(nb_y_pred, y_test, average="weighted")

print("NB Accuracy:", nb_accuracy)
print("NB F1 Score:", nb_f1)
auc_score_nb = roc_auc_score(y_test, nb_y_pred)
print("auc_score1", auc_score_nb)

# labels = ["No Steady state", "steady state"]
nb_cm = confusion_matrix(y_test, nb_y_pred)
print("nb_cm = ", nb_cm)
nb_disp=ConfusionMatrixDisplay(confusion_matrix=nb_cm, display_labels=None)
nb_disp.plot()
# plt.show()
plt.savefig("D:/advanced verification and validation/NB_norm_cm.png")
plt.show()
plt.close()

NB_res=[]
NB_res.append("NB")
NB_res.append(nb_accuracy)
NB_res.append(nb_f1)
NB_res.append(auc_score_nb)
print("NB_res = ", NB_res)
######### RandomForestClassifier ###############
print("######### RandomForestClassifier ###############")
rf = RandomForestClassifier()
rf.fit(nor_x_train, y_train)
rf_y_pred = rf.predict(nor_x_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("rf Accuracy:", rf_accuracy)
rf_f1=f1_score(y_test,rf_y_pred, average="weighted")
print("rf F1 score : ", rf_f1)

rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)
print("rf precision : ", rf_precision)
print("rf_recall : ", rf_recall)
auc_score_rf = roc_auc_score(y_test, rf_y_pred)
print("auc_score1", auc_score_rf)

# Create the confusion matrix
rf_cm = confusion_matrix(y_test, rf_y_pred)
print("rf_cm = ", rf_cm)
ConfusionMatrixDisplay(confusion_matrix=rf_cm).plot()
# plt.show()
plt.savefig("D:/advanced verification and validation/RF_norm_cm.png")
plt.show()
plt.close()

RF_res=[]
RF_res.append("RF")
RF_res.append(rf_accuracy)
RF_res.append(rf_f1)
RF_res.append(auc_score_rf)
print("RF_res = ", RF_res)
####### KNN ###########
print("####### KNN ###########")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(nor_x_train, y_train)
knn_y_pred = knn.predict(nor_x_test)

knn_cm = confusion_matrix(y_test, knn_y_pred)
knn_ac = accuracy_score(y_test,knn_y_pred)
knn_f1_score=f1_score(y_test,knn_y_pred, average="weighted")
print("knn accuracy : ", knn_ac)
print("knn F1 score : ", knn_f1_score)
print("knn cm = ", knn_cm)
auc_score_knn = roc_auc_score(y_test, knn_y_pred)
print("auc_score1", auc_score_knn)

ConfusionMatrixDisplay(confusion_matrix=knn_cm).plot()
# plt.show()
plt.savefig("D:/advanced verification and validation/KNN_norm_cm.png")
plt.show()
plt.close()

knn_res=[]
knn_res.append("KNN")
knn_res.append(knn_ac)
knn_res.append(knn_f1_score)
knn_res.append(auc_score_knn)
print("knn_res = ", knn_res)

######Logistic Regression########
print("###### Logistic Regression ########")
lor_clf=LogisticRegression(random_state=0).fit(nor_x_train,y_train)
lor_score=lor_clf.score(nor_x_test,y_test)
lor_y_pred=lor_clf.predict(nor_x_test)
lor_ac=accuracy_score(y_test,lor_y_pred)
lor_f1_score=f1_score(y_test,lor_y_pred)
auc_score_lor=roc_auc_score(y_test,lor_y_pred)

print("Logistic Regression Score : ", lor_score)
print("Logistic Regression accuracy : ", lor_ac)
print("Logistic Regression F1 score : ", lor_f1_score)
print("Logistic Regression AUC score : ", auc_score_lor)

lor_res=[]
lor_res.append("LoR")
lor_res.append(lor_ac)
lor_res.append(lor_f1_score)
lor_res.append(auc_score_lor)
print("LoR result = ", lor_res)

# labels = ["No Steady state", "steady state"]
lor_cm = confusion_matrix(y_test, lor_y_pred)
print("lor_cm = ", lor_cm)
lor_disp=ConfusionMatrixDisplay(confusion_matrix=lor_cm, display_labels=None)
lor_disp.plot()
# plt.show()
plt.savefig("D:/advanced verification and validation/LoR_norm_cm.png")
plt.show()
plt.close()

###########Decision Tree########
print("########### Decision Tree ########")
dt_clf=DecisionTreeClassifier().fit(nor_x_train,y_train)
dt_score=dt_clf.score(nor_x_test,y_test)
dt_y_pred=dt_clf.predict(nor_x_test)
dt_ac=accuracy_score(y_test,dt_y_pred)
dt_f1_score=f1_score(y_test,dt_y_pred)
auc_score_dt=roc_auc_score(y_test,dt_y_pred)

print("Decision Tree Score : ", dt_score)
print("Decision Tree accuracy : ", dt_ac)
print("Decision Tree F1 score : ", dt_f1_score)
print("Decision Tree AUC score : ", auc_score_dt)

dt_res=[]
dt_res.append("DT")
dt_res.append(dt_ac)
dt_res.append(dt_f1_score)
dt_res.append(auc_score_dt)
print("DT result = ", dt_res)

# labels = ["No Steady state", "steady state"]
dt_cm = confusion_matrix(y_test, dt_y_pred)
print("dt_cm = ", dt_cm)
dt_disp=ConfusionMatrixDisplay(confusion_matrix=dt_cm, display_labels=None)
dt_disp.plot()
# plt.show()
plt.savefig("D:/advanced verification and validation/DT_norm_cm.png")
plt.show()
plt.close()

###########XG Boost########
print("########### XG Boost ########")
xgb_clf=XGBClassifier().fit(nor_x_train,y_train)
xgb_score=xgb_clf.score(nor_x_test,y_test)
xgb_y_pred=xgb_clf.predict(nor_x_test)
xgb_ac=accuracy_score(y_test,xgb_y_pred)
xgb_f1_score=f1_score(y_test,xgb_y_pred)
auc_score_xgb=roc_auc_score(y_test,xgb_y_pred)

print("XG Boost Score : ", xgb_score)
print("XG Boost accuracy : ", xgb_ac)
print("XG Boost F1 score : ", xgb_f1_score)
print("XG Boost AUC score : ", auc_score_xgb)

xgb_res=[]
xgb_res.append("XGB")
xgb_res.append(xgb_ac)
xgb_res.append(xgb_f1_score)
xgb_res.append(auc_score_xgb)
print("XGB result = ", xgb_res)

# labels = ["No Steady state", "steady state"]
xgb_cm = confusion_matrix(y_test, xgb_y_pred)
print("xgb_cm = ", xgb_cm)
xgb_disp=ConfusionMatrixDisplay(confusion_matrix=xgb_cm, display_labels=None)
xgb_disp.plot()
# plt.show()
plt.savefig("D:/advanced verification and validation/XGB_norm_cm.png")
plt.show()
plt.close()

###########SGD########
print("########### SGD ########")
sgd_clf=SGDClassifier().fit(nor_x_train,y_train)
sgd_score=sgd_clf.score(nor_x_test,y_test)
sgd_y_pred=sgd_clf.predict(nor_x_test)
sgd_ac=accuracy_score(y_test,sgd_y_pred)
sgd_f1_score=f1_score(y_test,sgd_y_pred)
auc_score_sgd=roc_auc_score(y_test,sgd_y_pred)

print("SGD Score : ", sgd_score)
print("SGD accuracy : ", sgd_ac)
print("SGD F1 score : ", sgd_f1_score)
print("SGD AUC score : ", auc_score_sgd)

sgd_res=[]
sgd_res.append("SGD")
sgd_res.append(sgd_ac)
sgd_res.append(sgd_f1_score)
sgd_res.append(auc_score_sgd)
print("SGD result = ", sgd_res)

# labels = ["No Steady state", "steady state"]
sgd_cm = confusion_matrix(y_test, sgd_y_pred)
print("xgb_cm = ", sgd_cm)
sgd_disp=ConfusionMatrixDisplay(confusion_matrix=sgd_cm, display_labels=None)
sgd_disp.plot()
# plt.show()
plt.savefig("D:/advanced verification and validation/SGD_norm_cm.png")
plt.show()
plt.close()

result=[]
result.append(svc_res)
result.append(NB_res)
result.append(RF_res)
result.append(knn_res)
result.append(lor_res)
result.append(dt_res)
result.append(xgb_res)
result.append(sgd_res)
print("#####result#####", result)
result_df=pd.DataFrame(result,columns=['Clf Name','Accuracy','F1 Score','ROC score'])
print(result_df)

result_df.to_csv("D:/advanced verification and validation/result_norm_df.csv",index=False)

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, svc_y_pred, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, nb_y_pred, pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, rf_y_pred, pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(y_test, knn_y_pred, pos_label=1)
fpr5, tpr5, thresh5 = roc_curve(y_test, lor_y_pred, pos_label=1)
fpr6, tpr6, thresh6 = roc_curve(y_test, dt_y_pred, pos_label=1)
fpr7, tpr7, thresh7 = roc_curve(y_test, xgb_y_pred, pos_label=1)
fpr8, tpr8, thresh8 = roc_curve(y_test, sgd_y_pred, pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for n in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='SVC')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='NB')
plt.plot(fpr3, tpr3, linestyle='--',color='red', label='RF')
plt.plot(fpr4, tpr4, linestyle='--',color='yellow', label='KNN')
plt.plot(fpr5, tpr5, linestyle='--',color='violet', label='LoR')
plt.plot(fpr6, tpr6, linestyle='--',color='brown', label='DT')
plt.plot(fpr7, tpr7, linestyle='--',color='black', label='XGB')
plt.plot(fpr8, tpr8, linestyle='--',color='purple', label='SGD')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
# plt.legend()
# plt.legend(loc='best')
plt.savefig('D:/advanced verification and validation/nor_ROC.png',dpi=300)
plt.show()
