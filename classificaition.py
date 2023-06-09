import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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

x_train = pd.read_csv("D:/advanced verification and validation/df_x_train.csv")
y_train = pd.read_csv("D:/advanced verification and validation/df_y_train.csv")
# x_train=x_train.drop(['filename'],axis=1)
y_train=y_train.values.reshape(-1,1)

x_test = pd.read_csv("D:/advanced verification and validation/df_x_test.csv")
y_test = pd.read_csv("D:/advanced verification and validation/df_y_test.csv")
# x_test=x_test.drop(x_test.iloc[:,3002],axis=1)
y_test=y_test.values.reshape(-1,1)

###### SVC ############
print("###### SVC ############")
svc_model = SVC(kernel='linear', random_state=0)

svc_model.fit(x_train, y_train)

svc_y_pred=svc_model.predict(x_test)

print("svc_y_pred = ", svc_y_pred)

# labels = ["No Steady state", "steady state"]
svc_cm = confusion_matrix(y_test, svc_y_pred)
print("svc confusion matrix = ", svc_cm)
svc_disp=ConfusionMatrixDisplay(confusion_matrix=svc_cm, display_labels=None)
svc_disp.plot()
# plt.show()
plt.savefig("D:/advanced verification and validation/svc_cm.png")
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
nb_model.fit(x_train, y_train)

# Predict Output
nb_y_pred = nb_model.predict(x_test)
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
plt.savefig("D:/advanced verification and validation/NB_cm.png")
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
rf.fit(x_train, y_train)
rf_y_pred = rf.predict(x_test)
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
plt.savefig("D:/advanced verification and validation/RF_cm.png")
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
knn.fit(x_train, y_train)
knn_y_pred = knn.predict(x_test)

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
plt.savefig("D:/advanced verification and validation/KNN_cm.png")
plt.show()
plt.close()

knn_res=[]
knn_res.append("KNN")
knn_res.append(knn_ac)
knn_res.append(knn_f1_score)
knn_res.append(auc_score_knn)
print("knn_res = ", knn_res)

result=[]
result.append(svc_res)
result.append(NB_res)
result.append(RF_res)
result.append(knn_res)
print("#####result#####", result)
result_df=pd.DataFrame(result,columns=['Clf Name','Accuracy','F1 Score','ROC score'])
print(result_df)

result_df.to_csv("D:/advanced verification and validation/result_df.csv",index=False)

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, svc_y_pred, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, nb_y_pred, pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, rf_y_pred, pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(y_test, knn_y_pred, pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for n in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='SVC')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='NB')
plt.plot(fpr3, tpr3, linestyle='--',color='red', label='RF')
plt.plot(fpr4, tpr4, linestyle='--',color='yellow', label='KNN')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
# plt.legend()
# plt.legend(loc='best')
plt.savefig('D:/advanced verification and validation/ROC.png',dpi=300)
plt.show()
'''''
x_train = pd.read_csv("D:/advanced verification and validation/x_train_csv.csv")
y_train = pd.read_csv("D:/advanced verification and validation/y_train_csv.csv")
# x_train=x_train.drop(['filename'],axis=1)
y_train=y_train.values.reshape(-1,1)

x_test = pd.read_csv("D:/advanced verification and validation/x_test_csv.csv")
y_test = pd.read_csv("D:/advanced verification and validation/y_test_csv.csv")
# x_test=x_test.drop(x_test.iloc[:,3002],axis=1)
y_test=y_test.values.reshape(-1,1)

svc_model = SVC(kernel='linear', random_state=0)

svc_model.fit(x_train, y_train)

y_pred=svc_model.predict(x_test)

print("y_pred = ", y_pred)

print(confusion_matrix(y_test,y_pred))


score=accuracy_score(y_test,y_pred)
print("accuracy_score : ",score)
'''


'''''
# Plotting the training set
fig, ax = plt.subplots(figsize=(12, 7))
# removing to and right border
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
# adding major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
ax.scatter(x_train, x_test, color="#8C7298")
plt.show()
'''
'''
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# import test_train
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.impute import SimpleImputer




clf = LinearSVC()
x_train = pd.read_csv("D:/advanced verification and validation/df_x.csv")
y_train = pd.read_csv("D:/advanced verification and validation/df_y.csv")
x_train=x_train.drop(['filename'],axis=1)

svc_model = SVC(kernel='linear', random_state=32)
svc_model.fit(x_train, y_train)

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(x_train.iloc[:, 1:2999])
x_train.iloc[:, 1:2999] = imputer.transform(x_train.iloc[:, 1:2999])

plt.figure(figsize=(10, 8))
# Plotting our two-features-space
sns.scatterplot(x=x_train[:,-1],
                y=x_train[:,-1],
                hue=y_train,
                s=8);
# Constructing a hyperplane using a formula.
w = svc_model.coef_[0]           # w consists of 2 elements
b = svc_model.intercept_[0]      # b consists of 1 element
x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
# Plotting a red hyperplane
plt.plot(x_points, y_points, c='r');
'''''
'''''
# fitting x samples and y classes
clf.fit(x_train, y_train)
print(clf)

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x_train, y_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)
lin_svc = svm.LinearSVC(C=C).fit(x_train, y_train)

# create a mesh to plot in
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

# plt.scatter(x_train, x_train, c=y_train, cmap='winter');
# plt.show()

'''
'''''
xlim = ax.get_xlim()
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - clf.intercept_[0] / w[1]
plt.plot(xx, yy)
yy = a * xx - (clf.intercept_[0] - 1) / w[1]
plt.plot(xx, yy, 'k--')
yy = a * xx - (clf.intercept_[0] + 1) / w[1]
plt.plot(xx, yy, 'k--')
'''
'''''
x_test = pd.read_csv("D:/advanced verification and validation/df_x_test.csv")
y_test = pd.read_csv("D:/advanced verification and validation/df_y_test.csv")
x_test=x_test.drop(['filename'],axis=1)

expected_y  = y_test
predicted_y = clf.predict(x_test)
'''
'''''
print(metrics.classification_report(expected_y, predicted_y))
print(metrics.confusion_matrix(expected_y, predicted_y))
'''

