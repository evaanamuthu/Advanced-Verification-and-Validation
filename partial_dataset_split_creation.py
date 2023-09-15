import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

ds=pd.read_csv("D:/advanced verification and validation/df_all.csv")
#########partial dataset creation############1500:3000###############
filename=ds['filename']
# if ds['filename']==filename:
df=ds.iloc[:,1500:]
print(ds.iloc[:,1500:])

partial_df=pd.DataFrame(df)
partial_df.to_csv("D:/advanced verification and validation/partial_df.csv",index=False)
partial_ds=pd.read_csv(f"D:/advanced verification and validation/partial_df.csv")

ds['forks'] = [1 if i == "steady state" else 0 for i in ds['forks']]
y=ds['forks']

y.to_csv("D:/advanced verification and validation/partial_df_y.csv",index=False)

# ds=ds.drop(['forks','filename','run','steady_state_starts'],axis=1)


# ds=ds.drop(['forks','filename','run','steady_state_starts'],axis=1)
partial_df_x=ds.iloc[:,1500:3000]
print(ds.iloc[:,1500:3000])

# partial_df_x=partial_ds.drop(['forks'],axis=1)
partial_df_x.to_csv("D:/advanced verification and validation/partial_df_x.csv",index=False)
##########manual split#################
with open("D:/advanced verification and validation/partial_df_x.csv",'r') as firstfile, open("D:/advanced verification and validation/partial_df_x_train.csv", 'w') as secondfile, open("D:/advanced verification and validation/partial_df_x_test.csv", 'w') as thirdfile: # read content from first file
     count = 0
     for line in firstfile: # write content to second file

         count= count+1
         if count<=4231:
             secondfile.write(line)

         if count >4231:
             thirdfile.write(line)

         if count >=5291:
             break

with open("D:/advanced verification and validation/partial_df_y.csv",'r') as fourthfile, open("D:/advanced verification and validation/partial_df_y_train.csv", 'w') as fifthfile, open("D:/advanced verification and validation/partial_df_y_test.csv", 'w') as sixthfile: # read content from first file
     count1 = 0
     for lines in fourthfile: # write content to second file

         count1= count1+1
         if count1<=4231:
             fifthfile.write(lines)

         if count1 >4231:
             sixthfile.write(lines)

         if count1 >=5291:
             break


x_train = pd.read_csv(f"D:/advanced verification and validation/partial_df_x_train.csv")
y_train = pd.read_csv(f"D:/advanced verification and validation/partial_df_y_train.csv")


x_test = pd.read_csv(f"D:/advanced verification and validation/partial_df_x_test.csv")
y_test = pd.read_csv(f"D:/advanced verification and validation/partial_df_y_test.csv")


print(x_train)
nor_x_train=x_train/np.linalg.norm(x_train, axis=1, keepdims=True)
print(nor_x_train)
nor_x_train.to_csv("D:/advanced verification and validation/partial_df_norm_x_train.csv",index=False)

print(x_test)
nor_x_test=x_test/np.linalg.norm(x_test, axis=1, keepdims=True)
print(nor_x_test)
nor_x_test.to_csv("D:/advanced verification and validation/partial_df_norm_x_test.csv",index=False)
#################80:20 split without considering fork name##############
ds=pd.read_csv("D:/advanced verification and validation/partial_df.csv")
filename=ds['filename']
# if ds['filename']==filename:
ds['forks'] = [1 if i == "steady state" else 0 for i in ds['forks']]
y=ds['forks']
x=ds.drop(['forks','filename','run','steady_state_starts'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42)
print("*********X_train*********",X_train)
print("*********X_test*********",X_test)
print("*********y_train**********",y_train)
print("**********y_test********",y_test)

X_train.to_csv('D:/advanced verification and validation/dataset/partial_df_x_train.csv',index=False)
X_test.to_csv('D:/advanced verification and validation/dataset/partial_df_x_test.csv',index=False)
Y_train.to_csv('D:/advanced verification and validation/dataset/partial_df_y_train.csv',index=False)
Y_test.to_csv('D:/advanced verification and validation/dataset/partial_df_y_test.csv',index=False)

nor_x_train=X_train/np.linalg.norm(X_train, axis=1, keepdims=True)
print(nor_x_train)
nor_x_train.to_csv(f"D:/advanced verification and validation/dataset/partial_df_norm_x_train.csv",index=False)

nor_x_test=X_test/np.linalg.norm(X_test, axis=1, keepdims=True)
print(nor_x_test)
nor_x_test.to_csv(f"D:/advanced verification and validation/dataset/partial_df_norm_x_test.csv",index=False)
#################80:20 split considering fork name##############
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
    ds=ds.drop(['forks','run','steady_state_starts'],axis=1)
    x=ds.iloc[:,1500:3000]
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

X_train_all.to_csv(f"D:/master thesis/dataset/train/partial_x_train.csv",index=False)
X_test_all.to_csv(f"D:/master thesis/dataset/test/partial_x_test.csv",index=False)
y_train_all.to_csv(f"D:/master thesis/dataset/train/partial_y_train.csv",index=False)
y_test_all.to_csv(f"D:/master thesis/dataset/test/partial_y_test.csv",index=False)


X_train=pd.read_csv(f"D:/master thesis/dataset/train/partial_x_train.csv")
X_test=pd.read_csv(f"D:/master thesis/dataset/test/partial_x_test.csv")
y_train=pd.read_csv(f"D:/master thesis/dataset/train/partial_y_train.csv")
y_test=pd.read_csv(f"D:/master thesis/dataset/test/partial_y_test.csv")

nor_x_train=X_train/np.linalg.norm(X_train, axis=1, keepdims=True)
print(nor_x_train)
nor_x_train.to_csv(f"D:/master thesis/dataset/train/partial_df_norm_x_train.csv",index=False)

nor_x_test=X_test/np.linalg.norm(X_test, axis=1, keepdims=True)
print(nor_x_test)
nor_x_test.to_csv(f"D:/master thesis/dataset/test/partial_df_norm_x_test.csv",index=False)
