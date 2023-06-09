import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split

usersDf = pd.read_csv(f'D:/advanced verification and validation/df_all.csv')

print(usersDf['forks'])
print(usersDf['run'])

usersDf['forks'] = [1 if i == "steady state" else 0 for i in usersDf['forks']]
usersDf['run'] = [1 if i == "steady state" else 0 for i in usersDf['run']]
# pd.Series(np.where(usersDf.forks.values == 'steady state', 1, 0),usersDf.index)

# usersDf['forks'] == [1 if i == "steady state" else 0 for i in usersDf['forks']]
#result_col = usersDf.drop(['forks'], axis= 1)
#result_col_val = usersDf['forks'].values
print(usersDf['forks'])
print(usersDf['run'])
x = usersDf.drop(['forks','filename','run','steady_state_starts'], axis =1)
# x = usersDf.drop(['filename'],axis=1)
y = usersDf['forks']

print(x)
print(y)

df_x=x.to_csv('D:/advanced verification and validation/df_x.csv',index=False)
df_y=y.to_csv('D:/advanced verification and validation/df_y.csv',index=False)

# reading the csv file using read_csv
# storing the data frame in variable called df
df_column_names_x = pd.read_csv('D:/advanced verification and validation/df_x.csv')

# creating a list of column names by
# calling the .columns
list_of_column_names_x = list(df_column_names_x.columns)

# displaying the list of column names
print('List of column names of x : ', list_of_column_names_x)

# reading the csv file using read_csv
# storing the data frame in variable called df
df_column_names_y = pd.read_csv('D:/advanced verification and validation/df_y.csv')

# creating a list of column names by
# calling the .columns
list_of_column_names_y = list(df_column_names_y.columns)

# displaying the list of column names
print('List of column names of y : ', list_of_column_names_y)
'''''
# open CSV file and assign header
with open("D:/advanced verification and validation/df_x_test.csv", 'w') as file:
    dw = csv.DictWriter(file, delimiter=',',
                        fieldnames=list_of_column_names)
    dw.writeheader()

# display csv file
fileContent = pd.read_csv("D:/advanced verification and validation/df_x_test.csv")
print(fileContent)
'''
'''''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

X_train_df=pd.DataFrame(X_train)
X_test_df=pd.DataFrame(X_test)

y_train_df=pd.DataFrame(y_train)
y_test_df=pd.DataFrame(y_test)

X_train_csv=X_train_df.to_csv("D:/advanced verification and validation/x_train_csv.csv",index=False)
X_test_csv=X_test_df.to_csv("D:/advanced verification and validation/x_test_csv.csv",index=False)

y_train_csv=y_train_df.to_csv("D:/advanced verification and validation/y_train_csv.csv",index=False)
y_test_csv=y_test_df.to_csv("D:/advanced verification and validation/y_test_csv.csv",index=False)
'''

with open("D:/advanced verification and validation/df_x.csv",'r') as firstfile, open("D:/advanced verification and validation/df_x_train.csv", 'w') as secondfile, open("D:/advanced verification and validation/df_x_test.csv", 'w') as thirdfile: # read content from first file
     count = 0
     for line in firstfile: # write content to second file

         count= count+1
         if count<=4231:
             secondfile.write(line)

         if count >4231:
             thirdfile.write(line)

         if count >=5291:
             break

with open("D:/advanced verification and validation/df_y.csv",'r') as fourthfile, open("D:/advanced verification and validation/df_y_train.csv", 'w') as fifthfile, open("D:/advanced verification and validation/df_y_test.csv", 'w') as sixthfile: # read content from first file
     count1 = 0
     for lines in fourthfile: # write content to second file

         count1= count1+1
         if count1<=4231:
             fifthfile.write(lines)

         if count1 >4231:
             sixthfile.write(lines)

         if count1 >=5291:
             break
'''''
# open CSV file and assign header
with open("D:/advanced verification and validation/df_x_test.csv") as file:

    dw = csv.DictWriter(file, delimiter=',',
                        fieldnames=list_of_column_names)
    dw.writeheader()
'''
# display csv file
fileContent_x = pd.read_csv("D:/advanced verification and validation/df_x_test.csv", header=None)
fileContent_x.columns=list_of_column_names_x
print(fileContent_x)
fileContent_x.to_csv("D:/advanced verification and validation/df_x_test.csv",index=False)

# display csv file
fileContent_y = pd.read_csv("D:/advanced verification and validation/df_y_test.csv", header=None)
fileContent_y.columns=list_of_column_names_y
print(fileContent_y)
fileContent_y.to_csv("D:/advanced verification and validation/df_y_test.csv",index=False)
'''''
def logic(index):
    for index in range(2,5231):
        return True
    return False

Df = pd.read_csv(f'D:/advanced verification and validation/df_x.csv', skiprows= lambda x: logic(x) )
print('Contents of the Dataframe created by skipping remaining row from csv file ')
print(Df)

Df.to_csv(f'D:/advanced verification and validation/x_train.csv',index=False)

def logic1(index):
    for index in range(2,5231):
        return False
    return True

Df = pd.read_csv(f'D:/advanced verification and validation/df_x.csv', skiprows= lambda x: logic1(x) )
print('Contents of the Dataframe created by skipping remaining row from csv file ')
print(Df)

Df.to_csv(f'D:/advanced verification and validation/x_test.csv',index=False)
'''

'''''
usersDf['forks'] == [1 if i == "steady state" else 0 for i in usersDf['forks']]
usersDf['run'] == [1 if i == "steady state" else 0 for i in usersDf['run']]

# result_col = usersDf.drop(['forks'], axis= 1)
# result_col_val = usersDf['forks'].values
print(usersDf['forks'])
print(usersDf['run'])

train, test = train_test_split(usersDf, test_size=0.2)

train_df=pd.DataFrame(train)
test_df=pd.DataFrame(test)


train_csv=train_df.to_csv("D:/advanced verification and validation/train_csv.csv")
test_csv=test_df.to_csv("D:/advanced verification and validation/test_csv.csv")
'''






