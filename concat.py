import json
import  csv
import glob
import os, json
import pandas as pd
import numpy as np
temp=[]
count=0
df4=pd.DataFrame()
df_all=pd.DataFrame()
df_clf=pd.DataFrame()
import os, glob
path1 = 'D:/PycharmProjects/advanced verification and validation/steady-state/data/timeseries/all/'
path2 = 'D:/PycharmProjects/advanced verification and validation/steady-state/data/classification/'

for filename in glob.glob(os.path.join(path1, '*.json')):
    with open(os.path.join(os.getcwd(), filename), 'r') as f1 :
        df1=pd.read_json(f1)
        df_all=df_all.append(df1,ignore_index=True)
        print(df_all)

for filename in glob.glob(os.path.join(path2, '*.json')):
    with open(os.path.join(os.getcwd(), filename), 'r') as f2:
        df2=pd.read_json(f2)
        df2['filename']=os.path.basename(filename)
        df_clf=df_clf.append(df2,ignore_index=True)
        print(df_clf)

df3=pd.concat([df_all,df_clf],axis=1,ignore_index=False)
print(df3)

df3.to_csv(f"D:/advanced verification and validation/df_all.csv",index=False)

