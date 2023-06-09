import json
import  csv
import glob
import os, json
import pandas as pd

count=0
import os, glob
path = 'D:/PycharmProjects/advanced verification and validation/steady-state/data/timeseries/all/'
for filename in glob.glob(os.path.join(path, '*.json')):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        df=pd.read_json(f)
        print(df)
        df.to_csv(f"D:/advanced verification and validation/df_{count}.csv",index=False)
        count=count+1

count=0
path = 'D:/PycharmProjects/advanced verification and validation/steady-state/data/classification/'
for filename in glob.glob(os.path.join(path, '*.json')):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        df1=pd.read_json(f)
        print(df)
        df1.to_csv(f"D:/advanced verification and validation/df1_{count}.csv",index=True)
        count=count+1

