import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("../datasets/train.csv")
data_test=pd.read_csv("../datasets/test.csv")


df["Died"]=1-df["Survived"]

# Extracting the TITLE from the Name
df['age_groups']=pd.cut(df["Age"],5)

df.groupby('age_groups')[['Survived','Died']].agg('sum').plot(kind='bar')
plt.title("Distribution with respect to Age")
plt.show()
