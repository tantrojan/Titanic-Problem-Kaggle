import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train=pd.read_csv("./datasets/train.csv")
data_test=pd.read_csv("./datasets/test.csv")

# print(data_train.describe())
data_train["Died"]=1-data_train["Survived"]
data_train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7), colors=['b', 'r']);

plt.title("Distribution based on GENDER")
plt.show()

# label=data_train["Survived"]
data_train.drop(["Survived"],1,inplace=True)
