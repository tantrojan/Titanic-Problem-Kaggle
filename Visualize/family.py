import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("../datasets/train.csv")
data_test=pd.read_csv("../datasets/test.csv")


# Counting no of family members
sum_list = np.array(df[["Parch","SibSp"]])
c = [ sum(i)+1 for i in sum_list ]
df["Family"] = c


df["Died"]=1-df["Survived"]
df.groupby('Family')[['Survived','Died']].agg('sum').plot(kind='bar')
plt.title("Distribution based on number of Family members")
plt.show()