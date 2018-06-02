import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("./datasets/train.csv")
data_test=pd.read_csv("./datasets/test.csv")


# converting male/female to 1/0
df["Gender"] = [ int(item=="male") for item in df["Sex"].tolist()]

# Counting no of family members
sum_list = np.array(df[["Parch","SibSp"]])
c = [ sum(i) for i in sum_list ]
df["Family"] = c

# Keeping the required Items only
df = df[["Pclass","Age","Family","Gender","Fare","Survived"]]

df["Died"]=1-df["Survived"]
print(df['Fare'].max())
df.groupby('Family')[['Survived','Died']].agg('sum').plot(kind='bar')
plt.show()