import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
df=pd.read_csv("../datasets/train.csv")
data_test=pd.read_csv("../datasets/test.csv")


# Counting no of family members
df["Gender"]=[ int(x=='male') for x in df["Sex"]]


df["Died"]=1-df["Survived"]
df.groupby('Sex')[['Survived','Died']].agg('sum').plot(kind='bar',figsize=(15,8))
plt.title("Distribution based on Gender")
plt.xlabel("Groups")
plt.ylabel("Count")
plt.show()