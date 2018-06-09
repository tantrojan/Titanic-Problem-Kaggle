import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

data_train=pd.read_csv("../datasets/train.csv")
data_test=pd.read_csv("../datasets/test.csv")


data_train["Died"]=1-data_train["Survived"]
data_train.groupby('Pclass').agg('sum')[['Survived','Died']].plot(kind='bar',figsize=(15,8))

plt.title("Distribution based on CLASS")
plt.xlabel("Groups")
plt.ylabel("Count")
plt.show()
