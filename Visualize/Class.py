import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train=pd.read_csv("./datasets/train.csv")
data_test=pd.read_csv("./datasets/test.csv")


data_train["Died"]=1-data_train["Survived"]
print(data_train.groupby('Pclass').agg('sum')[['Survived','Died']].plot(kind='bar'))

# plt.title("Distribution based on CLASS")
# plt.show()
