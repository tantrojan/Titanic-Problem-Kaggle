import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("../datasets/train.csv")
data_test=pd.read_csv("../datasets/test.csv")


# converting male/female to 1/0
df["Gender"] = [ int(item=="male") for item in df["Sex"].tolist()]

# Counting no of family members
sum_list = np.array(df[["Parch","SibSp"]])
c = [ sum(i) for i in sum_list ]
df["Family"] = c

df["Died"]=1-df["Survived"]

print((df.Name))
# Extracting the TITLE from the Name
df['Title']=[ i.split(",")[1].split('.')[0] for i in (df['Name'])]

print((df.Title))

df.groupby('Title')[['Survived','Died']].agg('sum').plot(kind='bar')
plt.title("Distribution with respect to TITLE")
plt.show()

# Assigning a number to the title 
Titles = dict([ (obj,i) for i,obj in enumerate(set(df.Title))])
print(Titles)

df.Title = [ Titles[i] for i in (df.Title)]
print((df.Title))
