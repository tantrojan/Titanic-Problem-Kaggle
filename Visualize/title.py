import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("../datasets/train.csv")
data_test=pd.read_csv("../datasets/test.csv")


df["Died"]=1-df["Survived"]

# Extracting the TITLE from the Name
df['Title']=[ i.split(",")[1].split('.')[0] for i in (df['Name'])]
# Titles={' Col', ' Mr', ' Mrs', ' Rev', ' Dr', ' Mme', ' Master', ' Lady', ' Sir', ' Jonkheer', ' Ms', ' Major', ' the Countess', ' Dona', ' Don', ' Capt', ' Miss', ' Mlle'}
df['Title']=df['Title'].replace([' Col',' Rev', ' Dr', ' Mme', ' Lady', ' Sir', ' Jonkheer',  ' Major', ' the Countess', ' Dona', ' Don', ' Capt', ' Mlle'],"Rare")
df['Title']=df['Title'].replace([' Ms'],' Miss')

df.groupby('Title')[['Survived','Died']].agg('sum').plot(kind='bar')
plt.title("Distribution with respect to TITLE")
plt.show()

df.Title = [ Titles[i] for i in (df.Title)]
print((df.Title))
