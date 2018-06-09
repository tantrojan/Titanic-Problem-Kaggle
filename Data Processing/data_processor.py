import pandas as pd
import numpy as np

df=pd.read_csv("../datasets/train.csv")
df2=pd.read_csv("../datasets/test.csv")

# ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      

PassengerId=df2["PassengerId"]
Survived=df["Survived"]

df.drop(["Survived"],1,inplace=True)

df=pd.concat([df,df2])
# There are 891 train samples

# Filling the NaN fields with 0
df.fillna(0,inplace=True)

print(df.columns)


#Gender
# df['Gender']=[ int(i=='male') for i in np.array(df.Sex)]
df['Gender']=list(map(lambda x: int(x=='male'),np.array(df['Sex'])))

#Title
df['Title']=[ (i.split(",")[1]).split(".")[0] for i in df['Name']]
# Titles={' Col', ' Mr', ' Mrs', ' Rev', ' Dr', ' Mme', ' Master', ' Lady', ' Sir', ' Jonkheer', ' Ms', ' Major', ' the Countess', ' Dona', ' Don', ' Capt', ' Miss', ' Mlle'}
df['Title']=df['Title'].replace([' Col',' Rev', ' Dr', ' Mme', ' Lady', ' Sir', ' Jonkheer',  ' Major', ' the Countess', ' Dona', ' Don', ' Capt', ' Mlle'],"Rare")
df['Title']=df['Title'].replace([' Ms'],' Miss')
Titles=dict([(obj,i) for i,obj in enumerate(set(df['Title']))])
df['Title']=[ Titles[i] for i in df['Title']]

#Family
sum_list = np.array(df[["Parch","SibSp"]])
c = [ sum(i) for i in sum_list ]
df["Family"] = c

#Embarkment
Embarks={'S':0 ,'C':1 ,'Q':2 ,'0':0}
df['Embarked']=[ Embarks[str(i)] for i in df['Embarked'] ]

#Age
Ages=df["Age"].tolist()
for i in range(len(Ages)):
	if(Ages[i]==0):
		Ages[i]=df["Age"].mean()
	Ages[i]=int(Ages[i])

df["Age"]=Ages
#Making Age Groups
df["age_groups"]=pd.cut(df["Age"],5)
# print(df.groupby('age_groups')[["Survived","age_groups"]].mean())
#                Survived
# age_groups             
# (-0.08, 16.0]  0.550000
# (16.0, 32.0]   0.344762
# (32.0, 48.0]   0.403226
# (48.0, 64.0]   0.434783
# (64.0, 80.0]   0.090909
df.loc[ df['Age'] <= 16, 'Age'] = 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
df.loc[ df['Age'] > 64, 'Age'] = 4

#Fare
#Making Fare groups
df["fare_groups"]=pd.qcut(df["Fare"],5)
# print(df.groupby('fare_groups')[["Survived","fare_groups"]].mean())
#                    Survived
# fare_groups                
# (-0.001, 7.854]    0.217877
# (7.854, 10.5]      0.201087
# (10.5, 21.679]     0.424419
# (21.679, 39.688]   0.444444
# (39.688, 512.329]  0.642045

df.loc[df['Fare']<= 8,"Fare"]=0
df.loc[(df['Fare']>8) & (df['Fare']<=11),"Fare"]=1
df.loc[(df['Fare']>11) & (df['Fare']<=22),"Fare"]=2
df.loc[(df['Fare']>22) & (df['Fare']<=40),"Fare"]=3
df.loc[(df['Fare']>40) & (df['Fare']<=513),"Fare"]=4




#Writing new CSVs
df=df[["Pclass","Gender","Family","Title","Embarked","Age","PassengerId","Fare"]]
train_data=df[:891]
train_data=train_data.join(Survived)

train_data.set_index('PassengerId', inplace=True)
train_data.to_csv("../datasets/train_processed_data.csv")

test_data=df[891:]
test_data.set_index('PassengerId',inplace=True)
test_data.to_csv("../datasets/test_processed_data.csv")