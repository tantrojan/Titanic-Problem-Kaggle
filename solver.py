import pandas as pd
import numpy as np
from sklearn import preprocessing ,cross_validation,neighbors

# Reading the csv to pandas dataframe
df = pd.read_csv("./datasets/train.csv");

# converting male/female to 1/0
df["Gender"] = [ int(item=="male") for item in df["Sex"].tolist()]

# Counting no of family members
sum_list = np.array(df[["Parch","SibSp"]])
c = [ sum(i) for i in sum_list ]
df["Family"] = c

# Keeping the required Items only
df = df[["Pclass","Age","Family","Gender","Fare","Survived"]]

# Dropping the Rows consisting of NaN
df.dropna(inplace=True)

# Features
X = np.array(df.drop(['Survived'],1))

# Label
y = np.array(df['class'])



clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)
