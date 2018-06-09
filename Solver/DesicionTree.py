import pandas as pd
import numpy as np
from sklearn import preprocessing ,cross_validation,neighbors ,svm
from sklearn.tree import DecisionTreeClassifier

# Reading the training csv to pandas dataframe
df = pd.read_csv("../datasets/train.csv");

# converting male/female to 1/0
df["Gender"] = [ int(item=="male") for item in df["Sex"].tolist()]

# Counting no of family members
sum_list = np.array(df[["Parch","SibSp"]])
c = [ sum(i) for i in sum_list ]
df["Family"] = c

# Extracting the TITLE from the Name
df['Title']=[ i.split(",")[1].split('.')[0] for i in (df['Name'])]
# Assigning a number to the title 
Titles = dict([ (obj,i) for i,obj in enumerate(set(df.Title))])
df.Title = [ Titles[i] for i in (df.Title)]

# Keeping the required Items only
df = df[["Pclass","Age","Family","Gender","Fare","Survived","Title"]]

# Dropping the Rows consisting of NaN
df.fillna(0,inplace=True)

# Separating Features and label
X = np.array(df.drop(['Survived'],1))
y = np.array(df['Survived'])

# Separating training and testing datas
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# Setting Up Classifier
# clf = neighbors.KNeighborsClassifier()
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)

# Checking for accuracy
accuracy = clf.score(X_test,y_test)
print(accuracy)

# Reading the file goes result is to be estimated
df2 = pd.read_csv("../datasets/test.csv");
ID = np.array(df2["PassengerId"])

# converting male/female to 1/0
df2["Gender"] = [ int(item=="male") for item in df2["Sex"].tolist()]

# Counting no of family members
sum_list = np.array(df2[["Parch","SibSp"]])
c = [ sum(i) for i in sum_list ]
df2["Family"] = c

# Extracting the TITLE from the Name
df2['Title']=[ i.split(",")[1].split('.')[0] for i in (df2['Name'])]
# Assigning a number to the title 
Titles = dict([ (obj,i) for i,obj in enumerate(set(df2.Title))])
df2.Title = [ Titles[i] for i in (df2.Title)]

# Keeping the required Items only
df2 = df2[["Pclass","Age","Family","Gender","Fare","Title"]]
df2.fillna(0,inplace=True)
print(np.array(df2))


prediction = clf.predict(np.array(df2))

result = {
	"PassengerId" :ID,
	"Survived" : prediction
}

res_df = pd.DataFrame(result);
res_df.set_index('PassengerId', inplace=True)
res_df.to_csv("./final_res.csv")

