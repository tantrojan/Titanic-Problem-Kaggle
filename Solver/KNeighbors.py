import pandas as pd
import numpy as np
from sklearn import preprocessing ,cross_validation,neighbors ,svm

# Reading the training csv to pandas dataframe
X=pd.read_csv("../datasets/train_processed_data.csv")

Y=X["Survived"]
X.drop("Survived",1,inplace=True)
X.drop("PassengerId",1,inplace=True)


# X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.4)

clf=neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(X,Y)

# print(clf.score(X_test,Y_test))


Features=pd.read_csv("../datasets/test_processed_data.csv")
ID=Features["PassengerId"].tolist()
Features.drop("PassengerId",1,inplace=True)

Prediction = clf.predict(np.array(Features))

Result={ "PassengerId" : ID, "Survived" : Prediction }

Resultdf=pd.DataFrame(Result)
Resultdf.set_index('PassengerId', inplace=True)
Resultdf.to_csv("./final_res.csv")