
import pandas as pd 
import pickle
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("train.csv")
train.drop(columns = ["Name", "Ticket", "Cabin", "Embarked","PassengerId","Parch","Age","SibSp"], inplace = True)
train.Sex = train.Sex.replace({"female" : 1 , "male" : 0})
x_train = train[["Pclass","Sex","Fare"]]
y_train = train[["Survived"]]
log_reg = LogisticRegression(random_state = 69).fit(x_train,y_train)
with open("model","wb") as f:
    pickle.dump(log_reg,f)