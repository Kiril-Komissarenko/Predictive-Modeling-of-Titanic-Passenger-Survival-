import pandas as pd 
import pickle
with open("model", "rb") as f:
    model = pickle.load(f)
print("Enter name of file. Example(test.csv) accepting only csv\n")
name = input()
test = pd.read_csv(name)
test.drop(columns = ["Name", "Ticket", "Cabin", "Embarked","Parch", "Age" ,"SibSp"], inplace = True)
test.Sex = test.Sex.replace({"female" : 1, "male" : 0})
test.Fare.fillna(value = test["Fare"].mean(), inplace = True)
test.Sex.fillna(value = test["Sex"].mean(), inplace = True)
test.Pclass.fillna(value = test["Pclass"].mean(), inplace = True)
result = model.predict(test[["Pclass","Sex","Fare"]])
result = pd.DataFrame(result)
result = pd.concat([test["PassengerId"],result],axis = 1)
result.columns = ["PassengerId" , "Survived"]
result.to_csv("predictions.csv" , sep = "," ,index = False )
