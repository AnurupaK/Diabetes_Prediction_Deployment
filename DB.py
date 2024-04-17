import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings("ignore")

diabetes = pd.read_csv("Diabetes_data.csv")
df = diabetes.copy()
# print(df.head(5))



X = df.drop(columns='Outcome', axis=1)
Y = diabetes['Outcome']



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1,stratify=Y, random_state=2)

LogR_model = LogisticRegression()

LogR_model.fit(X_train,Y_train)


pickle.dump(LogR_model,open('LogR_model.pkl','wb'))
LogR_model=pickle.load(open('LogR_model.pkl','rb'))


