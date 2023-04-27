#Importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

"""Data collection and analysis
PIMA Diabetes Dataset
"""
def diabetesPrediction(input_data):
  # loading diabetes dataset to pandas data frame
  d_d = pd.read_csv('diabetes.csv')
  # diabetes_dataset = d_d

  # help
  # pd.read_csv?

  # printing first 5 row of dataset
  print("head")
  print(d_d.head())

  # number of rows and colums in this dataset
  print("shape")
  print(d_d.shape)

  # getting statictical meausure of data
  print("describe")
  print(d_d.describe())

  print("value_counts")
  print(d_d['Outcome'].value_counts())
  # 0 --> non_diabetic 
  # 1 --> diabetic

  print("Mean")
  print(d_d.groupby('Outcome').mean())

  # seprating data and labels
  x = d_d.drop(columns='Outcome',axis=1)
  y = d_d['Outcome']

  print("X , Y")
  print(x)
  print(y)

  """Data standirized"""

  scaler = StandardScaler()

  scaler.fit(x)

  standirized_data = scaler.transform(x)

  print("standirized_data")
  print(standirized_data)

  x = standirized_data
  y = d_d['Outcome']
  print("x , y")
  print(x,y,end="\n")

  """Train Test Split"""
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state = 40, stratify=y)
  print(x.shape,x_train.shape,x_test.shape)

  lr_model = LogisticRegression()
  lr_model.fit(x_train,y_train)
  accuracy_lr = lr_model.score(x_test,y_test)
  print("Final Accuracy Score ")
  print("Logistic Regression accuracy is :",accuracy_lr)

  """Model Evaluation"""

  # lr_pred= lr_model.predict(x_test)
  # report = classification_report(y_test,lr_pred)
  # print(report)
  # if the accuracy of train data is very high and test data is very low is know as  overhidding

  """Making a prediction data"""

  # from numpy.core.fromnumeric import std
  # input_data = (17,163,72,41,114,40.9,0.817,47)
  # 2,88,74,19,53,29,0.229,22,0
  # 17,163,72,41,114,40.9,0.817,47,1 

  # changing input data into numpy array
  np_array = np.asarray(input_data)

  # reshape array as we are predicting for one instance
  input_shapped = np_array.reshape(1,-1)

  # standardized data
  std_data = scaler.transform(input_shapped)
  #LOGISTIC REGRESSION CONFUSION MATRIX
  # plt.figure(figsize=(4,3))
  # sns.heatmap(confusion_matrix(y_test, lr_pred),
  #                 annot=True,fmt = "d",linecolor="k",linewidths=3)
      
  # plt.title("LOGISTIC REGRESSION CONFUSION MATRIX",fontsize=14)
  # plt.show()
  predicition = lr_model.predict(std_data)
  print(std_data,predicition)

  if(predicition == 0):
    # print("Non Diabetics 😎")
    return False
  else:
    # print("Diabetics 🥲")
    return True