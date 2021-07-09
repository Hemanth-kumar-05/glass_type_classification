# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

feature_col = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

@st.cache()
def prediction(model, feat):
    pred = model.predict([feat])
    pred = pred[0]
    if pred == 1:
        return "building windows float processed".upper()
    elif pred == 2:
        return "building windows non float processed".upper()
    elif pred == 3:
        return "vehicle windows float processed".upper()
    elif pred == 4:
        return "vehicle windows non float processed".upper()
    elif pred == 5:
        return "containers".upper()
    elif pred == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()

st.title("Glass Type Prediction Web app")
st.sidebar.title("Glass Type Prediction Web app")


if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Glass Type Dataset")
    st.dataframe(glass_df)

st.sidebar.subheader("Visualization selector")
plot_list = st.sidebar.multiselect("Select the charts/plots:", ("Correlation Heatmap", "Line Chart", "Area Chart", "Count Plot", "Pie Chart", "Boxplot"))

import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

if 'Correlation Heatmap' in plot_list:
  # plot correlation heatmap
  st.subheader("Correlation Heatmap")
  plt.figure(figsize=(6, 6))
  sns.heatmap(glass_df.corr(), annot=True)
  st.pyplot()

if 'Line Chart' in plot_list:
  # plot line chart 
  st.subheader("Line Chart")
  st.line_chart(glass_df)

if 'Area Chart' in plot_list:
  # plot area chart 
  st.subheader("Area Chart")
  st.area_chart(glass_df)

if 'Count Plot' in plot_list:
  # plot count plot 
  st.subheader("Count Plot")
  plt.figure(figsize=(14, 4))
  sns.countplot(x="GlassType", data=glass_df)
  st.pyplot()

if 'Pie Chart' in plot_list:
  # plot pie chart
  st.subheader("Pie Chart")
  pie_data = glass_df["GlassType"].value_counts()

  plt.figure(dpi=108)
  plt.pie(pie_data, labels=pie_data.index, autopct="%1.2f%%", startangle=30)
  st.pyplot()

if 'Boxplot' in plot_list:
  # plot box plot
  st.subheader("Box Plot")
  column = st.sidebar.selectbox("Select the columns for boxplot", ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

  plt.figure(figsize=(14, 4))
  sns.boxplot(x=column, data=glass_df)
  st.pyplot()