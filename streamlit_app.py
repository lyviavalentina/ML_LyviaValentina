import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('👩🏻‍🎓 Student Performance Factors Machine Learning')

st.write('This is app builds a machine learning model!')

with st.expander('**Data**'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/lyviavalentina/data/refs/heads/main/StudentPerformanceFactors%20(1).csv')

  # Replace categorical values in 'Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home' with numerical values
  df['Teacher_Quality'] = df['Teacher_Quality'].replace({'Low': 0, 'Medium': 1, 'High': 2})
  df['Parental_Education_Level'] = df['Parental_Education_Level'].replace({'High School': 0, 'College': 1, 'Postgraduate': 2})
  df['Distance_from_Home'] = df['Distance_from_Home'].replace({'Near': 0, 'Moderate': 1, 'Far': 2})

  mean1 = df['Teacher_Quality'].mean()
  df['Teacher_Quality'] = df['Teacher_Quality'].fillna(mean1)

  mean2 = df['Parental_Education_Level'].mean()
  df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(mean2)

  mean3 = df['Distance_from_Home'].mean()
  df['Distance_from_Home'] = df['Distance_from_Home'].fillna(mean3)

  df.isnull().any()

  df['Teacher_Quality'] = df['Teacher_Quality'].replace({0: 'Low', 1: 'Medium', 2: 'High'})
  df['Parental_Education_Level'] = df['Parental_Education_Level'].replace({0: 'High School', 1: 'College', 2: 'Postgraduate'})
  df['Distance_from_Home'] = df['Distance_from_Home'].replace({0: 'Near', 1: 'Moderate', 2: 'Far'})
  df

with st.expander('**Data Visualization**'):
  # Identify categorical columns (object type)
  categorical_columns = df.select_dtypes(include=['object']).columns

  # Apply Label Encoding to each categorical column
  from sklearn.preprocessing import LabelEncoder

  data_encoded = df.copy()
  label_encoders = {}
  for column in categorical_columns:
    le = LabelEncoder()
    data_encoded[column] = le.fit_transform(data_encoded[column].astype(str))
    label_encoders[column] = le

  # Display the first few rows of the encoded dataset and column info
  data_encoded.head(), data_encoded.info()
