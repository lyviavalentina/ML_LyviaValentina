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

  mean2 = df['Parental_Education_Level'].mean()
  df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(mean2)

  mean3 = df['Distance_from_Home'].mean()
  df['Distance_from_Home'] = df['Distance_from_Home'].fillna(mean3)

  df.isnull().any()

  df['Teacher_Quality'] = df['Teacher_Quality'].replace({0: 'Low', 1: 'Medium', 2: 'High'})
  df['Parental_Education_Level'] = df['Parental_Education_Level'].replace({0: 'High School', 1: 'College', 2: 'Postgraduate'})
  df['Distance_from_Home'] = df['Distance_from_Home'].replace({0: 'Near', 1: 'Moderate', 2: 'Far'})
  df

with st.expander('**Descriptive Statistics**'):
  df.head(), df.info(), df.describe(include='all')

with st.expander('**Data Visualization**'):
  st.write('**Heatmap**')
    numerical_columns = df.select_dtypes(include=['int64']).columns
    correlation_matrix = df[numerical_columns].corr()

    # Menampilkan korelasi yang signifikan
    threshold = 0.5  # Atur ambang batas korelasi signifikan
    significant_correlations = correlation_matrix[(correlation_matrix >= threshold) & (correlation_matrix != 1.0)]

    # Visualisasi heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix')  # Tambahkan judul pada heatmap
    st.pyplot(plt)

  # Pilih variabel penting (contoh: "Exam_Score")
  important_variable = 'Exam_Score'
  st.write('**Distribution Graph**')
  # Grafik Distribusi
  plt.figure(figsize=(10, 6))
  sns.histplot(df[important_variable], kde=True, bins=30, color='blue')
  plt.title(f'Distribution of {important_variable}')
  plt.xlabel(important_variable)
  plt.ylabel('Frequency')
  st.pyplot(plt)
  
  st.write('**Boxplot**')
  # Boxplot
  plt.figure(figsize=(8, 6))
  sns.boxplot(x=df[important_variable], color='orange') # Changed line
  plt.title(f'Boxplot of {important_variable}')
  plt.xlabel(important_variable)
  st.pyplot(plt)
