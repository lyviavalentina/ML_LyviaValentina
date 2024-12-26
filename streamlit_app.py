import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

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

  st.write('**Distribution of Exam Scores**')
  # Membuat figure dengan ukuran tertentu
  plt.figure(figsize=(15, 6))

  # Membuat diagram batang berdasarkan nilai Exam_Score
  ax = df.Exam_Score.value_counts().sort_index().plot.bar(color='skyblue', edgecolor='black')

  # Menambahkan anotasi (angka) di atas setiap batang
  for patch in ax.patches:
    # Mengambil ketinggian batang
    height = patch.get_height()
    # Menambahkan teks di atas batang
    ax.text(patch.get_x() + patch.get_width() / 2,  # Koordinat x (di tengah batang)
            height + 0.1,  # Koordinat y (sedikit di atas batang)
            int(height),  # Nilai yang ditampilkan
            ha="center", fontsize=10)  # Perataan horizontal dan ukuran font

  # Menambahkan judul dan label sumbu
  plt.title('Distribution of Exam Scores', fontsize=14)
  plt.xlabel('Exam Score', fontsize=12)
  plt.ylabel('Frequency', fontsize=12)
  plt.xticks(rotation=45)
  plt.tight_layout()  # Tata letak yang lebih rapi

  # Menampilkan plot
  st.pyplot(plt)

  exam_score_counts = df['Exam_Score'].value_counts().sort_index(ascending=False)
  st.dataframe(exam_score_counts)

  st.write('**Exam Score vs Attendance**')
  st.scatter_chart(data=df, x='Exam_Score', y='Attendance')
  
  st.write('**Exam Score vs Access to Resources**')
  # Create a new column 'Exam_Category'
  df['Exam Score'] = pd.cut(df['Exam_Score'], bins=[0, 70, 100], labels=['<70', '>=70'], right=False)

  # Membuat plot countplot dengan seaborn
  plt.figure(figsize=(10, 6))
  ax = sns.countplot(x='Exam Score', hue='Access_to_Resources', data=df, palette='pastel')

  # Menambahkan anotasi jumlah di atas setiap batang
  for container in ax.containers:
    for patch in container:
        # Mendapatkan tinggi batang
        height = patch.get_height()
        if height > 0:  # Hanya tambahkan anotasi jika tinggi batang > 0
            ax.text(patch.get_x() + patch.get_width() / 2,  # Posisi x
                    height + 0.1,  # Posisi y sedikit di atas batang
                    int(height),  # Nilai yang ditampilkan
                    ha='center', va='bottom', fontsize=9)  # Alignment dan ukuran font

  # Menambahkan judul dan label
  plt.title('Exam Score vs Access to Resources', fontsize=14)
  plt.xlabel('Exam Score Category', fontsize=12)
  plt.ylabel('Count', fontsize=12)
  plt.legend(title='Access to Resources', loc='upper right')
  plt.tight_layout()

  # Menampilkan plot
  st.pyplot(plt)

with st.sidebar:
  st.header('Input features')
