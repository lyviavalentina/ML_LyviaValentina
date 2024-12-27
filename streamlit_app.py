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

  mean1 = df['Teacher_Quality'].mean()
  mean1_rounded = round(mean1)
  df['Teacher_Quality'] = df['Teacher_Quality'].fillna(mean1_rounded)

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
  c = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x="Exam_Score",
        y=alt.Y("Attendance", scale=alt.Scale(domain=[55, 100])),
        size="Hours_Studied",
        color="Attendance",
        tooltip=["Exam_Score", "Attendance"]
    )
  )

  # Display the chart in Streamlit
  st.altair_chart(c, use_container_width=True)

  st.write('**Exam Score vs Hours Studied**')
  d = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x="Exam_Score",
        y="Hours_Studied",
        size="Hours_Studied",
        color="Exam_Score",
        tooltip=["Exam_Score", "Hours_Studied"]
    )
  )

  # Display the chart in Streamlit
  st.altair_chart(d, use_container_width=True)
  
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

  st.write('**Exam Score vs Parental Involvement**')
  # Create a new column 'Exam_Category'
  df['Exam Score'] = pd.cut(df['Exam_Score'], bins=[0, 70, 100], labels=['<70', '>=70'], right=False)
  df['Previous Scores'] = pd.cut(df['Previous_Scores'], bins=[0, 70, 100], labels=['<70', '>=70'], right=False)

  # Membuat plot countplot dengan seaborn
  plt.figure(figsize=(10, 6))
  ax = sns.countplot(x='Exam Score', hue='Previous Scores', data=df, palette='pastel')

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
  plt.title('Exam Score vs Previous Scores', fontsize=14)
  plt.xlabel('Exam Score Category', fontsize=12)
  plt.ylabel('Count', fontsize=12)
  plt.legend(title='Previous Scores', loc='upper right')
  plt.tight_layout()

  # Menampilkan plot
  st.pyplot(plt)

  st.write('**Exam Score vs Tutoring Sessions**')
  df['Exam Score'] = pd.cut(df['Exam_Score'], bins=[0, 70, 100], labels=['<70', '>=70'], right=False)

  # Membuat plot countplot dengan seaborn
  plt.figure(figsize=(10, 6))
  ax = sns.countplot(x='Exam Score', hue='Tutoring_Sessions', data=df, palette='pastel')

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
  plt.title('Exam Score vs Tutoring Sessions', fontsize=14)
  plt.xlabel('Exam Score Category', fontsize=12)
  plt.ylabel('Count', fontsize=12)
  plt.legend(title='Tutoring Sessions', loc='upper right')
  plt.tight_layout()

  # Menampilkan plot
  st.pyplot(plt)

with st.sidebar:
  st.header('Input features')
  Hours_Studied = st.slider('Hours Studied', 1, 44, 20)
  Attendance = st.slider('Attendance', 60, 100, 80)
  Parental_Involvement = st.selectbox('Parental Involvement', ('Low', 'Medium', 'High'))
  Access_to_Resources = st.selectbox('Access to Resources', ('Low', 'Medium', 'High'))
  Extracurricular_Activities = st.selectbox('Extracurricular Activities', ('Yes', 'No'))
  Sleep_Hours = st.slider('Sleep Hours', 4, 10, 7)
  Previous_Score = st.slider('Previous Score', 50, 100, 75)
  Motivation_Level = st.selectbox('Motivation Level', ('Low', 'Medium', 'High'))
  Internet_Access = st.selectbox('Internet Access', ('Yes', 'No'))
  Tutoring_Sessions = st.slider('Tutoring Sessions', 0, 8, 4)
  Family_Income = st.selectbox('Family Income', ('Low', 'Medium', 'High'))
  Teacher_Quality = st.selectbox('Teacher Quality', ('Low', 'Medium', 'High'))
  School_Type = st.selectbox('School Type', ('Public', 'Private'))
  Peer_Influence = st.selectbox('Peer Influence', ('Positive', 'Neutral', 'Negative'))
  Physical_Activity = st.slider('Physical Activity', 0, 6, 3)
  Learning_Disabilities = st.selectbox('Learning Disabilities', ('Yes', 'No'))
  Parental_Education_Level = st.selectbox('Parental Education Level', ('High School', 'College', 'Postgraduate'))
  Distance_from_Home = st.selectbox('Distance from Home', ('Near', 'Moderate', 'Far'))
  Gender = st.selectbox('Gender', ('Male', 'Female'))

  data = {'Hours_Studied': Hours_Studied,
          'Attendance': Attendance,'Parental_Involvement': Parental_Involvement,
          'Access_to_Resources': Access_to_Resources,
          'Extracurricular_Activities': Extracurricular_Activities,
          'Sleep_Hours': Sleep_Hours,
          'Previous_Score': Previous_Score,
          'Motivation_Level': Motivation_Level,
          'Internet_Access': Internet_Access,
          'Tutoring_Sessions': Tutoring_Sessions,
          'Family_Income': Family_Income,
          'Teacher_Quality': Teacher_Quality,
          'School_Type': School_Type,
          'Peer_Influence': Peer_Influence,
          'Physical_Activity': Physical_Activity,
          'Learning_Disabilities': Learning_Disabilities,
          'Parental_Education_Level': Parental_Education_Level,
          'Distance_from_Home': Distance_from_Home,
          'Gender': Gender}
  filtered_df = df.copy()

  # Filter by teacher quality
  if teacher_quality != 'All':
    filtered_df = filtered_df[filtered_df['Teacher_Quality'] == teacher_quality]

  # Filter by parental education level
  if parental_education != 'All':
    filtered_df = filtered_df[filtered_df['Parental_Education_Level'] == parental_education]

  # Filter by distance from home
  if distance_from_home != 'All':
    filtered_df = filtered_df[filtered_df['Distance_from_Home'] == distance_from_home]

  # Filter by attendance range
  filtered_df = filtered_df[(filtered_df['Attendance'] >= attendance[0]) & (filtered_df['Attendance'] <= attendance[1])]

  # Display the filtered DataFrame
st.write('### Filtered Data', filtered_df)


