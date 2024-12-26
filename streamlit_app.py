import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('👩🏻‍🎓 Student Performance Factors Machine Learning')

st.write('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/lyviavalentina/data/refs/heads/main/StudentPerformanceFactors%20(1).csv')
  df

