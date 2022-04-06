import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser
import joblib

@st.cache(allow_output_mutation=True)
def load_dataset(data_pred):
  df = pd.read_csv(data_pred)
  return df

@st.experimental_memo
def preprocessing_pred(df):
  df['screen_list'] = df.screen_list.astype(str) + ','
  df['num_screens'] = df.screen_list.astype(str).str.count(',')
  #menghapus kolom numsreens yng lama
  df.drop(columns=['numscreens'], inplace=True)
  #mengubah kolom hour
  df.hour=df.hour.str.slice(1,3).astype(int)
  #karena tipe data first_open itu adalah string, maka perlu diubah ke datetime
  df.first_open=[parser.parse(i) for i in df.first_open]
  #import top_screen
  top_screens=pd.read_csv('data/top_screens.csv')
  top_screens=np.array(top_screens.loc[:,'top_screens'])
  for i in top_screens:
      df[i]=df.screen_list.str.contains(i).astype(int)
  for i in top_screens:
      df['screen_list']=df.screen_list.str.replace(i+',','')
  #menghitung jumlah item non top screen yang(tersisa) ada di screenlist
  df['lainnya']=df.screen_list.str.count(',')
  #menghapus double layar
  layar_loan = ['Loan','Loan2','Loan3','Loan4']
  df['jumlah_loan']=df[layar_loan].sum(axis=1)
  df.drop(columns=layar_loan, inplace=True)

  layar_saving = ['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']
  df['jumlah_loan']=df[layar_saving].sum(axis=1)
  df.drop(columns=layar_saving, inplace=True)

  layar_credit = ['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']
  df['jumlah_credit']=df[layar_credit].sum(axis=1)
  df.drop(columns=layar_credit, inplace=True)

  layar_cc = ['CC1','CC1Category','CC3']
  df['jumlah_cc']=df[layar_cc].sum(axis=1)
  df.drop(columns=layar_cc, inplace=True)
  #mendefenisikan variabel numerik
  pred_numerik=df.drop(columns=['first_open','screen_list','user'], inplace=False)
  scaler = joblib.load('data/standar.joblib')
  fitur = pd.read_csv('data/fitur_pilihan.csv')
  fitur = fitur['0'].tolist()
  pred_numerik = pred_numerik[fitur]
  pred_numerik = scaler.transform(pred_numerik)
  model = joblib.load('data/stack_model.pkl')
  prediksi = model.predict(pred_numerik)
  probabilitas = model.predict_proba(pred_numerik)
  user_id = df['user']
  prediksi_akhir = pd.Series(prediksi)
  hasil_akhir= pd.concat([user_id,prediksi_akhir], axis=1).dropna()
  return probabilitas, hasil_akhir
def app():
  # if "load" not in st.session_state:
  #   st.session_state.load = False
  with st.sidebar.header('3. Predict'):
    data_pred = st.sidebar.file_uploader("Unggah File CSV",type=['csv'], key="file_uploader")
    # st.session_state.file_uploader
  if data_pred is not None:
    st.session_state.file_uploader
    df = load_dataset(data_pred)
    st.dataframe(df)
    if "load" not in st.session_state:
      st.session_state.load = False
    if st.sidebar.button("predict Data") or st.session_state.load:
      st.session_state.load = True
      df = load_dataset(data_pred)
      st.dataframe(df)
      hasil_akhir, probabilitas = preprocessing_pred(df)
      layout = st.columns((1,1,1,1))
      with layout[1]:
        st.write(hasil_akhir)
      with layout[2]:
        st.write(probabilitas)
    # if data_pred:
      # if 'load_csv' in st.session_state:
      #   df=st.session_state.load_csv
      #   st.write(data_pred.name + " " +  "is loaded")
      # else:
      #   df = load_csv(data_pred)
      #   st.session_state.load_csv = df
      # df = load_dataset(data_pred)
      # st.dataframe(df)
      # if "load" not in st.session_state:
      #   st.session_state.load = False
      # if st.sidebar.button("predict Data") or st.session_state.load:
      #   st.session_state.load = True
      #   df = load_dataset(data_pred)
      #   st.dataframe(df)
      #   hasil_akhir, probabilitas = preprocessing_pred(df)
      #   layout = st.columns((1,1,1,1))
      #   with layout[1]:
      #     st.write(hasil_akhir)
      #   with layout[2]:
      #     st.write(probabilitas)

  # else:
  #   if "state" not in st.session_state:
  #     st.session_state.state = False
  #   if st.button('Press to use Example Dataset') or st.session_state.state:
  #     st.session_state.state = True
  #     df = pd.read_csv('testing.csv')
  #     st.dataframe(df)
  #     # df = load_dataset(data_pred)
  #     # st.dataframe(df)
  #     hasil_akhir, probabilitas = preprocessing_pred(df)
  #     layout = st.columns((1,1,1,1))
  #     with layout[1]:
  #       st.write(hasil_akhir)
  #     with layout[2]:
  #       st.write(probabilitas)
