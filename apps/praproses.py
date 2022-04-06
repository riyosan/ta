import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser

# @st.experimental_singleton(suppress_st_warning=True)
# def upload_dataset():
#   # data=st.sidebar.file_uploader("Unggah File CSV", type=['CSV']).header('1. Unggah File CSV')
#   with st.sidebar.header('1. Unggah File CSV'):
#     data=st.sidebar.file_uploader("Unggah File CSV",type=['csv'])
#   return data


@st.experimental_memo
def load_dataset():
  df = pd.read_csv('data/fintech_data.csv')
  return df

@st.experimental_memo
def load_datasets(dataset):
  df = pd.read_csv(dataset)
  return df

@st.experimental_memo
def preprocessing():
  df=load_dataset()
  #menghitung ulang isi dari kolom screen_litst karena tidak pas di kolom num screens
  df['screen_list'] = df.screen_list.astype(str) + ','
  df['num_screens'] = df.screen_list.astype(str).str.count(',')
  return df

# @st.experimental_memo
# def preprocessings(df):
#   # df=load_dataset()
#   #menghitung ulang isi dari kolom screen_litst karena tidak pas di kolom num screens
#   df['screen_list'] = df.screen_list.astype(str) + ','
#   df['num_screens'] = df.screen_list.astype(str).str.count(',')
#   return df

@st.experimental_memo
def preprocessing1():
  df1 = preprocessing()
  # df1=df.copy()
  #feature engineering
  #karena kolom hour ada spasinya, maka kita ambil huruf ke 1 sampai ke 3
  df1.hour=df1.hour.str.slice(1,3).astype(int)
  #karena tipe data first_open dan enrolled_date itu adalah string, maka perlu diubah ke datetime
  df1.first_open=[parser.parse(i) for i in df1.first_open]
  #didalam dataset orang yg belum langganan itu NAN, maka jika i=string biarin, klo ga string diuah ke datetime kolom nan nya biarin tetap nat
  df1.enrolled_date=[parser.parse(i) if isinstance(i, str)else i for i in df1.enrolled_date]
  #membuat kolom selisih , yaitu menghitung berapa lama orang yg firs_open menjadi enrolled
  df1['selisih']=(df1.enrolled_date-df1.first_open).astype('timedelta64[h]')
  #karna digrafik menunjukkan orang kebanyakan enroll selama 24 jam pertama, maka kalau lebih dari 24 jam dianggap ga penting
  df1.loc[df1.selisih>24, 'enrolled'] = 0
  return df1

@st.experimental_memo
def preprocessing2():
  df2=preprocessing1()
  # b=df2['screen_list'].apply(pd.Series).stack()
  # c = b.tolist()
  # from collections import Counter
  # p = Counter(' '.join(b).split()).most_common(100)
  # rslt = pd.DataFrame(p)
  # rslt.to_csv('data/top_screens.csv', index=False)
  top_screens=pd.read_csv('data/top_screens.csv')
  # diubah ke numppy arry dan mengambil kolom ke2 saja karna kolom1 isinya nomor
  top_screens=np.array(top_screens.loc[:,'top_screens'])
  df3 = df2.copy()
  #mengubah isi dari file top screen menjadi numerik
  for i in top_screens:
    df3[i]=df3.screen_list.str.contains(i).astype(int)
  #semua item yang ada di file top screen dihilangkan dari kolom screen list
  for i in top_screens:
    df3['screen_list']=df3.screen_list.astype(str).str.replace(i+',','')
  #menghitung jumlah item non top screen yang(tersisa) ada di screenlist
  df3['lainnya']=df3.screen_list.astype(str).str.count(',')
  return df3

@st.experimental_memo
def funneling():
  df=preprocessing2()
  #menggabungkan item yang mirip mirip, seperti kredit 1 kredit 2 dan kredit 3
  #funneling = menggabungkan beberapa screen yang sama dan menghapus layar yang sama
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
  #menghilangkan kolom yang ga relevan
  df_numerik=df.drop(columns=['user','first_open','screen_list','enrolled_date','selisih','numscreens'], inplace=False)
  df_numerik.to_csv('data/data_praproses.csv', index=False)
  from sklearn.feature_selection import mutual_info_classif
  #determine the mutual information
  mutual_info = mutual_info_classif(df_numerik.drop(columns=['enrolled']), df_numerik.enrolled)
  mutual_info = pd.Series(mutual_info)
  mutual_info.index = df_numerik.drop(columns=['enrolled']).columns
  mutuals = mutual_info.sort_values(ascending=False)
  return df_numerik, mutuals
  # korelasi = df_numerik.drop(columns=['enrolled'], inplace=False).corrwith(df_numerik.enrolled)
  # plot=korelasi.plot.bar(title='korelasi variabel')
  # st.set_option('deprecation.showPyplotGlobalUse', False)
  # # st.pyplot()
  # # st.text('Membuat plot korelasi tiap koklom terhadap kelasnya(enrolled)')
  # from sklearn.feature_selection import mutual_info_classif
  # #determine the mutual information
  # mutual_info = mutual_info_classif(df_numerik.drop(columns=['enrolled']), df_numerik.enrolled)
  # mutual_info = pd.Series(mutual_info)
  # mutual_info.index = df_numerik.drop(columns=['enrolled']).columns
  # mutual_info.sort_values(ascending=False)
  # mutual_info.sort_values(ascending=False).plot.bar(title='urutannya')
  # # st.set_option('deprecation.showPyplotGlobalUse', False)
  # # st.pyplot()
  # # st.text('mengurutkan korelasi setiap kolom terhadap kelasnya(enrolled)')
  # df_numerik.to_csv('data/main_data.csv', index=False)
  # df1.to_csv('data/df1.csv', index=False)
  # return df_numerik
def app():
  # if "load_state" not in st.session_state:
  #   st.session_state.load_state = False
  # with st.sidebar.header('1. Unggah Dataset'):
  #   dataset = st.sidebar.file_uploader("Unggah File CSV", type=['csv'])
  # if dataset is not None:
  #   df=load_datasets(dataset)
  #   # st.dataframe(df)
  #   # if "load_state" not in st.session_state:
  #   #   st.session_state.load_state = False
  #   # if st.button('Preprocess Data') or st.session_state.load_state:
  #   if st.button('Preprocess Data'):
  #     # st.session_state.load_state = True
  #     df = preprocessing()
  #     container = st.columns((1.9, 1.1))
  #     df_types = df.dtypes.astype(str)
  #     with container[0]:
  #       st.write(df)
  #       st.text('merevisi kolom numscreens')
  #     with container[1]:
  #       st.write(df_types)
  #       st.text('Tipe data setiap kolom')
  # else:
  with st.sidebar.header('1. Load Dataset'):
    # st.text('Atau:\n ')
    # st.text('Load Dataset:\n ')
    load = st.sidebar.button('Press to use Example Dataset')
  if "load_state" not in st.session_state:
    st.session_state.load_state = False
  if load or st.session_state.load_state:
  # if load:
    st.session_state.load_state = True
    df=pd.read_csv('data/fintech_data.csv')
    st.dataframe(df)
    st.write(load_dataset())
    df=preprocessing()
    container = st.columns((1.9, 1.1))
    df_types = df.dtypes.astype(str)

    with container[0]:
      st.write(df)
      st.markdown('''
      Merevisi kolom numscreens''')
      # st.text('Merevisi kolom numscreens')
    with container[1]:
      st.write(df_types)
      st.markdown('''
      Merevisi kolom numscreens''')
      # st.text('Tipe data setiap kolom')
    
    df1=preprocessing1()
    container1 = st.columns((1.9, 1.1))
    df1_types = df1.dtypes.astype(str)
    with container1[0]:
      st.write(df1)
      st.text('Merevisi kolom hour')
    with container1[1]:
      st.write(df1_types)
      st.text('Tipe data setiap kolom')

    df4=preprocessing2()
    st.write(df4)
    st.text('Mengubah isi screen_list menjadi kolom baru')

    df_numerik, mutuals = funneling()
    st.write(df_numerik)
    #membuat plot korelasi tiap kolom dengan enrolled
    # korelasi = df_numerik.drop(columns=['enrolled'], inplace=False).corrwith(df_numerik.enrolled)
    # plot=korelasi.plot.bar(title='korelasi variabel')
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.pyplot()
    # st.text('Membuat plot korelasi tiap koklom terhadap kelasnya(enrolled)')
    # from sklearn.feature_selection import mutual_info_classif
    # #determine the mutual information
    # mutual_info = mutual_info_classif(df_numerik.drop(columns=['enrolled']), df_numerik.enrolled)
    # mutual_info = pd.Series(mutual_info)
    # mutual_info.index = df_numerik.drop(columns=['enrolled']).columns
    # mutual_info.sort_values(ascending=False)
    mutuals.plot.bar(title='urutannya')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.text('mengurutkan korelasi setiap kolom terhadap kelasnya(enrolled)')
