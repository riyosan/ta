import streamlit as st
st.set_page_config(page_title='Skripsi', layout='wide')
from PIL import Image
from multiapp import MultiApp
from apps import praproses, data, prediksi # import your app modules here

app = MultiApp()


image=Image.open('logo.png')
logo = st.columns((1.5, 0.5))

#page layout
with logo[1]:
	st.image(image,width=200)

with logo[0]:
	st.markdown("""
# Skripsi

Penggunaan Algoritma Stacking Ensemble Learning 
""")
	st.markdown("""
Dalam Memprediksi Pengguna Enroll.

""")
	st.markdown("""
""")
	st.markdown("""
**Riyo Santo Yosep - 171402020**

""")

# Add all your application here
app.add_app("Praproses", praproses.app)
app.add_app("Latih & Uji", data.app)
app.add_app("Prediksi", prediksi.app)
# The main app
app.run()
