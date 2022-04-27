import streamlit as st
import pandas as pd

## Title
st.title("Ini Title")

## Header
st.header("Ini Header")

## Sunheader
st.subheader("Ini Subheader")

## Text
st.text("Ini Text: Hai semua, saya lagi magang di Nanosense")

## Markdown
st.markdown("# Heading 1")
st.markdown("## Heading 2")
st.markdown("### Heading 3")

## Code
code = '''
def hello():
    print("Hello, Streamlit!")
    dhdh
'''
st.code(code, language='python')

## Success Status
st.success("Berhasil!")

## Info Status
st.info("Ini sebuah informasi")

## Warning Status
st.warning("Ini Warning")

## Error Status
st.error("Ini Error ")

## Write
st.write("Ini hasil print write: string.")

## Menampilkan gambar
from PIL import Image
img = Image.open("K:/Nanosense/Software/2.pyside/Streamlit/gambarku.jpg")
st.image(img, width=300, caption="Ini caption gambar")


## Checkbox
agree = st.checkbox('Coba Centang')
if agree:
     st.write('Mantappp')

## Radio checkbox
genre = st.radio(
     "What's your favorite movie genre",
     ('Comedy', 'Drama', 'Documentary'))

if genre == 'Comedy':
     st.write('You selected comedy.')
else:
     st.write("You didn't select comedy.")

## Select Box
option = st.selectbox(
     'How would you like to be contacted?',
     ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)

## Multiselect
options = st.multiselect(
     'What are your favorite colors',
     ['Green', 'Yellow', 'Red', 'Blue']) 

st.write('You selected:', options)

## slider (Interger Number)
age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')

## slider (Float Number)
age = st.slider('How old are you?', 0.0, 130.0, 25.0)
st.write("I'm ", age, 'years old')

## slider (Range)
values = st.slider(
     'Select a range of values',
     0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)

## Upload File
uploaded_files = st.file_uploader("Choose a CSV file")
if uploaded_files is not None:
    data = pd.read_csv(uploaded_files)
    st.dataframe(data)


