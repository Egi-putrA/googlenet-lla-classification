import streamlit as st
images_uploaded = st.file_uploader("Upload file", type=['jpg', 'png'], accept_multiple_files=True)

for image in images_uploaded:
    print(image)
    st.image(image)
