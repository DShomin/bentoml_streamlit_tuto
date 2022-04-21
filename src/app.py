import streamlit as st

import requests
import io

from utils import load_cls_list, text2list, load_image


def post_data(url, data: dict):
    r = requests.post(url, headers={"content-type": "application/json"}, json=data)
    return r.json()


def post_image_file(url, img):
    byte_io = io.BytesIO()
    img.save(byte_io, "png")
    byte_io.seek(0)
    r = requests.post(url, files={"files[]": ("1.png", byte_io, "image/png")},)
    return r.json()


st.sidebar.header("Select service")
name = st.sidebar.selectbox("Service", ["Iris", "ImageNet"])

if name == "Iris":
    st.title("Iris classification")
    with st.form("my_form"):
        data = st.text_input("input number", "[5, 2, 3, 3]")
        st.write("input number", data)
        data = text2list(data)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Result")
            result = post_data("http://127.0.0.1:3000/classifier_iris", data)
            if result:
                st.text(result)
            else:
                st.error("Error")
else:
    st.title("ImageNet classification")

    with st.form("my_form"):
        image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
        if not image_file is None:
            img = load_image(image_file)
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Result")
            st.image(img)
            result = post_image_file("http://127.0.0.1:3000/classifier_imagenet", img)
            if result:
                idx2label = load_cls_list()
                for idx, score in result.items():
                    st.text(f"{idx2label[int(idx)]} : {score}")
            else:
                st.error("Error")
