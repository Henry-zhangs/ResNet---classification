import streamlit as st
from PIL import Image
from predict import predict


st.set_option('deprecation.showfileUploaderEncoding', False)

st.title(" CIFAR10 Image Classification ")
st.write("")
st.write("Category：'plane','car','bird','cat','deer','dog','frog','horse','ship','truck'")
st.write("")
st.write("")

file_up = st.file_uploader("Upload an image")

if file_up is None:
    image =r'D:\pycharm\python_study\ResNet\image\img.png'
    img = Image.open(image)
    st.image(img, caption='Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")

    label, ret = predict(image)   # 预测图片
    st.success('successful prediction')
    for i in range(5):
        st.write("Prediction\t", label[i], ",\tScore: ", ret[i])
        st.write("")

else:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    label,ret = predict(file_up)

    st.success('successful prediction')
    for i in range(5):
        st.write("Prediction\t", label[i], ",\tScore: ", ret[i])
        st.write("")


