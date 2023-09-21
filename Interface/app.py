import streamlit as st
import os
import imageio
import tensorflow as tf
import numpy as np
from util import load_data,num_to_char
from moduleutil import load_model

st.set_page_config(layout='wide')

with st.sidebar:
  st.image('/content/isro.jpg')
  st.title("Rocket")
  st.info("This is Chandrayan 3,")

st.title("Deep Reading")
options=[file for file in os.listdir(os.path.join('drive','MyDrive','s1')) if file.endswith(('.mpg'))]
selected_video=st.selectbox('select a value:',options)

col1,col2=st.columns(2)
if options:
  with col1:
    st.info('The Video Below is Converted from mpg to mp4')
    file_path=os.path.join('drive','MyDrive','s1',selected_video)
    os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
    video=open('test_video.mp4','rb')
    video_bytes=video.read()
    st.video(video_bytes)

    
    
  with col2:
    st.info('Concentrated Area of Lips')
    video,annotations=load_data(tf.convert_to_tensor(file_path))
    imageio.mimsave('animation.gif',tf.squeeze(video*255,axis=-1),duration=10)
    st.image('animation.gif',width=400)
    st.info('Out of the model as a token')
    model=load_model()
    y_hat=model.predict(tf.expand_dims(video,axis=0))
    decoder=tf.keras.backend.ctc_decode(y_hat,[75],greedy=True)[0][0]
    st.text(decoder)
    st.info('Decoding tokens to Characters')
    converted_pred=tf.strings.reduce_join(num_to_char(decoder)).numpy()
    st.text(converted_pred)