


import streamlit as st


st.set_page_config(page_title='申诉',layout='centered',page_icon='res\justice.ico')

st.title('申诉页面')

st.header('流程1')

video_file = st.file_uploader("上传视频", type=['mp4'])
# col1,col2,col3 = st.columns(3)


if st.button('提交'):
    if video_file is None:
        st.warning("请先上传视频！")

    else:
        
        st.video(video_file, start_time=0)   
        st.success("提交成功！")