import streamlit as st
# from multipage import MultiPage
# from page import Tensorboard, home, pre_action,train,protect

# ------------------------------------------------------------------------------------------
st.set_page_config(page_title="ML", page_icon="res\home).ico", layout="centered")
st.title('Machine Learning Application')
with st.container():
        st.write("---")
        st.header("Blender tutorials")
        st.write("##")
    
        st.subheader("The explosion ball")
        st.write(
            """
            Learn how to model a explosion ball!
            In this tutorial, I'll show you exactly how to do it
            """
        )
        st.markdown("[Watch Video...](https://www.bilibili.com/video/BV1DK411H795)")
        st.write(""".pt类型的文件是从预训练模型的基础上进行训练,
                    若我们选择 yolov8n.pt这种.pt类型的文件，
                    其实里面是包含了模型的结构和训练好的参数的，
                    也就是说拿来就可以用，就已经具备了检测目标的能力了，
                    yolov8n.pt能检测coco中的80个类别。假设你要检测不同种类的狗，
                    那么yolov8n.pt原本可以检测狗的能力对你训练应该是有帮助的，
                    你只需要在此基础上提升其对不同狗的鉴别能力即可。
                    但如果你需要检测的类别不在其中，例如口罩检测，
                    那么就帮助不大。
                    .yaml文件是从零开始训练。采用yolov8n.yaml这种.yaml文件的形式，在文件中指定类别，以及一些别的参数。""")
        

        

# 官方给的多页设置，可以更个性化设置页面
# st.page_link(home.app, label="Home", icon="🏠")
# st.page_link(pre_action.app, label="Page 1", icon="1️⃣")
# st.page_link(train.app, label="Page 2", icon="2️⃣", disabled=True)
# st.page_link(protect.app, label="Google", icon="🌎")
# add applications










# -----------------------------------------------------------------------------------------------------------------------------------------------------
# 分页控制类脚本，以select_box的形式
# st.set_page_config(page_title="ML", page_icon="res\start.ico", layout="wide")
# st.title('Machine Learning Application')
# app = MultiPage()
# app.add_page('Home', home.app)
# app.add_page('Preprocessing', pre_action.app)
# app.add_page('train',train.app)

# app.add_page('TensorBoard',Tensorboard.app)
# app.add_page("ProTect",protect.app)
# # Run application
# if __name__ == '__main__':
#     app.run()