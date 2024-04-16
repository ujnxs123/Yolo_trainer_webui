import streamlit as st
from multipage import MultiPage
from page import Tensorboard, home, pre_action,train,protect

st.set_page_config(page_title="ML", page_icon="res\start.ico", layout="wide")
st.title('Machine Learning Application')

app = MultiPage()

# st.page_link("your_app.py", label="Home", icon="🏠")
# st.page_link("pages/page_1.py", label="Page 1", icon="1️⃣")
# st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)
# st.page_link("http://www.google.com", label="Google", icon="🌎")
# add applications
app.add_page('Home', home.app)
app.add_page('Preprocessing', pre_action.app)
app.add_page('train',train.app)

app.add_page('TensorBoard',Tensorboard.app)
app.add_page("ProTect",protect.app)
# Run application
if __name__ == '__main__':
    app.run()