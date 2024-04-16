import streamlit as st
from streamlit_tensorboard import st_tensorboard
import subprocess
import socket
import webbrowser

# def find_free_port():
#         """查找可用的端口号"""
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(('localhost', 0))
#         port = s.getsockname()[1]
#         s.close()
#         st.success(f"TensorBoard running on port {port}")
#         return port

class TensorBoardComponent:
    def __init__(self, logdir, port=None):
        self.logdir = logdir
        self.port = port

    def find_free_port(self):
        """查找可用的端口号"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def start_tensorboard(self):
        """启动TensorBoard"""
        if self.port is None:
            self.port = self.find_free_port()
        command = f"tensorboard --logdir={self.logdir} --port={self.port}"
        subprocess.Popen(command, shell=True)

    def open_tensorboard(self):
        """打开TensorBoard网页"""
        url = f"http://localhost:{self.port}"
        webbrowser.open(url)

    def render(self):
        """渲染TensorBoard组件"""
        self.start_tensorboard()
        self.open_tensorboard()
        st.success(f"TensorBoard running on port {self.port}")

def app():
     with st.container():
        
      # tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2 限定多个log地址
        logdir = 'outputs'
        
        tensorboard_component = TensorBoardComponent(logdir)
        tensorboard_component.render()
        # st_tensorboard(logdir=logdir,port=find_free_port(),width=1080)