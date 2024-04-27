import streamlit as st
from streamlit_tensorboard import st_tensorboard
import subprocess
import socket
import webbrowser
import psutil
# def find_free_port():
#         """查找可用的端口号"""
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(('localhost', 0))
#         port = s.getsockname()[1]
#         s.close()
#         st.success(f"TensorBoard running on port {port}")
#         return port


# 还是有老问题，第一次启动确实可以保证6006不会被占用，但是第二次第三次启动都会杀死6006再开启6006
class TensorBoardComponent:
    def __init__(self, logdir):
        self.logdir = logdir
        self.port = 6006

    def is_port_in_use(self):
        """检查端口是否被占用"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', self.port)) == 0

    def start_tensorboard(self):
        """启动TensorBoard"""
        if self.is_port_in_use():
            self.kill_tensorboard_process()  # 如果端口被占用，则杀死占用该端口的进程
        command = f"tensorboard --logdir={self.logdir} --port={self.port}"
        subprocess.Popen(command, shell=True)

    def kill_tensorboard_process(self):
        """杀死固定端口上的TensorBoard进程"""
        for proc in psutil.process_iter():
            try:
                if proc.name() == "tensorboard" and f"--port={self.port}" in proc.cmdline():
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    def open_tensorboard(self):
        """打开TensorBoard网页"""
        url = f"http://localhost:{self.port}"
        webbrowser.open(url)

    def render(self):
        """渲染TensorBoard组件"""
        self.start_tensorboard()
        # 只在需要时打开浏览器
        if self.is_port_in_use():
            self.open_tensorboard()
            st.success(f"TensorBoard running on port {self.port}")
            st.markdown(f"[http://localhost:{self.port}](http://localhost:{self.port})")
# -----------------------------------------------------------------------''
# app
#%%
st.set_page_config(page_title='Tensorboard展示',page_icon='res\start.ico')
st.title('TensorBoard启动')
with st.container():

# tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2 限定多个log地址
    logdir = 'outputs'
    if st.button('启动'):
        with st.spinner("running...."):
            tensorboard_component = TensorBoardComponent(logdir)
            tensorboard_component.render()
    # st_tensorboard(logdir=logdir,port=find_free_port(),width=1080)
        