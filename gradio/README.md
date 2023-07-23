# Gradio 部署页面
## 快速调试
1. 在AiStudio创建NoteBook，并进入环境。(本地也可以，但是需要自行检查有没有其他没有安装的Python库)
2. Clone本项目
3. 将 gradio/main.gradio.py 的前11行复制到NoteBook中运行。该步骤用于安装必要的包，本地自行安装即可。如果你使用CPU环境可以不用安装paddlepaddle和paddlenlp，这两个包仅用于部署，调试时不需要
4. 安装完成后，将第三行改为if 0:
5. 注释16、21行关于GLM的代码，取消注释第17行代码
6. 将speech2text_model拷贝到项目根目录，以Aistudio为例，需要拷贝到/home/aistudio下
7. 运行gradio脚本即可展示界面

**调试过程中使用文心一言API，请自行在文心千帆官方网站申请Key，并填入yiyan.py文件中**

## 关于ASR模型
具体信息参考 https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/211-speech-to-text 。

## 关于文心一言API
请自行申请API，替换对应的KEY即可。调试的时候建议直接注释GLM相关的Import代码和模型创建代码，使用Yiyan进行快速体验和调试。
