# OralCounsellor
一个基于大模型的口语对话顾问

## About Project
本项目的目标是构建一个英语练习环境。

项目的目标是能够构造一个应用，你可以自由地和机器人围绕某个话题使用特定语言进行对话，机器人能够像真人一样，尝试了解你支支吾吾的表达中蕴含的意思，并且尝试引导你进行正确的表达，这些引导内容包括直接给出正确的表达，并且跟你确认你的意图是否和它猜测的一样。机器人也能够推动话题的发展，而不是被动的接收你的说辞。从而最终起到提高语言表达能力的作用。

## 项目进展

当前项目属于第一阶段，目标是建立一个能够通过语音进行交互的Gradio界面，部署在[Aistudio](https://aistudio.baidu.com/aistudio/index)上。

部署页面的输入是语音，输出也是语音，包含一个对话框展示历史记录。

用户进入界面后能够基于某个随机生成的主题和机器人进行多轮对话，机器人拥有一定的语音质量评价能力和纠错能力。

Task列表如下：
- [ ] 接入ASR模块
- [x] 接入TTS模块
- [ ] 自动生成对话场景和人物设定，并进行多轮对话
- [ ] 语音输入质量评价
- [ ] 语音输入自动识错
- [x] 主模块Gradio界面
- [ ] 语音质量Gradio界面
- [ ] 语音识错Gradio界面
- [ ] 将语音质量评价应用于整个对话结束后的评价中，并构造Gradio界面
- [ ] 将自动识错应用于对话过程中

部分代码和功能在线调试环境见 [DEMO of OralCounsellor](https://aistudio.baidu.com/aistudio/projectdetail/6559166)

本阶段项目主要合作人如下：[@Liyulingyue](https://github.com/Liyulingyue/), [@mrcangye](https://github.com/mrcangye/), [@ccsuzzh](https://github.com/ccsuzzh/), [@gouzil](https://github.com/gouzil/), [@Tomoko-hjf](https://github.com/Tomoko-hjf/)

## 项目计划

### 第一阶段
基于文心一言等大模型，构建基于Prompt信息补全工程。

初步目标是能够创建具有某种性格的角色，并且围绕指定主题进行文本对话。

### 第二阶段
基于文心一言等大模型，构建基于Prompt信息补全工程。

第二阶段目标是能够创建具有某种性格的角色，并且围绕指定主题进行对话，该对话能够识别表达中不清晰的部分，并且识别对应错误，进行反馈，与纠错与引导。

### 第三阶段
基于文心一言API或ChatGLM微调。

从LLM应用层面，希望第三阶段比第二阶段更为深化。

从LLM应用前处理层面，希望第三阶段能够接入`speech-to-text`模型，使用语音输入替代文字输入。

从LLM应用后处理层面，希望第三阶段能够接入`text-to-speech`模型，使用语音输出替代文字输出。

### 第三阶段
基于文心一言API或ChatGLM微调。

从LLM应用层面，希望第三阶段比第二阶段更为深化。能够增加对话人数，并且能够通过提示词等设置，控制用户在对话中的参与程度。

### 第四阶段
基于文心一言API或ChatGLM微调，辅助前端界面封装。

能够支持各种对话难度和语言表达方式。并且封装界面，不暴露底层代码。

### 第五阶段（暂定）
能够支持多种语言练习。

## 项目招聘
该项目属于一个非盈利性质的创意项目，整体内容也较为粗糙，我们欢迎任何`Just for fun`的朋友加入和共创。目前缺少系统架构师、项目经理、产品经理、AI后端工程师、前端工程师等！

希望加入此项目的话在本项目提交Issue，留下微信或者邮件联系方式即可。

