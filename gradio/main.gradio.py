import os

if 1:
    os.system("python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html")
    os.system("pip install --pre --upgrade paddlenlp==2.6.0rc0")
    os.system("pip install edge_tts")
    os.system("pip install pytest-runner")
    os.system("pip install paddlespeech")
    os.system("pip install numpy==1.21.6")
    os.system("pip install openvino")
    os.system("pip install soundfile")

import gradio as gr
import librosa
from utils.asr import preprocess_of_gradio_input, create_asr_model, preprocess_of_wav, asr
from utils.llm import load_chatGLM, glm_single_QA
# from utils.yiyan import yiyanchat

# global info
asr_model, output_ir = create_asr_model()
model, tokenizer = load_chatGLM()

def chat_via_audio(audio, input_text, chat_history, cb_use_textinput):
    # prepare the input
    if not cb_use_textinput:
        # speech2text
        preprocess_of_gradio_input(audio)
        audio = preprocess_of_wav('input_audio2.wav')
        transcription = asr(asr_model, output_ir, audio)
        input_text = transcription
    else:
        input_text = input_text
    # text generate
    try:
        try:
            output_text=glm_single_QA(model,tokenizer,'Please chat with me in English.' + input_text,2048,2048)
        except:
            output_text = yiyanchat('Please chat with me in English.' + input_text)
    except:
        output_text = "None reply"
    # text2speech
    output_text2 = "System: " + "".join(filter(lambda x: (str.isalnum(x) or x==' '), output_text))
    tts_cmd = 'edge-tts --text "' + output_text2 + '" --write-media output_audio.wav'
    os.system(tts_cmd)
    data, sr = librosa.load(path="output_audio.wav")
    # update chatbox
    chat_history.append((input_text, output_text))
    return (sr, data), chat_history

def audio_text_input_change_function(use_text):
    text_visible = use_text
    audio_visible = not use_text
    return gr.update(visible=audio_visible), gr.update(visible=text_visible)

def fn_generate_topic():
    try:
        prompts = 'Please give me a random topic for two person chat, please output only one sentence with English'
        try:
            topic_text = glm_single_QA(model,tokenizer,prompts,2048,2048)
        except:
            topic_text = yiyanchat(prompts)
    except:
        topic_text = 'None topic generated'
    return topic_text

def fn_make_comment(text, audio):
    # speech2text
    preprocess_of_gradio_input(audio)
    audio = preprocess_of_wav('input_audio2.wav')
    transcription = asr(asr_model, output_ir, audio)
    input_text = transcription

    # text generate
    Prompts = '我想要朗读的文本是：' + text + "。朗读后，语音转文字的结果是" + input_text + "。请结合上述信息，对我的发音进行评价。并按照满分10分，给出打分。"
    try:
        try:
            output_text=glm_single_QA(model,tokenizer,Prompts,2048,2048)
        except:
            output_text = yiyanchat(Prompts)
    except:
        output_text = "Fail to make score"
    return output_text

def fn_correct_asr(audio):
    # speech2text
    preprocess_of_gradio_input(audio)
    audio = preprocess_of_wav('input_audio2.wav')
    transcription = asr(asr_model, output_ir, audio)
    input_text = transcription

    # text generate
    Prompts = '我在朗读英语，我的发音或者语音识别软件可能有点问题。请尝试将识别结果修正为我想要表达的内容。语音识别内容是' + input_text + "。请对识别结果进行修正。请不要重复我对你的要求，也不要对语音内容进行回复，输出修正后的识别结果即可。"
    try:
        try:
            output_text=glm_single_QA(model,tokenizer,Prompts,2048,2048)
        except:
            output_text = yiyanchat(Prompts)
    except:
        output_text = "Fail to correct"
    return input_text, output_text


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 基于ChatGLM的语言练习台

        这是一个基于ChatGLM的语言练习台，主要的目标是营造一种语言的使用场景。例如你可以在这里通过英语和机器人对话，练习你的听力和文字表达。
        """
    )
    with gr.Tab('Chat'):
        gr.Markdown(
        """
        项目主框架，目前只是样本，可以试着玩一玩，留下宝贵的评论。但请不要以最终品质要求目前体验结果，因为多轮对话和各种Prompt机制暂时都没做~
        """
        )
        with gr.Accordion("主题生成", open=False):
            with gr.Row():
                with gr.Column(scale=3):
                    text_background = gr.Textbox(label="对话主题")
                with gr.Column(scale=1):
                    btn_generate_background = gr.Button(value="生成主题")
        audio_input = gr.Microphone(label="Your Voice")
        text_input = gr.Textbox(label="请输入文本",visible=False)
        audio_output = gr.Audio(label="Output Voice")
        with gr.Row():
            btn = gr.Button(value="Send")
            cb_use_textinput = gr.Checkbox(label='Text input')
            cb_show_history = gr.Checkbox(label='Show History')
        chatbot = gr.Chatbot(label="对话记录", visible=False)
        
        btn_generate_background.click(fn=fn_generate_topic, inputs=None, outputs=text_background)
        btn.click(fn=chat_via_audio, inputs=[audio_input, text_input, chatbot, cb_use_textinput], outputs=[audio_output, chatbot])
        cb_show_history.change(lambda x: gr.update(visible=x), inputs=cb_show_history, outputs=chatbot)
        cb_use_textinput.change(audio_text_input_change_function, inputs=cb_use_textinput, outputs=[audio_input, text_input])
    with gr.Tab('Score'):
        gr.Markdown(
            """
            这是一个实验室性质的章节。功能是使用LLM基于文本和语音识别内容进行打分，对你的发音进行评价。

            请输入一段英文，并朗读它，点击按钮后，LLM会评价你的发音。
            """
        )
        text_input_score = gr.Textbox(label="请输入文本")
        audio_input_score = gr.Microphone(label="请读出文本内容")
        text_output_score = gr.Textbox(label="评价")
        btn_score = gr.Button(value="进行评价")
        btn_score.click(fn=fn_make_comment, inputs=[text_input_score, audio_input_score], outputs=[text_output_score])
    with gr.Tab('Correct'):
        gr.Markdown(
            """
            这是一个实验室性质的章节。由于一些语音识别模型效果不是特别好，因此语音输入对话的准确性难以得到保证，本章节的功能是使用LLM对语音识别内容进行修正。

            请朗读一段英文，点击按钮后，ASR会识别你的英文，LLM会修正识别结果。
            """
        )
        audio_input_correct = gr.Microphone(label="请朗读一句话")
        text_output_correct_asr = gr.Textbox(label="识别结果")
        text_output_correct_llm = gr.Textbox(label="修正结果")
        btn_correct = gr.Button(value="进行修正")
        btn_correct.click(fn=fn_correct_asr, inputs=[audio_input_correct], outputs=[text_output_correct_asr, text_output_correct_llm])


if __name__ == "__main__":
    demo.launch()
