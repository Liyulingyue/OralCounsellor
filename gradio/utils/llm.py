import paddle
from paddlenlp.transformers import (
    ChatGLMConfig,
    ChatGLMForConditionalGeneration,
    ChatGLMTokenizer,
)

def load_chatGLM():
    #读取原始的chatglm-6b模型
    # model_name_or_path = '/home/aistudio/chatglm-6b/chatglm-merged'
    model_name_or_path = 'THUDM/chatglm-6b'
    tokenizer = ChatGLMTokenizer.from_pretrained(model_name_or_path)
    config = ChatGLMConfig.from_pretrained(model_name_or_path)
    # paddle.set_default_dtype(config.paddle_dtype)
    # 通常用fp16更好，但是不知道为啥fp16跑不通
    # config.paddle_dtype = 'float32'
    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_name_or_path,
        tensor_parallel_degree=paddle.distributed.get_world_size(),
        tensor_parallel_rank=0,
        load_state_as_np=True,
        dtype=config.paddle_dtype,
    )
    model.eval()
    return model, tokenizer

# 函数定义，用于一问一答
# 输入参数：初始prompt, 最长输入长度，最长输出长度
def glm_single_QA(model,tokenizer,next_inputs,input_length,output_length):
    # 输入格式转换
    inputs = tokenizer(
        next_inputs,
        return_tensors="np",
        padding=True,
        max_length=input_length,
        truncation=True,
        truncation_side="left",
    )
    input_map = {}
    for key in inputs:
        input_map[key] = paddle.to_tensor(inputs[key])

    # 获取结果
    infer_result = model.generate(
        **input_map,
        decode_strategy="sampling",
        top_k=1,
        # top_p =5,
        max_length=output_length,
        use_cache=True,
        use_fast=True,
        use_fp16_decoding=True,
        repetition_penalty=1,
        temperature = 0.95,
        length_penalty=1,
    )[0]

    # 结果转换
    output = ''
    result = []
    for x in infer_result.tolist():
        res = tokenizer.decode(x, skip_special_tokens=True)
        res = res.strip("\n")
        result.append(res)
        output = output + res
    return output