import time
from json import JSONDecodeError
from openai import OpenAI
from openai import (
    AuthenticationError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    BadRequestError,
    APIError,
)
from utils.get_api_key import get_api_key
from utils.get_base_url import get_base_url

vonder = 'deepseek'
API_KEY = get_api_key(vonder)
BASE_URL = get_base_url(vonder)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL,)

# 模型重定向
def model_redirect(model):
    if vonder == 'bytedance':
        if model == "deepseek-r1":
            return "deepseek-r1-250120"
        elif model == "deepseek-v3":
            return "deepseek-v3-241226"
        else:
            return model
    elif vonder == "deepseek":
        if model == "deepseek-r1":
            return "deepseek-reason"
        elif model == "deepseek-v3":
            return "deepseek-chat"
        else:
            return model
    else:
        return model
def one_chat(message, model="deepseek-v3", temperature=0.5, timeout=300, max_tokens=None, functions=None, function_call='auto'):
    """
    与LLM进行一次对话, 不设置max_token因为有的可能会超
    """
    
    if functions:
        # 如果使用了函数调用, 则传入对应参数
        completion = client.chat.completions.create(
                    model=model,
                    messages=message,
                    timeout=timeout,
                    # max_tokens=max_tokens,
                    temperature=temperature,
                    functions=functions,
                    function_call=function_call,
                )
        # 生成内容
        content = completion.choices[0].message.content
        # token 相关信息
        usage = completion.usage
        # 函数调用信息
        function_call = completion.choices[0].message.function_call
        return content, function_call, usage
    
    else:
        # 正常传参
        completion = client.chat.completions.create(
                    model=model,
                    messages=message,
                    timeout=timeout,
                    # max_tokens=max_tokens,
                    temperature=temperature,
                )
        # 获取tokens使用情况
        usage = completion.usage
        # 获取模型文本输出
        content = completion.choices[0].message.content

        if model == "deepseek-r1":
            # 获取模型推理过程
            reasoning_content = completion.choices[0].message.reasoning_content
            return content, reasoning_content, usage
        
        else:
            return content, usage
        

def one_completion(message, model="deepseek-v3", temperature=0.5, timeout=300, max_tokens=None, functions=None, function_call='auto'):
    """
    与LLM进行一次对话, 不设置max_token因为有的可能会超
    需要自己处理返回的completion
    """

    if functions:
        # 如果使用了函数调用, 则传入对应参数
        completion = client.chat.completions.create(
                    model=model,
                    messages=message,
                    timeout=timeout,
                    # max_tokens=max_tokens,
                    temperature=temperature,
                    functions=functions,
                    function_call=function_call,
                )
    else:
        # 正常传参
        completion = client.chat.completions.create(
                    model=model,
                    messages=message,
                    timeout=timeout,
                    # max_tokens=max_tokens,
                    temperature=temperature,
                ) 
    return completion

def create_chat_completion(messages,model="deepseek-v3",temperature=0.5,timeout=360,max_tokens=8192,tools=None,tool_choice='auto'):
    """
    带异常处理的API调用函数, 有自动重试机制
    返回结果(success, result)
    - success=True 时 result 为 completion 对象
    - success=False 时 result 为错误字典
    """

    max_retry = 4
    retry_count = 0
    base_delay = 1 # 基础等待时间

    while retry_count < max_retry:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                # response_format={
                #     'type': 'json_object'
                # }
                # tools=tools,
                # tool_choice=tool_choice,   
            )

            if completion.choices[0].finish_reason == "length":
                error_info = {
                    "error_code": "MAX_OUTPUT_LENGTH",
                    "message": "达到模型输出最大长度",
                    "retryable": False
                }
                return (False, error_info)
            else:
                return (True, completion)
        
        except AuthenticationError as e:
            error_info = {
                # "error_code": "AUTHENTICATION_ERROR",
                "error_code": e,
                "message": "API 认证失败，请检查 API KEY 和权限",
                "retryable": False
            }
            return (False, error_info)
        except BadRequestError as e:
            error_info = {
                # "error_code": "BadRequestError",
                "error_code": e,
                "message": f"无效请求参数: {str(e)} or 检查网络问题",
                "retryable": False
            }
            return (False, error_info)
        except APITimeoutError as e:
            error_info = {
                # "error_code": "TIMEOUT",
                "error_code": e,
                "message": f"请求超时({timeout}s)",
                "retryable": True
            }
        except RateLimitError as e:
            error_info = {
                # "error_code": "RATE_LIMIT",
                "error_code": e,
                "message": "请求频率过高，触发速率限制",
                "retryable": True
            }
        except APIConnectionError as e:
            error_info = {
                # "error_code": "API_CONNECTION_ERROR",
                "error_code": e,
                "message": "API连接错误",
                "retryable": True
            }
        except APIError as e:
            error_info = {
                # "error_code": "API_ERROR",
                "error_code": e,
                "message": "API服务暂时不可用",
                "retryable": True
            } 
        except JSONDecodeError as e:
            error_info = {
                # "error_code": "JSON_DECODE_ERROR",
                "error_code": e,
                "message": "deepseek API JSON解析错误",
                "retryable": True
            }
        except Exception as e:
            error_info = {
                # "error_code": "UNKNOWN_ERROR",
                "error_code": e,
                "message": f"未知错误: {str(e)}",
                "retryable": False
            }
            return (False, error_info)
        # 处理可重试的错误
        if error_info.get("retryable", False):
            if retry_count < max_retry:
                # 指数退避 + 随机抖动
                delay = base_delay * (2 ** retry_count) + 0.1 * retry_count
                time.sleep(delay)
                retry_count += 1
                continue

        break  # 不可重试错误或达到最大重试次数
    
    return (False, error_info)

def dp_official_create_chat_completion(messages,model="deepseek-v3",temperature=0.5,timeout=360,max_tokens=8192,tools=None,tool_choice='auto'):
    vonder = 'deepseek'
    API_KEY = get_api_key(vonder)
    BASE_URL = get_base_url(vonder)
    local_client = OpenAI(api_key=API_KEY, base_url=BASE_URL,)
    max_retry = 4
    retry_count = 0
    base_delay = 1 # 基础等待时间

    while retry_count < max_retry:
        try:
            completion = local_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                tools=tools,
                tool_choice=tool_choice,   
            )
            # 如果生成了函数调用
            if completion.choices[0].message.tool_calls:
                return (True, completion)
            else:
                error_info = {
                    "error_code": "未生成函数调用",
                    "message": "LLM返回的函数调用为空, 请检查LLM的配置是否正确",
                    "retryable": True
                }
        
        except AuthenticationError as e:
            error_info = {
                # "error_code": "AUTHENTICATION_ERROR",
                "error_code": e,
                "message": "API 认证失败，请检查 API KEY 和权限",
                "retryable": False
            }
            return (False, error_info)
        except BadRequestError as e:
            error_info = {
                # "error_code": "BadRequestError",
                "error_code": e,
                "message": f"无效请求参数: {str(e)} or 检查网络问题",
                "retryable": False
            }
            return (False, error_info)
        except APITimeoutError as e:
            error_info = {
                # "error_code": "TIMEOUT",
                "error_code": e,
                "message": f"请求超时({timeout}s)",
                "retryable": True
            }
        except RateLimitError as e:
            error_info = {
                # "error_code": "RATE_LIMIT",
                "error_code": e,
                "message": "请求频率过高，触发速率限制",
                "retryable": True
            }
        except APIConnectionError as e:
            error_info = {
                # "error_code": "API_CONNECTION_ERROR",
                "error_code": e,
                "message": "API连接错误",
                "retryable": True
            }
        except APIError as e:
            error_info = {
                # "error_code": "API_ERROR",
                "error_code": e,
                "message": "API服务暂时不可用",
                "retryable": True
            } 
        except Exception as e:
            error_info = {
                # "error_code": "UNKNOWN_ERROR",
                "error_code": e,
                "message": f"未知错误: {str(e)}",
                "retryable": False
            }
            return (False, error_info)
        # 处理可重试的错误
        if error_info.get("retryable", False):
            if retry_count < max_retry:
                # 指数退避 + 随机抖动
                delay = base_delay * (2 ** retry_count) + 0.1 * retry_count
                time.sleep(delay)
                retry_count += 1
                continue

        break  # 不可重试错误或达到最大重试次数
    
    return (False, error_info)

def get_prompt_content(file_path):
    # 从文件中读取prompt
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content