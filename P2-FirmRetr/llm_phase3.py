import os
import re
import time
import random
import json
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.myllm_sdk import create_chat_completion, dp_official_create_chat_completion
from utils.myllm_sdk import get_prompt_content,model_redirect
from utils.utils import get_json_content_from_file, save_llm_phase_time, save_llm_usage, save_errors
from multi_request import Request_multi
from utils.logger import Logger, ensure_log_directory, get_latest_log_number
from config import source_data_path, result_root_path, process_dataset

log_dir = "logs/llm_phase3"
ensure_log_directory(log_dir)
last_log_num = get_latest_log_number(log_dir, "llm_phase3", process_dataset)
logger = Logger(name="phase3_logger", level="DEBUG",
                log_file=f"{log_dir}/llm_phase3_logger_{process_dataset}_{last_log_num+1}.log",
                log_file_level="DEBUG")

MODEL = model_redirect("deepseek-v3")

visited = []

# 定义外部调用函数库 
request_multi = Request_multi(logger)
available_function = {
    "make_request_name": request_multi.make_request_multi,
}
make_request_function_description = {
        "name": "make_request_name",
        "description": "用于发送get或post网络请求的函数",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],  # 限制方法为GET或POST
                    "description": "网络请求的方法"
                },
                "urls": {
                    "type": "array",
                    "items": {  # 定义数组元素的类型
                        "type": "string"  # 数组元素的类型为字符串
                    },
                    "description": "request请求可能的url列表"
                },
                "headers": {  # 一个字符串类型的字典
                    "type": "object",
                    "additionalProperties": {
                        "type": "string"  # 可以接收任意字符串类型的属性键和值。
                    },
                    "required": [],
                    "description": "request请求附带的header,需要为字典格式,且每个header对应一个具体值,忽略dynamic动态值,只保留一个具体值,如果没有则置为空"
                },
                "parameter": {
                    "oneOf": [
                        {
                            "type": "object",
                            "additionalProperties": {
                                "type": "string"
                            },
                            "required": [],
                            "description": "从param需要为字典格式,且每个header对应一个值; 忽略dynamic动态值,只保留一个具体值,如果没有则置为空"
                        },
                        {
                            "type": "string",
                            "description": "一条字符串数据"
                        }
                    ],
                    "description": "method为GET时，从param字段获取为字典；method为POST时，从body字段获取为字典或字符串"
                },
            },
            "required": ["method", "url", "parameter"]  # 强制要求method和url
        }
    }
functions_description = [
    {
        "type":"function",
        "function": make_request_function_description
    }
]

def is_list_format(content):
    try:
        result = ast.literal_eval(content)
        return isinstance(result, list)
    
    except (ValueError, SyntaxError):
        return False

def get_list_from_llm_response(response):
    # 从LLM响应中提取python list内容
    
    pattern = "```python\s*([\s\S]*?)\s*```"
    match = re.search(pattern,response)
    if match:
        json_content = match.group(1).strip()
        return json_content
    else:
        # 提取不到，则直接返回空字典
        return "[]"

def get_function_call(message):
    # 获取函数调用的 function name 和 params
    
    pass


def download_complete_file(app_name,dataset):
    # 下载complete类型的固件
    # 对complete.json文件进行处理的时候，需要提前检查是否是逻辑空的json文件，如果是，则跳过该文件
    logger.info(f"========== llm phase3 | start process {app_name} ==========")
    phase3_start_time = time.time()
    # path_prefix = "/data/firmproj/result"
    dir_path = os.path.join(result_root_path,dataset,app_name,'llm_phase2')
    if not os.path.exists(dir_path):
        logger.info(f"{app_name} llm_phase2 not exists. skip")
        return
    result_path = os.path.join(result_root_path,dataset,app_name,'llm_phase3')
    # logger.debug(result_path)
    if not os.path.exists(result_path):
        # 创建文件夹
        os.makedirs(result_path)

    # 考虑 llm_phase2 为空的情况
    if not os.listdir(dir_path):
        logger.info(f"No json file found, skip")
        # phase3_end_time = time.time()
        # save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "phase3", phase3_end_time - phase3_start_time)
        return
    
    input_file = [file for file in os.listdir(dir_path) if file.startswith('complete')][0]
    input_file_path = os.path.join(dir_path,input_file)

    

    function_call_prompt = get_prompt_content("prompt/2_prompt_functioncall.txt")

    # 读取json内容
    json_content = get_json_content_from_file(input_file_path)
    logger.debug(f"file_content: \n{json_content}")
    if not json_content:
        logger.info(f"json_content is empty")
        return
    # 统计函数调用中所花费的时间和tokens
    total_function_call_time = 0
    total_function_call_tokens = 0

    # 统计下载过程中所花费的时间和tokens
    total_download_time = 0
    total_download_tokens = 0

    total_items = len(json_content)
    for index, (key,value) in enumerate(json_content.items(),start=1):
        logger.info(f" {index}/{total_items} | Processing key: {key} value: {value}")
        query_content = json.dumps(value)
        message = [
            {'role': 'system', 'content': function_call_prompt},
            {'role': 'user', 'content': query_content}
        ]
        logger.debug(f"message: {message}")
        llm_chat_start_time = time.time()
        success, completion = dp_official_create_chat_completion(messages=message,
                                                     model="deepseek-chat",
                                                     temperature=0.5,
                                                     tools=functions_description,
                                                     )
        llm_chat_end_time = time.time()
        if not success:
            logger.error(f"error code: {completion['error_code']}, message: {completion['message']}")
            logger.error(f"faild app: {app_name} ,faild file: {input_file_path}, error key:{key}")
            continue
        llm_response_content = completion.choices[0].message
        usage = completion.usage
        logger.debug(f"response_message: {llm_response_content}")
        logger.debug(f"finish reason: {completion.choices[0].finish_reason}")
        logger.info(f"success app: {app_name} ,success file: {input_file_path}")
        logger.info(f"used token: {usage.total_tokens} (input_tokens: {usage.prompt_tokens},output_tokens: {usage.completion_tokens})")
        logger.info(f"used time: {llm_chat_end_time - llm_chat_start_time}s")
        total_function_call_time += llm_chat_end_time - llm_chat_start_time
        total_function_call_tokens += usage.total_tokens
        

        # 获取返回的需要调用的函数名及其参数
        function_name = completion.choices[0].message.tool_calls[0].function.name
        logger.debug(f"LLM decided to call function: {function_name}")
        function_args = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
        function_args.update({"app_name":app_name,"dataset":dataset})
        logger.debug(f"LLM decided to call function arguments: {function_args}")
        function_to_call = available_function[function_name]
        function_response = function_to_call(**function_args)
        logger.debug(f"function_call_response: {function_response}")
        
        # 下面的代码直接copy自原始代码，未做逻辑功能修改。(能跑就行)
        total_response = len(function_response)
        for index, res in enumerate(function_response):
            logger.info(f" download processing : {index+1}/{total_response} function_response items") 
            download_start_time = time.time()
            results, download_usage = startDownload(res, request_multi, function_args.get('urls')[index], app_name=app_name, dataset=dataset)
            download_end_time = time.time()
            total_download_tokens += download_usage
            total_download_time += download_end_time - download_start_time
            if not results:
                logger.debug(f"startDownload function return []")
                continue
            # 下边这个感觉不是很必要
            for index, result in enumerate(results):
                try:
                    logger.info(f"deeper download processing : {index+1}/{len(results)}")
                    results_2, _ = startDownload(result, request_multi, "", app_name=app_name, dataset=dataset)
                    logger.debug(f"file download result: {results_2}")
                except UnboundLocalError as e:
                    logger.error(f"{e}, failed app: {app_name}")
                finally:
                    continue
            
    # 保存LLM chat时间和usage, 
    save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase3_chat1", total_function_call_time)
    save_llm_usage(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase3_usage1", total_function_call_tokens)
    save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase3_chat2", total_download_time)
    save_llm_usage(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase3_usage2", total_download_tokens)
    phase3_end_time = time.time()
    save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "phase3", phase3_end_time - phase3_start_time)   


def startDownload(res, request_multi, function_args="", app_name=None, dataset=process_dataset):
    global visited
    downloadlink_prompt = get_prompt_content("prompt/extract_download_link_prompt.txt")
    messages = [
        {"role": "system", "content": downloadlink_prompt},
        {"role": "user",
         "content": "Request url: " + function_args + "\nResponse: " + str(res)}
    ]

    llm_chat_start_time = time.time()
    success, completion = create_chat_completion(model=MODEL,
                                                 messages=messages,
                                                 temperature=0.7)
    llm_chat_end_time = time.time()
    if not success:
        logger.error(f"error code: {completion['error_code']}, message: {completion['message']} in startDownload")
        logger.error(f"faild app: {app_name}, faild request url: {function_args}")
        return None
    
    llm_response_content = completion.choices[0].message.content
    usage = completion.usage
    logger.debug(f"response_content: {llm_response_content}")
    logger.debug(f"finish reason: {completion.choices[0].finish_reason}")
    logger.info(f"download used token: {usage.total_tokens} (input_tokens: {usage.prompt_tokens},output_tokens: {usage.completion_tokens})")
    logger.info(f"download used time: {llm_chat_end_time - llm_chat_start_time}s")
    
    # # 保存 LLM chat时间和usage
    # save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase3_chat2", llm_chat_end_time - llm_chat_start_time)
    # save_llm_usage(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase3_usage2", usage.total_tokens)
    
    if not is_list_format(llm_response_content):
        llm_response_content = get_list_from_llm_response(llm_response_content)
        logger.debug(f"LLM_response match python list format:\n{llm_response_content}")

    # 将str转换为list 列表 
    downloadlink_list = ast.literal_eval(llm_response_content)
    downloadlink_list = list(set(downloadlink_list))
    logger.debug(f"downloadlist: {downloadlink_list}")

    for url in downloadlink_list:
        if url in visited:
            downloadlink_list.remove(url)
        visited.append(url)
    
    result = request_multi.make_request_multi("GET", downloadlink_list, download=True,app_name=app_name,dataset=dataset)

    return result, usage.total_tokens
    


def download_incomplete_file():
    # 下载incomplete类型的固件
    pass

if __name__ == "__main__":
    # app = "com.aidong.ishoes.apk"
    # applist = os.listdir(os.path.join(source_data_path, process_dataset))

    with open("./IoT-VER-applist.txt", "r") as file:
        applist = file.read().splitlines()
    # 每次跑1000个app
    index = 9000
    applist = applist[index:]
    # applist = applist[index:index+1000]
    # 注意保存日志！！！

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(download_complete_file, app, process_dataset):app
            for app in applist
            }

        completed = 0
        total = len(applist)
        errors = []  # 用于记录所有错误信息

        for future in as_completed(futures):
            app = futures[future]  # 获取当前任务对应的 app
            completed += 1
            try:
                result = future.result()  # 获取任务结果，可能会抛出异常
                
            except Exception as e:
                # 捕获异常并记录详细信息
                error_message = f"Error processing {app}: {str(e)}"
                errors.append(error_message)
                print(error_message)
                result = "error"
            print(f"Progress: {completed}/{total} - {result}")

        # 所有任务完成后，打印汇总的错误信息
        if errors:
            error_log_path = f"logs/llm_preprocess/error_{process_dataset}.log"
            save_errors(errors, error_log_path)