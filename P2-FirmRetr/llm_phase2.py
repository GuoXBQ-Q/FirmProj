import os
import re
import time
import json
import random
import time
from collections import Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.myllm_sdk import one_chat, one_completion, create_chat_completion
from utils.myllm_sdk import get_prompt_content, model_redirect
from utils.logger import Logger, ensure_log_directory, get_latest_log_number
from utils.utils import save_llm_phase_time, save_llm_usage, save_errors
from config import result_root_path, process_dataset

log_dir = "logs/llm_phase2"
ensure_log_directory(log_dir)
last_log_num = get_latest_log_number(log_dir, "llm_phase2", process_dataset)
logger = Logger(name="phase2_logger", level="DEBUG",
                log_file=f"{log_dir}/llm_phase2_logger_{process_dataset}_{last_log_num+1}.log",
                log_file_level="DEBUG")

MODEL = model_redirect("deepseek-v3")
def get_json_content_from_file(json_path):
    # 从文件中读取json内容
    with open(json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content

def save2json(content, json_path):
    # 将内容保存为json文件
    if type(content) == str:
        content = json.loads(content)
    elif type(content) == dict:
        pass
    else:
        raise ValueError("content must be a string or dict")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    logger.debug(f"save to {json_path}")

def entropy(probabilities):
    """计算预测分布的熵"""
    probabilities = np.array(probabilities)
    probabilities = probabilities[probabilities > 0]  # 避免log(0)
    return -np.sum(probabilities * np.log(probabilities))

def dynamic_multi_round_voting(content, app_name, logger, initial_rounds=5, max_rounds=10, consistency_threshold=0.8, entropy_threshold=0.5,):
    """
    动态多轮投票机制

    param: 
    - content 文本内容 
    - logger  日志对象
    
    return:
    - final_result
    - total_time
    - total_token
    """

    predictions = []
    prompt = get_prompt_content("prompt/classify_url_prompt.txt")
    total_rounds = 0
    # 多轮投票总消耗的token量
    total_tokens = 0 
    total_input_tokens = 0
    total_output_tokens = 0

    total_llm_errors = 0

    messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
    ]
    voting_start_time = time.time()
    while total_rounds < max_rounds:

        success,completion = create_chat_completion(messages=messages,model=MODEL,temperature=1)
        
        if not success:
            logger.error(f"error code: {completion['error_code']}, message: {completion['message']}")
            logger.error(f"faild app: {app_name} ,faild content: {content}")
            # 如果一直LLM访问失败，可能会出现死循环，主要是网络原因
            # 需要加一个次数判断，如果一直失败多少次，直接返回
            total_llm_errors += 1
            if total_llm_errors >= 10:
                return None
            
            continue

        label = llm_response_content = completion.choices[0].message.content
        usage = completion.usage
        total_tokens += usage.total_tokens
        total_input_tokens += usage.prompt_tokens
        total_output_tokens += usage.completion_tokens
        predictions.append(label)
        logger.info(f"predictions: {predictions}")
        logger.info(f"used token: {usage.total_tokens} (input_tokens: {usage.prompt_tokens},output_tokens: {usage.completion_tokens})")
        total_rounds += 1
        time.sleep(1) # 避免速率限制

        if total_rounds >= initial_rounds:
            # 评估一致性与熵值
            counter = Counter(predictions)
            total_votes = sum(counter.values())
            probabilities = [count / total_votes for count in counter.values()]
            prediction_entropy = entropy(probabilities)
            most_common_label, count = counter.most_common(1)[0]
            consistency_score = count / total_votes

            logger.info(f"轮次: {total_rounds}, 当前预测分布: {counter}, 一致性: {consistency_score:.2f}, 熵: {prediction_entropy:.4f}")

            # 判断是否满足提前终止条件
            if consistency_score >= consistency_threshold and prediction_entropy <= entropy_threshold:
                logger.debug(f"符合终止条件,终止投票！")
                break
        
        else:
            logger.info(f"轮次: {total_rounds}, 当前预测: {label} (初始轮次阶段，继续收集...)")
    
    voting_end_time = time.time()
    # 返回值: 最有可能的标签、一致性分数、熵值、总的预测分布、总token数量、投票过程总用时.
    result = {
        "most_common_label":most_common_label,
        "consistency_core": consistency_score,
        "prediction_entry": prediction_entropy,
        "counter": counter,
        "total_usage": {
            "total_tokens":total_tokens,
            "total_input_tokens":total_input_tokens,
            "total_output_tokens":total_output_tokens
            },
        "voting_time": voting_end_time - voting_start_time
    }
    return result

def classify_url(app_name, dataset):
    # 完成LLM第二阶段子任务 - 对 URL 进行分类
    # 0: 完整的网络请求
    # 1,2,3: 不完整的网络请求
    logger.info(f"========== llm phase2 | start process {app_name} ==========")
    phase2_start_time = time.time()
    # path_prefix = "/data/guoxb/firmproj/result"
    dir_path = os.path.join(result_root_path,dataset,app_name,'llm_phase1')
    
    # 设置输出目录
    result_path = os.path.join(result_root_path,dataset,app_name,'llm_phase2')
    if not os.path.exists(result_path):
        # 创建文件夹
        os.makedirs(result_path)

    # 获取需要处理的json文件
    need_process_json = []
    for file in os.listdir(dir_path):
        if file.endswith(".json"):
            need_process_json.append(os.path.join(dir_path, file))
    logger.debug(need_process_json)
    
    # 如果需要处理的json文件为空
    if not need_process_json:
        logger.info(f"No json file found, skip")
        # phase2_end_time = time.time()
        # save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "phase2", phase2_end_time - phase2_start_time)
        return

    prompt = get_prompt_content("prompt/classify_url_prompt.txt")

    # 将所有的json文件内容都读取出来，避免结果出现多个文件的情况
    file_content = {}
    for index, file_path in enumerate(need_process_json,start=1):
        # json_file_name = os.path.basename(file_path)
        # logger.info(f" {index}/{len(need_process_json)} | processing file_path: {file_path}")
        file_content.update(get_json_content_from_file(file_path))
        # logger.debug(file_content)
        # logger.debug(type(file_content))
        
        # # 肯定用不到，因为上边已经过滤了，但是还是加上保险一下
        # if not file_content:
        #     logger.info(f"Json file {json_file_name} is empty, skip")
        #     continue
        
    result_0 = {}
    result_1 = {}
    result_2 = {}
    result_3 = {}

    # 该APP使用的总token量
    total_usage = 0
    total_time = 0
    total_items = len(file_content)
    for index, (key, value) in enumerate(file_content.items(),start=1):
        logger.debug(f" {index}/{total_items} | Processing key: {key} value: {value}")
        if value == "0":
            # 如果一个json文件内全是0,那么会得到一个逻辑空的json文件，需要在下一步处理时提前检查注意
            continue
        
        content = json.dumps(value)

        # 多轮投票, 加上了异常处理，如果返回的是None的话，表示在LLM访问时出现了错误，直接跳过该key-value队，并输出日志记录error情况。
        voting_result = dynamic_multi_round_voting(content,app_name,logger)
        if not voting_result:
            logger.error(f"faild app: {app_name}, faild key: {key}")
            continue
        llm_response_content = voting_result["most_common_label"]
        usage = voting_result["total_usage"]
        voting_time = voting_result["voting_time"]

        # messages = [
        #     {"role": "system", "content": prompt},
        #     {"role": "user", "content": content}
        # ]
        # llm_chat_start_time = time.time()
        # success, completion = create_chat_completion(messages=messages,model="deepseek-v3",temperature=0.5)
        # llm_chat_end_time = time.time()
        # if not success:
        #     logger.error(f"error code: {completion['error_code']}, message: {completion['message']}")
        #     logger.error(f"faild app: {app_name} ,faild file: {file_path}")
        #     continue
        # llm_response_content = completion.choices[0].message.content
        # usage = completion.usage
        # logger.debug(f"response_content: {llm_response_content}")
        # logger.debug(f"finish reason: {completion.choices[0].finish_reason}")
        logger.info(f"success app: {app_name} ,success file: {file_path}")
        logger.info(f"used token: {usage['total_tokens']} (input_tokens: {usage['total_input_tokens']},output_tokens: {usage['total_output_tokens']})")
        logger.info(f"used time: {voting_time}s")
        total_usage += usage['total_tokens']
        total_time += voting_time        
        if llm_response_content == "1":
            result_1[key] = value
            logger.info(f'key:{key} 网络请求不完整，类别：1')
            continue
        elif llm_response_content == "2":
            result_2[key] = value
            logger.info(f'key:{key} 网络请求不完整，类别：2')
            continue
        elif llm_response_content == "3":
            result_3[key] = value
            logger.info(f'key:{key} 网络请求不完整，类别：3')
            continue
        elif llm_response_content == "0":
            result_0[key] = value
            logger.info(f"key:{key} 网络请求完整！类别：0")
            continue
        else:
            logger.warning(f"LLM 输出不规范!!!!!!!")
    
    save2json(result_0, os.path.join(result_path, f"complete_0_{app_name}.json"))
    save2json(result_1, os.path.join(result_path, f"incomplete_1_{app_name}.json"))
    save2json(result_2, os.path.join(result_path, f"incomplete_2_{app_name}.json"))
    save2json(result_3, os.path.join(result_path, f"incomplete_3_{app_name}.json"))

    save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase2_chat", total_time)
    save_llm_usage(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase2_usage", total_usage)

    phase2_end_time = time.time()
    save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "phase2", phase2_end_time - phase2_start_time)

    time.sleep(random.randint(0, 10))




if __name__ == "__main__":
    # applist = os.listdir("/data/guoxb/firmproj/result/IoT-VER-Androzoo")
    applist = os.listdir(os.path.join(result_root_path, process_dataset))
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(classify_url, app, process_dataset):app 
            for app in applist
            }
        
        completed = 0
        total = len(applist)
        errors = []

        for future in as_completed(futures):
            app = futures[future]  # 获取当前任务对应的 app
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                # 捕获异常并记录详细信息
                error_message = f"Error processing {app}: {str(e)}"
                errors.append(error_message)
                print(error_message)
                result = "error"
            print(f"Progress: {completed}/{total} ")

    # 所有任务完成后，打印汇总的错误信息
        if errors:
            error_log_path = f"logs/llm_preprocess/error_{process_dataset}.log"
            save_errors(errors, error_log_path)
    