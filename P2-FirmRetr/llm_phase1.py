import os
import re
import time
import random
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.myllm_sdk import create_chat_completion, model_redirect
from utils.utils import save_llm_phase_time, save_llm_usage, save_errors
from utils.logger import Logger, ensure_log_directory, get_latest_log_number
from config import source_data_path, result_root_path, process_dataset

log_dir = "logs/llm_phase1"
ensure_log_directory(log_dir)
last_log_num = get_latest_log_number(log_dir, "llm_phase1", process_dataset)
logger = Logger(name="phase1_logger", level="DEBUG",
                log_file=f"{log_dir}/llm_phase1_logger_{process_dataset}_{last_log_num+1}.log",
                log_file_level="DEBUG")

MODEL = model_redirect("deepseek-v3")

def get_prompt_content(file_path):
    # 从文件中读取prompt
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

# def get_json_content_from_file(json_path,return_type='json'):
#     # 从文件中读取json内容
#     with open(json_path, 'r', encoding='utf-8') as f:
#         content = json.load(f)

#     # #  检查是否是 {} 空json
#     # if not content:
#     #     # 逻辑空的json，直接reutrn None
#     #     return None
    
#     # 根据需要返回的类型返回不同的内容
#     if return_type == 'string':
#         return json.dumps(content)
#     else:
#         return content

def get_json_content_from_file(json_path):
    # 从文件中读取json内容
    with open(json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content
def count_json_pairs(data):
    """
    统计JSON文件中键值对数量。
    """
    return len(data) if isinstance(data, dict) else 0

def is_json_format(s, logger):
    # 判断llm返回的content是否是json格式，预防出现markdown格式
    try:
        json.loads(s)
    except ValueError as e:
        return False
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解码错误: {e}")
        return False
    return True

def get_json_content_from_llm_response(response):
    # 从LLM响应中提取JSON内容
    
    pattern = "```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern,response)
    if match:
        json_content = match.group(1).strip()
        return json_content
    else:
        # 提取不到，则直接返回空字典
        return "{}"

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



# 三个阶段统一传参为 app package name
def format_url(app_name,dataset):
    # 该函数主要对通过静态分析获取的json格式中的url请求信息进行提取，并返回一个格式化的数据
    # path_prefix = "/data/guoxb/firmproj"
    logger.info(f"========== llm phase1 | start process {app_name} ==========")
    phase1_start_time = time.time()
    targetdir_path = os.path.join(result_root_path,dataset,app_name,"llm_preprocess")

    need_process_json = []
    for file in os.listdir(targetdir_path):
        if file.endswith(".json"):
            need_process_json.append(os.path.join(targetdir_path, file))
    logger.debug(need_process_json)

    # 可能会存在apk目录为空
    if not need_process_json:
        logger.info(f"apk dir is empty, skip!")
        # phase1_end_time = time.time()
        # save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "phase1", phase1_end_time - phase1_start_time)
        return

    result_path = os.path.join(result_root_path,dataset,app_name,'llm_phase1')
    if not os.path.exists(result_path):
        # 创建文件夹
        os.makedirs(result_path)

    prompt = get_prompt_content("prompt/extract_urlinfo_prompt.txt")

    group_threshold = 35

    for index, json_file in enumerate(need_process_json,start=1):
        # json file name
        json_file_name = os.path.basename(json_file)
        logger.info(f" {index}/{len(need_process_json)} | processing file_path: {json_file}")
        content = get_json_content_from_file(json_file)
        
        # # 如果json文件为{}空
        # if not content:
        #     logger.info(f"Json file {json_file_name} is empty, skip")
        #     continue
        
        json_pairs_num = count_json_pairs(content)
        # 对于小于50项的json文件，直接使用LLM进行解析
        if json_pairs_num == 0 :
            logger.info(f"Json file {json_file_name} is empty, skip")
            continue
        elif json_pairs_num <= group_threshold :
            message = [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': json.dumps(content)}
            ]

            llm_chat_start_time = time.time()
            success, completion = create_chat_completion(messages=message,model=MODEL,temperature=0.7)
            llm_chat_end_time = time.time()
            if not success:
                logger.error(f"error code: {completion['error_code']}, message: {completion['message']}")
                logger.error(f"faild app: {app_name} ,faild file: {json_file}")
                continue
            
            llm_response_content = completion.choices[0].message.content
            usage = completion.usage
            logger.debug(f"response_content: {llm_response_content}")
            logger.debug(f"finish reason: {completion.choices[0].finish_reason}")
            logger.info(f"success app: {app_name} ,success file: {json_file}")
            logger.info(f"used token: {usage.total_tokens} (input_tokens: {usage.prompt_tokens},output_tokens: {usage.completion_tokens})")
            logger.info(f"used time: {llm_chat_end_time - llm_chat_start_time}s")
            save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase1_chat", llm_chat_end_time - llm_chat_start_time)
            save_llm_usage(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase1_usage", usage.total_tokens)
            # 将内容保存到 json 文件, 作为下一阶段的输入
            if not is_json_format(llm_response_content,logger):
                llm_response_content = get_json_content_from_llm_response(llm_response_content)
                logger.debug(f"LLM_response match json format:\n{llm_response_content}")
                # 防止llm输出不完整导致格式化匹配不到json情况 
                if llm_response_content == "{}":
                    logger.error(f"llm response incomplete, faild app: {app_name}, faild_file: {json_file}")
            try:
                # 防止llm输出的json格式出错
                save2json(llm_response_content, os.path.join(result_path, f"{app_name}.json"))
            except Exception as e:
                logger.error(f"Json format error, faild app: {app_name}, faild_file: {json_file}")
        
        # 剩下的都是 json 项 >40 的,进行拆分处理，提高LLM的准确性和分析时间
        else:
            # 按照 group_size 分组
            group_size = group_threshold
            all_items = list(content.items())
            groups = [{str(k): v for k, v in all_items[i:i + group_size]} 
                        for i in range(0, len(all_items), group_size)]
            total_groups = len(groups)
            total_llm_time = 0
            total_tokens = 0
            for group_index, group in enumerate(groups):
                # 构造消息
                logger.info(f"processing group {group_index+1}/{total_groups} ")
                content = json.dumps(group)
                message = [
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': content}
                ]
                llm_chat_start_time = time.time()
                success, completion = create_chat_completion(messages=message, model=MODEL,temperature=0.7)
                llm_chat_end_time = time.time()
                if not success:
                    logger.error(f"error code: {completion['error_code']}, message: {completion['message']}")
                    logger.error(f"faild app: {app_name} ,faild file: {json_file}")
                    continue
                llm_response_content = completion.choices[0].message.content
                usage = completion.usage
                logger.debug(f"response_content: {llm_response_content}")
                logger.debug(f"finish reason: {completion.choices[0].finish_reason}")
                logger.info(f"success app: {app_name} ,success file: {json_file}")
                logger.info(f"used token: {usage.total_tokens} (input_tokens: {usage.prompt_tokens},output_tokens: {usage.completion_tokens})")
                logger.info(f"used time: {llm_chat_end_time - llm_chat_start_time}s")
                total_llm_time += llm_chat_end_time - llm_chat_start_time
                total_tokens += usage.total_tokens
                # 将内容保存到 json 文件, 作为下一阶段的输入
                if not is_json_format(llm_response_content,logger):
                    llm_response_content = get_json_content_from_llm_response(llm_response_content)
                    logger.debug(f"LLM_response match json format:\n{llm_response_content}")
                    # 防止llm输出不完整导致格式化匹配不到json情况
                    if llm_response_content == "{}":
                        logger.error(f"llm response incomplete, faild app: {app_name}, faild_file: {json_file}")
                try:
                    # 防止llm输出的json格式出错
                    save2json(llm_response_content, os.path.join(result_path, f"{app_name}_{group_index}.json"))
                except Exception as e:
                    logger.error(f"{e}")
                    logger.error(f"Json format error, faild app: {app_name}, faild_file: {json_file}")
            # 等所有分组都处理完，再来将总的llm time和 tokens保存到文件
            save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase1_chat", total_llm_time)
            save_llm_usage(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_phase1_usage", total_tokens)
        
        phase1_end_time = time.time()
        save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "phase1", phase1_end_time - phase1_start_time)

        time.sleep(random.randint(0, 10))

if __name__ == "__main__":
    applist = os.listdir(os.path.join(source_data_path,process_dataset))
    
    # llm incomplete 
    # applist = [
    #            "com.altec.shsm.apk",
    #            "com.doorguard.smartlock.apk",
    #            ]
    # input_length exceeds the maximum length 65536
    # applist = ["comb.blackvuec.apk", "comb.blackvuec.apk", "com.blinkhd.apk",
    #            "com.big8bits.fetchcam.apk","com.tplink.skylight.apk",""]
    # 401 
    # applist = ["com.ezio.multiwii.apk","com.logi.brownie.apk"]
    # 提交任务到线程池
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(format_url, app, process_dataset):app 
            for app in applist
        }
        
        completed = 0
        total = len(applist)
        errors = [] # 记录所有错误信息

        for future in as_completed(futures):
            app = futures[futures] # 获取当前任务对应的app
            completed += 1
            
            try:
                result = future.result() # 获取任务结果, 可能会抛出异常
            except Exception as e:
                # 捕获异常并记录详细信息
                error_message = f"Error processing {app}: {str(e)}"
                errors.append(error_message)
                print(error_message)
                result = "error"
            print(f"Progress: {completed}/{total} - {result}") # 输出进度和结果
        
        # 所有任务完成后，打印汇总的错误信息
        if errors:
            error_log_path = f"logs/llm_phase1/error_{process_dataset}.log"
            save_errors(errors, error_log_path)    
    
