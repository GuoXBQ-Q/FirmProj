import os
import re
import time
import random
import json
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.myllm_sdk import create_chat_completion, model_redirect
from utils.utils import save_llm_phase_time, save_llm_usage, save_errors
from utils.deepseek_tokenizer import count_tokens
from utils.logger import Logger, ensure_log_directory, get_latest_log_number
from config import source_data_path, result_root_path, process_dataset

# script_name = os.path.basename(__file__)
# log_dir = f"logs/{os.path.splitext(script_name)[0]}"
# print(log_dir)
# exit(1)
log_dir = "logs/llm_preprocess"
ensure_log_directory(log_dir)
last_log_num = get_latest_log_number(log_dir, "llm_preprocess", process_dataset)

logger = Logger(name="preprocess_logger", level="DEBUG",
                log_file=f"{log_dir}/llm_preprocess_logger_{process_dataset}_{last_log_num+1}.log",
                log_file_level="DEBUG")

def get_json_content_from_file(json_path) -> dict:
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

def count_json_pairs(data):
    """
    统计JSON文件中键值对数量。
    """
    return len(data) if isinstance(data, dict) else 0

# 可能的固件后缀名
possible_extensions = [
    ".bin", ".BIN" , ".zip", ".pat", ".eap", ".trx", ".ZIP", ".chk", ".ptz", ".all", ".rar", ".hex",".RAR",".dmg",
    ".img", ".imx", ".sbio", ".gz", ".dwn", ".bdl", ".npk", ".image", ".web", ".hqx",".upg", ".stk", ".flash",
    ".trx", ".tar", ".upg", ".UPDATE",'.update', ".iso", ".ebl", ".ggl", ".pak", ".dcf", ".uimage", ".cwe",
    ".fdt", ".rfu", ".dwn", ".smi", ".rom", ".nvu", ".bix", ".dat", ".spk", ".vcc", ".cab", ".kwb", ".hd", 
    ".imag", ".UPG", ".sit", ".tif", ".lzb",".tar.gz", ".fw", ".dfu", ".eeprom", ".ota", ".e2", ".xx"]

black_extension = [
    "ttf","jpg","apk","png","js","gif","svg","pdf","7","bwf","BWF","txt","ico_0","ico_0_1","zip_0",
    "db","asc","js_0","cyacd","exe","tgz","mp4","dls","0","bin_0_1","apk_0","msi"
]

FILTER_SET = [
    # firmware
    "downloadfirmware", "download_firmware", "firmware_download", "FirmwareUpdate", "Firmware_Update", "checkFirmwareStatus", "getFirmwareInfo",
    "pullFirmware", "pushFirmware", "firmwareVersion", "firmware_info", "firmware_version", "validateFirmware", "latestfirmware", "checkFirmware", 
    "firmware_url", "firmware",  "fwurl", 
    # update
    "update_info", "upgrade_info", "fw_update", "updateDevice", "checkUpdate", "deviceupdate", 
    "upgrade", "update",
    # download
    "download_upgrade", "DOWNLOAD_UPDATE", "downloadUrl", "DownVersion", "downLoadBinFile", "download_version",
    # OTA
    "otaCheck", "otaDownload", "deviceDetails", "deviceOTA", "checklsOTAUpdateNeeded",
    # version
    "currentVersion", "latestVersion", "getLatest", "checklatest", "getVersonJson", "checkVersion", "newVersion", "checknew", "latestfw",  
    "serverInfo", "newfwinfo", "getfwversion", "swversion", 
    # other
    # "getIP",  "serverzone",  "Region",
]

BLACK_SET = [

]

# 自定义预处理函数
def preprocess(text):
    return text.lower()

def check_partial_keywords_in_text(text:str, keyword_set:list[str]):
    # 将文本转换为小写并分割单词（支持常见分隔符）
    words = re.findall(r'\b[\w$-]+\b', text.lower())
    if words[0] == "possible":
        words = words[2:]
    words = [word for word in words if word != "url"]
    logger.debug(words)
    # 存储匹配结果
    matched_keywords = []
    for keyword in keyword_set:

        results = process.extract(keyword, words, processor=preprocess, scorer=fuzz.WRatio, score_cutoff=85)
        # logger.debug(results)
        if len(results) > 0:
            matched_keywords.append(keyword)
    logger.debug(f"matched_keywords:{matched_keywords}")
    if len(matched_keywords) > 0:
        return True
    else:
        return False    
    # # 遍历分割后的单词
    # for word in words:
    #     # 遍历词典中的关键词，检查是否是当前单词的子字符串
    #     for keyword in keyword_set:
    #         if keyword.lower() in word:  # 部分匹配
    #             return True
    return False


def extract_urls_and_suffixes(data):
    results = []
    # 正则表达式匹配 URL（支持 http 和 https）
    url_pattern = re.compile(r'(https?://[^\s\'",\]\[]+)')
    
    
    # 提取所有 URL
    urls = url_pattern.findall(data)
    if not urls:
        results.append([])  # 如果没有找到 URL，添加空列表
        return []
    
    # 处理每个 URL
    for url in urls:
        # 去掉查询参数部分
        clean_url = url.split('?')[0]
        
        # 提取最后一个文件的后缀名
        suffix_match = re.search(r'\.([a-zA-Z0-9]+)$', clean_url)
        suffix = suffix_match.group(1) if suffix_match else None
        
        # 添加 (url, 后缀) 元组
        results.append((clean_url, suffix))

    return results


def pre_filter(app_name, dataset):
    logger.info(f"==================== llm preprocess | start process {app_name} ===================")
    phase0_start_time = time.time()
    appdir_path = os.path.join(source_data_path,dataset,app_name)
    need_process_json = []
    for file in os.listdir(appdir_path):
        if file.endswith(".json"):
            need_process_json.append(os.path.join(appdir_path, file))
    logger.debug(need_process_json)
    # 可能会存在apk目录为空
    if not need_process_json:
        logger.info(f"apk dir is empty, skip!")
        # phase1_end_time = time.time()
        # save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "phase1", phase1_end_time - phase1_start_time)
        return f"Processed {app_name}"
    result_path = os.path.join(result_root_path,dataset,app_name,'llm_preprocess')
    if not os.path.exists(result_path):
        # 创建文件夹
        os.makedirs(result_path)
    
    for index, json_file in enumerate(need_process_json,start=1):
        # json file name
        json_file_name = os.path.basename(json_file)
        logger.info(f" {index}/{len(need_process_json)} | processing file_path: {json_file}")
        content = get_json_content_from_file(json_file)
        
        json_pairs_num = count_json_pairs(content)
        if json_pairs_num == 0 :
            logger.info(f"Json file {json_file_name} is empty, skip")
            continue
        else:
            # main filter process
            filtered_content = {}
            for key, value in content.items():
                tokens = count_tokens(value)
                # logger.info(tokens)
                # 只对tokens数量小于1k的进行处理
                if tokens < 1000:
                    
                    # filter-1 对于[Possible Url] 类，提取url，并判断后缀
                    if value.startswith("[Possible Url]"):
                        result = extract_urls_and_suffixes(value)
                        if result:
                            for url, suffix in result:
                                if suffix in black_extension:
                                    logger.info(f"key:{key} suffix:{suffix} is not firmware.")
                                    continue
                        
                    # # filter-1 词表模糊匹配-综合匹配，阈值90以上
                    flag = check_partial_keywords_in_text(value,FILTER_SET)
                    if flag:
                        filtered_content[key] = value
                        logger.info(f"{key} is valid. similarity > 90%")
                        # 满足条件，直接开始下一轮循环
                        continue                  
            logger.info(f"app {app_name} have {len(filtered_content)} valid requets.")
            with open(f"{result_path}/{app_name}_filtered.json", "w") as f:
                json.dump(filtered_content, f, indent=4)
    phase0_end_time = time.time()
    save_llm_phase_time(os.path.join(result_root_path, dataset, app_name, "firmproj_stats.json"), "llm_preprocess", phase0_end_time - phase0_start_time)
    return f"Processed {app_name}"

def pre_filter_error(app:str, process_dataset:str):
    if app.startswith("cn."):  # 假设这是一个会引发错误的任务
        raise ValueError(f"Error processing {app}")
    return f"Processed {app}"


if __name__ == "__main__":
    process_dataset = "LOCAL_APK"
    applist = os.listdir(os.path.join(source_data_path, process_dataset))
    # applist = ["com.roku.rokuhome.apk"]
    # 使用线程池执行任务
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(pre_filter, app, process_dataset): app  # 将 future 和对应的 app 关联起来
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