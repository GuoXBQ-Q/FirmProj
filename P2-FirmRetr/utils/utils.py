import os
import json

def get_json_content_from_file(json_path):
    # 从文件中读取json内容
    with open(json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content

def save_llm_phase_time(save_path, stage, times):
    # 记录每次llm调用的耗时
    if not os.path.exists(save_path):
        data = {}
    else:
        try:
            with open(save_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            # 文件存在但内容不是有效的 JSON，初始化为一个空字典
            print(f"Error reading JSON from {save_path}. Initializing as an empty dictionary.")
    
    data[f'{stage}_times'] = times

    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
def save_llm_usage(save_path, stage, tokens):
    # 记录每次llm调用使用的token量
    if not os.path.exists(save_path):
        data = {}
    else:
        try:
            with open(save_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            # 文件存在但内容不是有效的 JSON，初始化为一个空字典
            print(f"Error reading JSON from {save_path}. Initializing as an empty dictionary.")
    
    data[f'{stage}_tokens'] = tokens

    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

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


def init_stats_file(file_path):
    """
    检查指定路径的 JSON 文件是否存在。
    无论文件是否存在，都将内容设置为逻辑空（即包含一个空字典 {}）。

    :param file_path: JSON 文件的路径
    """
    try:
        # 打开文件，如果不存在则自动创建，并将内容设置为空字典
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump({}, file, ensure_ascii=False, indent=4)
        
        if os.path.exists(file_path):
            print(f"文件 {file_path} 已初始化为空字典。")
        else:
            print(f"文件 {file_path} 创建成功，并初始化为空字典。")
    except Exception as e:
        print(f"处理文件时出错: {e}")

def save_errors(errors: list, file_path):
    # 所有任务完成后，打印汇总的错误信息
    with open(file_path,'w') as file:
        file.write(f"Summary of Errors(total {len(errors)}):\n")
        file.write("\n".join(errors))
        file.write("\n")
