import os
from dotenv import load_dotenv, find_dotenv

def get_openai_base_url():
    _ =load_dotenv(find_dotenv())
    return os.getenv('OPENAI_BASE_URL')

def get_oneapi_base_url():
    _ =load_dotenv(find_dotenv())
    return os.getenv('ONE_API_BASE_URL')

def get_tencent_base_url():
    _ =load_dotenv(find_dotenv())
    return os.getenv('TENCENT_BASE_URL')

def get_aliyun_base_url():
    _ =load_dotenv(find_dotenv())
    return os.getenv('ALIYUN_BASE_URL')

def get_bytedance_base_url():
    _ =load_dotenv(find_dotenv())
    return os.getenv('BYTEDANCE_BASE_URL')

def get_third_party_base_url():
    _ = load_dotenv(find_dotenv())
    return os.getenv('THIRD_PARTY_BASE_URL')

def get_deepseek_base_url():
    _ = load_dotenv(find_dotenv())
    return os.getenv('DEEPSEEK_BASE_URL')

def get_base_url(api_type):
    if api_type == 'openai':
        return get_openai_base_url()
    elif api_type == "deepseek":
        return get_deepseek_base_url()
    elif api_type == 'tencent':
        return get_tencent_base_url()
    elif api_type == 'aliyun':
        return get_aliyun_base_url()
    elif api_type == 'bytedance':
        return get_bytedance_base_url()
    elif api_type == 'third_party':
        return get_third_party_base_url()
    elif api_type == 'one_api':
        return get_oneapi_base_url()
    else:
        return None

if __name__ == '__main__':
    print(f"OpenAI base URL: {get_openai_base_url()}")
    print(f"Tencent base URL: {get_tencent_base_url()}")
    print(f"Aliyun base URL: {get_aliyun_base_url()}")
    print(f"Bytedance base URL: {get_bytedance_base_url()}")
    print(f"Third party base URL: {get_third_party_base_url()}")