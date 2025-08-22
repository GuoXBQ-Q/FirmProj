import os
from dotenv import load_dotenv, find_dotenv

def get_oneapi_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv('ONE_API_KEY')

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv('OPENAI_API_KEY')

def get_tencent_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv('TENCENT_API_KEY')

def get_aliyun_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv('ALIYUN_API_KEY')

def get_third_party_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv('THIRD_PARTY_API_KEY')

def get_bytedance_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv('BYTEDANCE_API_KEY')

def get_deepseek_api_key():
    _ = load_dotenv(find_dotenv())
    return os.getenv('DEEPSEEK_API_KEY')

def get_api_key(api_name):
    if api_name == 'openai':
        return get_openai_api_key()
    elif api_name == "deepseek":
        return get_deepseek_api_key()
    elif api_name == 'tencent':
        return get_tencent_api_key()
    elif api_name == 'aliyun':
        return get_aliyun_api_key()
    elif api_name == 'bytedance':
        return get_bytedance_api_key()
    elif api_name == 'third_party':
        return get_third_party_api_key()
    elif api_name == 'one_api':
        return get_oneapi_api_key()
    else:
        return None
    
if __name__ == '__main__':
    print(f"OpenAI API Key: {get_openai_api_key()}")
    print(f"Tencent API Key: {get_tencent_api_key()}")
    print(f"Aliyun API Key: {get_aliyun_api_key()}")
    print(f"Third Party API Key: {get_third_party_api_key()}")