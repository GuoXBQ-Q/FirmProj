# pip3 install transformers
# python3 deepseek_tokenizer.py
import transformers

def count_tokens(content: str):

        chat_tokenizer_dir = "/data/guoxb/firmproj-llm/utils"

        tokenizer = transformers.AutoTokenizer.from_pretrained( 
                chat_tokenizer_dir, trust_remote_code=True
                )
        
        result = tokenizer.encode(content)
        token_length = len(result)

        return token_length


if __name__ == "__main__":

        content = "Hello, world!"
        token_length = count_tokens(content)
        print(f"The length of the token is {token_length}")