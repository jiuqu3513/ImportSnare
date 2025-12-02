from openai import OpenAI
from .Model import Model


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        base_url = config["api_key_info"]["base_url"]
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=api_keys[api_pos],base_url=base_url)

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(e)
            response = ""

        return response
    def translate(self,msg,lang='ch'):

        templates = {
                    "ch": {
                        "system": "你是一个专业的技术文档翻译专家",
                        "user": """请将以下英文内容逐字逐句翻译成简体中文：
        1. 保留所有Markdown格式、代码块和特殊符号
        2. 技术术语需保持英文原样不翻译
        3. 禁止添加解释或额外内容
        4. 输出必须与原文结构完全一致

        待翻译内容：{text}"""
                    },
                    "fr": {
                        "system": "You are a professional technical translator",
                        "user": """Translate the English text to French following these rules:
        1. Preserve all Markdown formatting, code blocks and special symbols
        2. Keep technical terms in English
        3. Never add explanations or additional content
        4. Maintain exactly the same structure as original

        English content: {text}"""
                    }
                }

        # 构建完整prompt
        full_prompt = templates[lang]["user"].format(text=msg)
        
        messages = [
            {"role": "system", "content": templates[lang]["system"]},
            {"role": "user", "content": full_prompt},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=messages,
            )
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(e)
            response = ""

        return response
