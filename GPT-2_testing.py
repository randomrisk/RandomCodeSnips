from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2Model

# 加载文本生成管道
generator = pipeline('text-generation', model='gpt2')

# 设置随机种子以确保结果可重复
set_seed(42)

# 生成文本
generated_texts = generator("你好，我是一个语言模型，", max_length=30, num_return_sequences=5)

# 输出生成的文本
for i, text in enumerate(generated_texts):
    print(f"生成的文本 {i+1}: {text['generated_text']}")

# 加载分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# 编码输入文本
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

# 获取模型输出
output = model(**encoded_input)

