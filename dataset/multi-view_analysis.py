import json
import httpx
from tqdm import tqdm
from openai import OpenAI


api_key = "YOUR_API_KEY"
base_url = "YOUR_BASE_URL"
with open("dataset/weibo_data.json") as file:
    weibo_data = json.load(file)


client = OpenAI(
    base_url=base_url, 
    api_key=api_key,
    http_client=httpx.Client(
        base_url=base_url,
        follow_redirects=True,
    ),
)

system_prompt = '你是一位语言学专家，精通文本幽默分析。'
prompts = {
    "语义分析": "请分析以下文本中是否使用了词语多义性、双关语等修辞手法，并解释它们是否可能产生幽默效果。待分析文本:",
    "语用分析": "请根据上下文分析以下文本中是否存在隐含意义和言外之意，并判断它们是否构成了幽默。待分析文本:",
    "语法和句法分析": "请检查以下文本的句子结构，分析是否存在不寻常的语法或句法现象，并判断它们是否为幽默服务。待分析文本:",
    "文化背景分析": "请根据以下文本的文化或社会背景，分析是否有文化相关的幽默存在。待分析文本:",
    "认知矛盾分析": "请分析以下文本是否包含认知矛盾或不一致，并判断其是否用于制造幽默。待分析文本:",
    "心理分析": "请分析以下文本可能引发的心理反应，并判断它们是否能够产生幽默。待分析文本:",
}


def multi_view_analysis(text):
    results = []
    for key, prompt in prompts.items():
        completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt + text}
          ]
        )
        results.append(completion.choices[0].message.content)
    return results


finished = 1423
finished_indexs = []
finished_indexs.extend(list(range(finished)))

for i, weibo in tqdm(enumerate(weibo_data), total=len(weibo_data)):
    if i in finished_indexs:
        continue
    text = weibo["text"]
    try:
        analysis = multi_view_analysis(text)
    except Exception as e:
        print(e)
        print("Error occurred while processing weibo_id:", weibo["weibo_id"], ", index:", i)
        continue
    weibo_data[i]["analysis"] = analysis

    with open("dataset_multi_view/weibo_data_multi-view-analysis.json", "a", encoding='utf-8') as file:
        file.write(json.dumps(weibo_data[i], ensure_ascii=False, indent=4) + "\n")


