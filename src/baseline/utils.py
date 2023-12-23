import nltk 
import json
import heapq

with open('../../dataset/final/train.json') as f:
    dictionary = json.load(f)


def get_n_smallest_distance_example(query: str, shot: int) -> list():
    min_heap = list()
    for word in dictionary:
        distance = nltk.edit_distance(word['input'], query, substitution_cost=2, transpositions=True)
        heapq.heappush(min_heap, (distance, word['input'], word['output']))

    return heapq.nsmallest(shot, min_heap)

def get_n_smallest_distance_example_2_part(query: str, shot: int) -> list():
    min_heap_1 = list()
    min_heap_2 = list()
    for word in dictionary:
        distance = nltk.edit_distance(word['input'], query[:len(query)//2], substitution_cost=2, transpositions=True)
        heapq.heappush(min_heap_1, (distance, word['input'], word['output']))
    
    for word in dictionary:
        distance = nltk.edit_distance(word['input'], query[len(query)//2:], substitution_cost=2, transpositions=True)
        heapq.heappush(min_heap_2, (distance, word['input'], word['output']))

    return heapq.nsmallest(shot // 2 , min_heap_1) + heapq.nsmallest(shot - shot // 2, min_heap_2)


def get_examples(query: str, shot: int, part: int = 1) -> list:
    """return silimar terminology examples prompt"""
    if part == 1:
        examples = get_n_smallest_distance_example(query, shot)
    elif part == 2:
        examples = get_n_smallest_distance_example_2_part(query, shot)
    prompt = ""
    for example in examples:
        prompt += f"{example[1]} -> {example[2]}\n"
    return prompt[:-2]



def get_prompt_sentence(source: str) -> str:
    return f"請幫我將這一句中文翻譯成越南語: {source}"

def get_prompt_3_shot_sentence(source: str) -> str:
    return f"""請幫我將這一句中文翻譯成越南語:
在北京的大街上，人群熙熙攘攘。 -> Trên con đường ở bắc kinh, đám đông đông đúc. 
在上海的公园里，人们散步欣赏花草。 -> Trong công viên ở thướng hải, mọi người đi dạo ngắm hoa cỏ.
靜岡的夜市里，各种美食琳琅满目。 -> Trong chợ đêm ở tĩnh cương, có đủ loại đồ ăn ngon.
請幫我將這一句中文翻譯成越南語:
{source} -> """

def get_prompt_6_shot_sentence(source: str) -> str:
    return f"""請幫我將這一句中文翻譯成越南語:
在北京的大街上，人群熙熙攘攘。 -> Trên con đường ở bắc kinh, đám đông đông đúc. 
在上海的公园里，人们散步欣赏花草。 -> Trong công viên ở thướng hải, mọi người đi dạo ngắm hoa cỏ.
靜岡的夜市里，各种美食琳琅满目。 -> Trong chợ đêm ở tĩnh cương, có đủ loại đồ ăn ngon.
在南京的古城区，古老的建筑沐浴在阳光下。 -> Ở khu phố cổ ở nam kinh, các công trình cổ được tắm nắng.
在杭州的运河畔，古老的石桥依然屹立。 -> Bên bờ kênh đào ở hàng châu， cầu đá cổ vẫn đứng vững.
在重慶的古城墙上，青石板铺就的路延伸而去。 -> Trên tường thành cổ ở trọng khanh, con đường được lát bằng đá xanh kéo dài.
請幫我將這一句中文翻譯成越南語:
{source} -> """

def get_prompt_terminology(source: str) -> str:
    return f"請幫我將中文翻譯成越南語: {source}"

def get_prompt_3_shot_terminology(source: str) -> str:
    return f"""請幫我將中文翻譯成越南語:
德惠縣 -> Huyện Đức Huệ 
𡊤槤市鎮 -> Thị trấn Giồng Riềng
平興和坊 -> Phường Bình Hưng Hòa
請幫我將中文翻譯成越南語:
{source} -> """

def get_prompt_6_shot_terminology(source: str) -> str:
    return f"""請幫我將中文翻譯成越南語:
德惠縣 -> Huyện Đức Huệ 
𡊤槤市鎮 -> Thị trấn Giồng Riềng
平興和坊 -> Phường Bình Hưng Hòa
忠文坊" -> Phường Trung Văn
柴峰" -> Sài Phong
和秀二社" -> Xã Hòa Tú 2
請幫我將中文翻譯成越南語:
{source} -> """

# 只用 請幫我將中文名詞翻譯成越南語 會受到影響，有些時候會翻譯到example的部分
def get_prompt_3_shot_dist_terminology(source: str) -> str:
    return f"""請幫我將中文名詞翻譯成越南語，以下一共有三個例子供你參考:
{get_examples(source, 3)}
請幫我將中文翻譯成越南語:
{source} -> """

def get_prompt_6_shot_dist_terminology(source: str) -> str:
    return f"""請幫我將中文翻譯成越南語，以下一共有六個例子供你參考:
{get_examples(source, 6)}
請幫我將中文翻譯成越南語:
{source} -> """

def get_prompt_3_shot_dist_terminology_2_part(source: str) -> str:
    return f"""請幫我將中文名詞翻譯成越南語，以下一共有三個例子供你參考:
{get_examples(source, 3, 2)}
請幫我將中文翻譯成越南語:
{source} -> """

def get_prompt_6_shot_dist_terminology_2_part(source: str) -> str:
    return f"""請幫我將中文翻譯成越南語，以下一共有六個例子供你參考:
{get_examples(source, 6, 2)}
請幫我將中文翻譯成越南語:
{source} -> """

