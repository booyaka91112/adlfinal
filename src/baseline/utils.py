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


def get_prompt_3_shot_sentence_taiwan(source: str) -> str:
    return f"""請幫我將這一句中文翻譯成越南語:
我在台南市工作。 -> Tôi làm việc tại thành phố Đài Nam.
台中市有許多知名的景點。 -> Thành phố Đài Trung có nhiều điểm du lịch nổi tiếng.
你知道花蓮縣有哪些著名的自然美景嗎？ -> Bạn có biết có những danh thắng tự nhiên nổi tiếng nào ở huyện Hòa Liên không?
請幫我將這一句中文翻譯成越南語:
{source} -> """


def get_prompt_3_shot_sentence_taiwan_google(source: str) -> str:
    """similar template as unseen test data, but translate result is from google translate"""
    return f"""請幫我將這一句中文翻譯成越南語:
我在台南市工作。 -> Tôi làm việc ở thành phố Đài Nam.
台中市有許多知名的景點。 -> Thành phố Đài Trung có nhiều điểm tham quan nổi tiếng.
你知道花蓮縣有哪些著名的自然美景嗎？ -> Bạn có biết huyện Hoa Liên có những vẻ đẹp thiên nhiên nổi tiếng nào không?
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


def get_prompt_6_shot_sentence_taiwan(source: str) -> str:
    return f"""請幫我將這一句中文翻譯成越南語:
我在台南市工作。 -> Tôi làm việc tại thành phố Đài Nam.
台中市有許多知名的景點。 -> Thành phố Đài Trung có nhiều điểm du lịch nổi tiếng.
你知道花蓮縣有哪些著名的自然美景嗎？ -> Bạn có biết có những danh thắng tự nhiên nổi tiếng nào ở huyện Hòa Liên không?
我计划明年去宜蘭縣旅行。 -> Tôi dự định đi du lịch tại huyện Nghệ Lan vào năm sau.
我们一家人住在新北市。 -> Cả gia đình chúng tôi sống tại thành phố Tân Bắc.
你知道雲林縣的特色是什麼嗎？ -> Bạn có biết đặc sản của huyện Vân Lâm là gì không?
請幫我將這一句中文翻譯成越南語:
{source} -> """


def get_prompt_6_shot_sentence_taiwan_google(source: str) -> str:
    """similar template as unseen test data, but translate result is from google translate"""
    return f"""請幫我將這一句中文翻譯成越南語:
我在台南市工作。 -> Tôi làm việc ở thành phố Đài Nam.
台中市有許多知名的景點。 -> Thành phố Đài Trung có nhiều điểm tham quan nổi tiếng.
你知道花蓮縣有哪些著名的自然美景嗎？ -> Bạn có biết huyện Hoa Liên có những vẻ đẹp thiên nhiên nổi tiếng nào không?
我计划明年去宜蘭縣旅行。 -> Tôi dự định đi du lịch đến huyện Yilan vào năm tới.
我们一家人住在新北市。 -> Gia đình chúng tôi sống ở thành phố Tân Đài Bắc.
你知道雲林縣的特色是什麼嗎？ -> Bạn có biết đặc điểm của huyện Vân Lâm là gì không?
請幫我將這一句中文翻譯成越南語:
{source} -> """


def get_prompt_terminology(source: str) -> str:
    return f"請幫我將中文翻譯成越南語: {source}"


def get_prompt_3_shot_sentence_term(source: str) -> str:
    return f"""請幫我將這一句中文翻譯成越南語:
在{"{"}德惠縣{"}"}的大街上，人群熙熙攘攘。 -> Trên con đường ở {"{"}Huyện Đức Huệ{"}"}, đám đông đông đúc. 
在{"{"}𡊤槤市鎮{"}"}的公园里，人们散步欣赏花草。 -> Trong công viên ở {"{"}Thị trấn Giồng Riềng{"}"}, mọi người đi dạo ngắm hoa cỏ.
{"{"}平興和坊{"}"}的夜市里，各种美食琳琅满目。 -> Trong chợ đêm ở {"{"}Phường Bình Hưng Hòa{"}"}, có đủ loại đồ ăn ngon.
請幫我將這一句中文翻譯成越南語:
{source} -> """

def get_prompt_6_shot_sentence_term(source: str) -> str:
    return f"""請幫我將這一句中文翻譯成越南語:
在{"{"}德惠縣{"}"}的大街上，人群熙熙攘攘。 -> Trên con đường ở {"{"}Huyện Đức Huệ{"}"}, đám đông đông đúc. 
在{"{"}𡊤槤市鎮{"}"}的公园里，人们散步欣赏花草。 -> Trong công viên ở {"{"}Thị trấn Giồng Riềng{"}"}, mọi người đi dạo ngắm hoa cỏ.
{"{"}平興和坊{"}"}的夜市里，各种美食琳琅满目。 -> Trong chợ đêm ở {"{"}Phường Bình Hưng Hòa{"}"}, có đủ loại đồ ăn ngon.
在{"{"}忠文坊{"}"}的古城区，古老的建筑沐浴在阳光下。 -> Ở khu phố cổ ở {"{"}Phường Trung Văn{"}"}, các công trình cổ được tắm nắng.
在{"{"}柴峰{"}"}的运河畔，古老的石桥依然屹立。 -> Bên bờ kênh đào ở {"{"}Sài Phong{"}"}， cầu đá cổ vẫn đứng vững.
在{"{"}和秀二社{"}"}的古城墙上，青石板铺就的路延伸而去。 -> Trên tường thành cổ ở {"{"}Xã Hòa Tú 2{"}"}, con đường được lát bằng đá xanh kéo dài.
請幫我將這一句中文翻譯成越南語:
{source} -> """

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