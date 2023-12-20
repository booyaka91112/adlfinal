def get_prompt(source: str) -> str:
    return f"請幫我將這一句中文翻譯成越南語: {source}"

def get_prompt_3_shot(source: str) -> str:
    return f"""請幫我將中文翻譯成越南語:
在北京的大街上，人群熙熙攘攘。 -> Trên con đường ở bắc kinh, đám đông đông đúc. 
在上海的公园里，人们散步欣赏花草。 -> Trong công viên ở thướng hải, mọi người đi dạo ngắm hoa cỏ.
靜岡的夜市里，各种美食琳琅满目。 -> Trong chợ đêm ở tĩnh cương, có đủ loại đồ ăn ngon.
請幫我將中文翻譯成越南語:
{source} -> """

def get_prompt_6_shot(source: str) -> str:
    return f"""請幫我將中文翻譯成越南語:
在北京的大街上，人群熙熙攘攘。 -> Trên con đường ở bắc kinh, đám đông đông đúc. 
在上海的公园里，人们散步欣赏花草。 -> Trong công viên ở thướng hải, mọi người đi dạo ngắm hoa cỏ.
靜岡的夜市里，各种美食琳琅满目。 -> Trong chợ đêm ở tĩnh cương, có đủ loại đồ ăn ngon.
在南京的古城区，古老的建筑沐浴在阳光下。 -> Ở khu phố cổ ở nam kinh, các công trình cổ được tắm nắng.
在杭州的运河畔，古老的石桥依然屹立。 -> Bên bờ kênh đào ở hàng châu， cầu đá cổ vẫn đứng vững.
在重慶的古城墙上，青石板铺就的路延伸而去。 -> Trên tường thành cổ ở trọng khanh, con đường được lát bằng đá xanh kéo dài.
請幫我將中文翻譯成越南語:
{source} -> """
