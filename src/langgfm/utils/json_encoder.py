import json

class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        # 如果是列表，将列表格式化为一行
        if isinstance(obj, list):
            return '[' + ', '.join(super().encode(item) for item in obj) + ']'
        # 如果是字典，逐项处理
        elif isinstance(obj, dict):
            items = []
            for key, value in obj.items():
                encoded_key = super().encode(key)  # 编码键
                encoded_value = self.encode(value)  # 递归处理值
                items.append(f'{encoded_key}: {encoded_value}')
            return '{\n  ' + ',\n  '.join(items) + '\n}'
        # 其他类型使用默认编码
        return super().encode(obj)

data = {
    "key1": [1, 2, 3, 4],
    "key2": ["a", "b", "c"],
    "key3": {"nested_key": [10, 20, 30]}
}

# 使用自定义 JSON Encoder
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, cls=CustomJSONEncoder)