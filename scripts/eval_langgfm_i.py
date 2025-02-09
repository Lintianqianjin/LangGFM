import json
import fire

def compare_first_part(file_path):
    total = 0            # 总行数
    consistent_count = 0 # 一致的条数
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"第 {total} 行 JSON 格式错误：{e}")
                continue
            
            # 获取 predict 和 label 字段，若不存在则使用空字符串
            predict_text = data.get("predict", "")
            label_text = data.get("label", "")
            
            # 按英文句号进行 split，并取第一个部分（去除前后空格）
            predict_first = predict_text.split('.')[0].strip()
            label_first = label_text.split('.')[0].strip()
            
            # 比较并输出结果
            if predict_first == label_first:
                print(f"第 {total} 行：一致")
                consistent_count += 1
            else:
                print(f"第 {total} 行：不一致")
                print(f"    predict 第一部分：{predict_first}")
                print(f"    label 第一部分：{label_first}")
    
    print("\n比较完毕")
    print(f"总共 {total} 条记录，其中一致 {consistent_count} 条，不一致 {total - consistent_count} 条。")

if __name__ == '__main__':
    # 请将 'data.jsonl' 替换为你的文件路径
    fire.Fire(compare_first_part)
    # compare_first_part('experiments/langgfm_i/wikics/test_200/ckpts/Qwen2.5-7B-Instruct/lora_rank=64/lora_alpha=256/lora_dropout=0.0/learning_rate=2e-05/num_train_epochs=50/warmup_ratio=0.2/batch_size=64/checkpoint-100/predictions.json')
