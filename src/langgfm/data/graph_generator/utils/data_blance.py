import pandas as pd


def balance_dataframe(df, label_column='label', random_state=42):
    """
    对 DataFrame 进行下采样，使得指定标签列中各类别样本数量相同
    
    参数:
    df (pd.DataFrame): 输入的 DataFrame
    label_column (str): 标签列的列名，默认为 'label'
    random_state (int): 随机种子，保证结果可重复，默认为 42
    
    返回:
    pd.DataFrame: 经过下采样后的平衡 DataFrame
    """
    
    # 获取标签列中的所有类别
    label_counts = df[label_column].value_counts()
    
    # 找出最少的类别样本数
    min_count = label_counts.min()
    
    # 对每个类别进行下采样
    balanced_df_list = []
    for label in label_counts.index:
        df_label = df[df[label_column] == label]
        df_label_downsampled = df_label.sample(min_count, random_state=random_state)
        balanced_df_list.append(df_label_downsampled)
    
    # 合并所有下采样后的 DataFrame
    df_balanced = pd.concat(balanced_df_list)
    
    # 打乱顺序
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df_balanced
