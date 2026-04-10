import joblib
import os
import glob
from tqdm import tqdm
import numpy as np
import argparse


# 1. 修改这里：指向 feature_extraction_papagei.py 输出的那个模型目录
# 例如: .../features/model_name (不要包含 train/val/test 子目录)

# splits = ['train', 'val', 'test']

# level = 'segment'

parser = argparse.ArgumentParser(description="Combine feature embeddings.")
    
# 1. 模型目录参数
parser.add_argument('-f', '--feature_model_dir', type=str, required=True, 
                        help="Path to the feature directory (e.g., .../features/model_name)")
    
# 2. Splits 参数 (支持传入多个值，例如 --splits train val test)
parser.add_argument('-s', '--splits', nargs='+', default=['train', 'val', 'test'],
                    help="List of splits to process (default: train val test)")

# 3. Level 参数
parser.add_argument('-l', '--level', type=str, default='segment', choices=['segment', 'patient'],
                        help="Aggregation level: 'segment' or 'patient'")

args = parser.parse_args()

feature_model_dir = args.feature_model_dir
splits = args.splits
level = args.level

for split in splits:
    split_dir = os.path.join(feature_model_dir, split)
    # 输出文件名通常约定为 dict_{split}.p
    output_file = os.path.join(feature_model_dir, f"dict_{split}_{level}.p")
    
    if not os.path.exists(split_dir):
        print(f"[WARN] 目录不存在，跳过: {split_dir}")
        continue

    print(f"正在合并 {split} 集...")
    files = glob.glob(os.path.join(split_dir, "*.p"))
    combined_dict = {}
    
    for f in tqdm(files):
        # 文件名即 case_id (例如 "2.p" -> "2")
        case_id = os.path.basename(f).replace(".p", "")
        embedding = joblib.load(f)        
        
        if level == 'patient':
            # Patient Level: 对所有片段取平均 (Mean Pooling)
            # embedding shape 通常是 (N_segments, Hidden_dim)
            # 取平均后变成 (Hidden_dim,)
            combined_dict[case_id] = np.mean(embedding, axis=0)
        else:
            # Segment Level: 保持原样
            combined_dict[case_id] = embedding
        
    joblib.dump(combined_dict, output_file)
    print(f"[OK] 已保存 {output_file}，包含 {len(combined_dict)} 个样本。")
