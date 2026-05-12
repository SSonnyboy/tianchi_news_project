"""数据统计分析脚本：查看原始数据和 LOO 切分后的数据分布。"""
import pandas as pd
import numpy as np
from src.config import TRAIN_CLICK_PATH, COL_USER, COL_ITEM, COL_TIME


def analyze_raw_data():
    """分析原始数据的用户点击分布。"""
    df = pd.read_csv(TRAIN_CLICK_PATH)
    user_counts = df.groupby(COL_USER).size()

    print("=" * 50)
    print("  原始数据统计")
    print("=" * 50)
    print(f"总点击数: {len(df):,}")
    print(f"总用户数: {df[COL_USER].nunique():,}")
    print(f"总物品数: {df[COL_ITEM].nunique():,}")
    print()

    buckets = {
        "1 次": (user_counts == 1).sum(),
        "2 次": (user_counts == 2).sum(),
        "3-5 次": ((user_counts >= 3) & (user_counts <= 5)).sum(),
        "6-10 次": ((user_counts >= 6) & (user_counts <= 10)).sum(),
        "11-20 次": ((user_counts >= 11) & (user_counts <= 20)).sum(),
        "21-50 次": ((user_counts >= 21) & (user_counts <= 50)).sum(),
        "50+ 次": (user_counts > 50).sum(),
    }
    total = len(user_counts)
    print("用户点击次数分布:")
    for label, count in buckets.items():
        print(f"  {label:>8s}: {count:>7,} ({count / total * 100:5.1f}%)")

    print(f"\n  平均: {user_counts.mean():.1f}, 中位数: {user_counts.median():.0f}, "
          f"最大: {user_counts.max()}")


def analyze_loo_split():
    """分析 LOO 切分后训练集中用户的点击次数分布。"""
    df = pd.read_csv(TRAIN_CLICK_PATH)
    df = df.drop_duplicates(subset=[COL_USER, COL_ITEM, COL_TIME])
    df = df.sort_values([COL_USER, COL_TIME])

    # LOO: 每个用户最后一次点击作为测试集
    last_clicks = df.groupby(COL_USER).tail(1)
    train_df = df.drop(last_clicks.index)

    train_counts = train_df.groupby(COL_USER).size()
    test_users = set(last_clicks[COL_USER])

    print()
    print("=" * 50)
    print("  LOO 切分后统计")
    print("=" * 50)
    print(f"训练集点击数: {len(train_df):,}")
    print(f"测试集用户数: {len(test_users):,}")

    cold_start = len(test_users) - len(set(train_counts.index) & test_users)
    print(f"冷启动用户 (训练集 0 点击): {cold_start}")
    print()

    buckets = {
        "0 次 (冷启动)": sum(1 for u in test_users if u not in train_counts),
        "1 次": (train_counts == 1).sum(),
        "2 次": (train_counts == 2).sum(),
        "3-5 次": ((train_counts >= 3) & (train_counts <= 5)).sum(),
        "6-10 次": ((train_counts >= 6) & (train_counts <= 10)).sum(),
        "11-20 次": ((train_counts >= 11) & (train_counts <= 20)).sum(),
        "21+ 次": (train_counts > 20).sum(),
    }
    total = len(test_users)
    print("训练集点击次数分布 (测试用户):")
    for label, count in buckets.items():
        print(f"  {label:>16s}: {count:>7,} ({count / total * 100:5.1f}%)")


if __name__ == "__main__":
    analyze_raw_data()
    analyze_loo_split()
