"""Utility functions: memory optimization, timing, caching."""
import time
import pickle
import os
import numpy as np
import pandas as pd


def reduce_mem(df):
    """Downcast numeric columns to smallest dtype to reduce memory usage."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != "category":
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory: {start_mem:.1f}MB -> {end_mem:.1f}MB ({100*(start_mem-end_mem)/start_mem:.0f}% reduction)")
    return df


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"{self.label} done in {elapsed:.1f}s")


def cache_result(path):
    """Decorator that caches function result to pickle."""
    def decorator(func):
        def wrapper(*args, use_cache=True, **kwargs):
            if use_cache and os.path.exists(path):
                print(f"Loading cache: {path}")
                with open(path, "rb") as f:
                    return pickle.load(f)
            result = func(*args, **kwargs)
            with open(path, "wb") as f:
                pickle.dump(result, f)
            print(f"Cached: {path}")
            return result
        return wrapper
    return decorator
