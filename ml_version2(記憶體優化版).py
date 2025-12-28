# -*- coding: utf-8 -*-
"""
記憶體優化版本 - 奧運對Airbnb價格和情緒影響分析
"""

import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

def optimize_dtypes(df):
    """優化數據類型以節省記憶體"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    return df

def process_calendar_in_chunks(file_path, chunk_size=500000):
    """分塊處理大型 calendar 檔案"""
    print("開始分塊處理 calendar 數據...")
    
    # 定義時間段
    period1_start = pd.to_datetime('2023-9-05')
    period1_end = pd.to_datetime('2024-06-30')
    period2_start = pd.to_datetime('2024-07-01')
    period2_end = pd.to_datetime('2024-09-06')
    
    # 初始化結果字典
    period1_prices = {}
    period2_prices = {}
    period1_counts = {}
    period2_counts = {}
    
    # 分塊讀取
    chunk_num = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk_num += 1
        print(f"  處理第 {chunk_num} 塊 (共 {len(chunk)} 筆)...")
        
        # 只保留需要的欄位
        chunk = chunk[['listing_id', 'date', 'adjusted_price']].dropna()
        
        # 轉換日期
        chunk['date'] = pd.to_datetime(chunk['date'])
        
        # 處理價格
        chunk['adjusted_price'] = chunk['adjusted_price'].str.replace('[\$,]', '', regex=True).astype(float)
        
        # 篩選期間1的數據
        mask1 = (chunk['date'] >= period1_start) & (chunk['date'] <= period1_end)
        period1_chunk = chunk[mask1]
        
        # 篩選期間2的數據
        mask2 = (chunk['date'] >= period2_start) & (chunk['date'] <= period2_end)
        period2_chunk = chunk[mask2]
        
        # 累加期間1的價格
        for listing_id, group in period1_chunk.groupby('listing_id'):
            if listing_id not in period1_prices:
                period1_prices[listing_id] = 0
                period1_counts[listing_id] = 0
            period1_prices[listing_id] += group['adjusted_price'].sum()
            period1_counts[listing_id] += len(group)
        
        # 累加期間2的價格
        for listing_id, group in period2_chunk.groupby('listing_id'):
            if listing_id not in period2_prices:
                period2_prices[listing_id] = 0
                period2_counts[listing_id] = 0
            period2_prices[listing_id] += group['adjusted_price'].sum()
            period2_counts[listing_id] += len(group)
        
        # 清理記憶體
        del chunk
        gc.collect()
    
    print(f"  共處理了 {chunk_num} 個資料塊")
    
    # 計算平均價格
    print("計算平均價格...")
    
    # 創建結果 DataFrame
    all_listings = set(period1_prices.keys()) | set(period2_prices.keys())
    
    result_data = []
    for listing_id in all_listings:
        row = {'listing_id': listing_id}
        
        # 期間1平均價格
        if listing_id in period1_prices and period1_counts[listing_id] > 0:
            row['period1_avg_price'] = period1_prices[listing_id] / period1_counts[listing_id]
        else:
            row['period1_avg_price'] = np.nan
        
        # 期間2平均價格
        if listing_id in period2_prices and period2_counts[listing_id] > 0:
            row['period2_avg_price'] = period2_prices[listing_id] / period2_counts[listing_id]
        else:
            row['period2_avg_price'] = np.nan
        
        result_data.append(row)
    
    result = pd.DataFrame(result_data)
    result['price_difference'] = result['period2_avg_price'] - result['period1_avg_price']
    
    # 移除缺失值
    result = result.dropna()
    
    print(f"價格比較完成: {len(result)} 個房源")
    
    return result

def process_reviews_efficiently(file_path):
    """高效處理評論數據"""
    print("\n處理評論數據...")
    
    # 讀取評論
    emoji = pd.read_csv(file_path, usecols=['listing_id', 'date', 'comments'])
    print(f"  原始評論數: {len(emoji)}")
    
    # 優化記憶體
    emoji = optimize_dtypes(emoji)
    
    # 轉換日期
    emoji['date'] = pd.to_datetime(emoji['date'])
    
    # 定義時間段
    period_start = pd.to_datetime('2024-7-01')
    period_end = pd.to_datetime('2024-09-06')
    
    # 篩選時間段
    period_data = emoji[(emoji['date'] >= period_start) & (emoji['date'] <= period_end)]
    print(f"  奧運期間評論數: {len(period_data)}")
    
    # 清理記憶體
    del emoji
    gc.collect()
    
    # 移除中性和不相關評論
    mask = ~period_data['comments'].isin(['Neutral', 'Irrelevant'])
    clean_data = period_data[mask].reset_index(drop=True)
    print(f"  清理後評論數: {len(clean_data)}")
    
    return clean_data

def sentiment_analysis_batch(clean_data, batch_size=1000):
    """批次進行情緒分析"""
    print("\n進行情緒分析...")
    
    import nltk
    try:
        nltk.download('vader_lexicon', quiet=True)
    except:
        pass
    
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    
    # 初始化結果欄位
    clean_data['compound'] = 0.0
    clean_data['pos'] = 0.0
    clean_data['neg'] = 0.0
    clean_data['neu'] = 0.0
    
    # 批次處理
    total_batches = (len(clean_data) + batch_size - 1) // batch_size
    
    for i in range(0, len(clean_data), batch_size):
        batch_end = min(i + batch_size, len(clean_data))
        batch_num = i // batch_size + 1
        
        if batch_num % 10 == 0:
            print(f"  處理批次 {batch_num}/{total_batches}...")
        
        for idx in range(i, batch_end):
            text = clean_data.loc[idx, 'comments']
            if isinstance(text, str):
                scores = sid.polarity_scores(text)
                clean_data.loc[idx, 'compound'] = scores['compound']
                clean_data.loc[idx, 'pos'] = scores['pos']
                clean_data.loc[idx, 'neg'] = scores['neg']
                clean_data.loc[idx, 'neu'] = scores['neu']
    
    print("  情緒分析完成")
    
    return clean_data

def simple_analysis(sentiment_data, price_data):
    """簡單的統計分析"""
    print("\n=== 分析結果 ===")
    
    # 合併數據
    merged_data = pd.merge(sentiment_data, price_data, on='listing_id', how='inner')
    print(f"合併後數據: {len(merged_data)} 筆")
    
    if len(merged_data) == 0:
        print("警告：沒有匹配的數據！")
        return
    
    # 基本統計
    print("\n1. 價格變化統計:")
    print(f"   平均價格變化: ${price_data['price_difference'].mean():.2f}")
    print(f"   中位數價格變化: ${price_data['price_difference'].median():.2f}")
    print(f"   漲價房源比例: {(price_data['price_difference'] > 0).mean():.2%}")
    print(f"   降價房源比例: {(price_data['price_difference'] < 0).mean():.2%}")
    
    print("\n2. 情緒分析統計:")
    print(f"   平均情緒分數: {merged_data['compound'].mean():.4f}")
    print(f"   正面評論比例: {(merged_data['compound'] > 0.05).mean():.2%}")
    print(f"   負面評論比例: {(merged_data['compound'] < -0.05).mean():.2%}")
    
    # 價格變化與情緒的關係
    print("\n3. 價格變化與情緒關係:")
    
    # 將價格變化分組
    merged_data['price_change_category'] = pd.cut(
        merged_data['price_difference'],
        bins=[-np.inf, -50, 0, 50, 100, np.inf],
        labels=['大幅降價', '小幅降價', '小幅漲價', '中幅漲價', '大幅漲價']
    )
    
    # 計算每組的平均情緒
    sentiment_by_price = merged_data.groupby('price_change_category')['compound'].agg(['mean', 'count'])
    print("\n價格變化類別的平均情緒:")
    print(sentiment_by_price)
    
    # 相關性分析
    if len(merged_data) > 30:
        correlation = merged_data['price_difference'].corr(merged_data['compound'])
        print(f"\n4. 價格變化與情緒的相關係數: {correlation:.4f}")
        
        if correlation < -0.1:
            print("   解釋: 價格上漲與負面情緒有關")
        elif correlation > 0.1:
            print("   解釋: 價格上漲與正面情緒有關")
        else:
            print("   解釋: 價格變化與情緒關係不明顯")

def main():
    """主程序"""
    print("=== 記憶體優化版 - 奧運對 Airbnb 影響分析 ===\n")
    
    try:
        # 1. 處理價格數據
        price_result = process_calendar_in_chunks('~/archive/calendar.csv', chunk_size=500000)
        
        # 保存中間結果
        price_result.to_csv('price_comparison_result.csv', index=False)
        print("價格數據已保存到 price_comparison_result.csv")
        
        # 2. 處理評論數據
        review_data = process_reviews_efficiently('~/archive/reviews.csv')
        
        # 3. 情緒分析
        sentiment_data = sentiment_analysis_batch(review_data, batch_size=1000)
        
        # 保存情緒分析結果
        sentiment_data.to_csv('sentiment_analysis_result.csv', index=False)
        print("情緒分析結果已保存到 sentiment_analysis_result.csv")
        
        # 4. 簡單分析
        simple_analysis(sentiment_data, price_result)
        
        print("\n✓ 分析完成！")
        
    except MemoryError:
        print("\n記憶體錯誤！建議：")
        print("1. 減少 chunk_size (如改為 100000)")
        print("2. 關閉其他程式釋放記憶體")
        print("3. 使用 Google Colab 或更大記憶體的環境")
    
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 顯示記憶體使用情況
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"系統記憶體: {memory.total / 1024**3:.1f} GB")
        print(f"可用記憶體: {memory.available / 1024**3:.1f} GB\n")
    except:
        pass
    
    main()