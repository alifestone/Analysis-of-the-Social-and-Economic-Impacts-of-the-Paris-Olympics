# Analysis-of-the-Social-and-Economic-Impacts-of-the-Paris-Olympics
## Abstract
本研究旨在探討 Airbnb 價格在奧運前後是否與社群情緒變動存在可預測性關聯。目標變數為由 TextBlob 分析出的留言情緒分數。特徵則為從 Airbnb 價格資料中處理過的數據，如價格變化、分散程度與時間區分等，作為模型學習預測依據。<br>
由於社群情緒受眾多潛在因素影響，資料極具雜訊，因此在實作層面，我們將資料依照留言主題（Airbnb、Olympic）分別建立模型，統一採用多種監督式回歸模型，並搭配特徵工程與模型評估流程。
## Workflow
原始資料<br>
├── calendar.csv (Kaggle)<br>
├── reviews.csv (Kaggle)<br>
├── listings.csv (Kaggle)<br>
└── Reddit 爬蟲資料<br>
         │<br>
         ▼<br>
    reddit_sensor_priority.ipynb<br>
         │<br>
         ▼<br>
    airbnb_paris_translated_sentiment.csv<br>
         │<br>
         │     ml.ipynb<br>
         │         │<br>
         │         ▼<br>
         │     (資料探索與視覺化)<br>
         │<br>
         │     ml_version2.py (記憶體優化版)<br>
         │         │<br>
         │         ▼<br>
         │     price_comparison_result.csv<br>
         │         │<br>
         └────┬────┘<br>
              │<br>
              ▼<br>
    ml_version2.py (ML分析版)<br>
              │<br>
              ▼<br>
    ┌─────────────────┐<br>
    │ • R² 表格       │<br>
    │ • 相關性熱圖    │<br>
    │ • 特徵重要性    │<br>
    └─────────────────┘<br>
<br>
## Author
張逸安, 黃裕媞, 周聖詠, 謝蕙宇<br>
指導老師：何承遠
## 銘謝
感謝何承遠老師在研究過程中提供的指導與建議，在遇到問題與困難時給予實質上的協助與方向上的啟發。
