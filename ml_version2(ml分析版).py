# -*- coding: utf-8 -*-
"""
使用多種回歸模型分析奧運對Airbnb價格和情緒的影響
"""

import pandas as pd
from pandas import to_datetime
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MultiModelRegression:
    """多模型回歸分析類"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
    def deal_sentiment(self, sentiment_df):
        """處理情緒數據"""
        print("=== 處理情緒數據 ===")

        sentiment_df['timestamp'] = to_datetime(sentiment_df['timestamp'])
        # 分類奧運前後
        sentiment_df['olympics_period'] = np.where(
            sentiment_df['timestamp'] < to_datetime('2024-07-01'), 
            'Before Olympics', 
            'After Olympics'
        )
        
        print(f"奧運前後樣本數:\n{sentiment_df['olympics_period'].value_counts()}")
        
        return sentiment_df
    
    def prepare_data(self, sentiment_df, price_df):
        """準備訓練數據"""
        print("=== 準備數據 ===")
        
        # 1. 創建時期統計特徵
        period_price_stats = {
            'Before Olympics': {
                'avg_price_change': 0,
                'std_price_change': price_df['period1_avg_price'].std(),
                'avg_price': price_df['period1_avg_price'].mean(),
                'median_price': price_df['period1_avg_price'].median(),
                'price_range': price_df['period1_avg_price'].max() - price_df['period1_avg_price'].min()
            },
            'After Olympics': {
                'avg_price_change': price_df['price_difference'].mean(),
                'std_price_change': price_df['price_difference'].std(),
                'avg_price': price_df['period2_avg_price'].mean(),
                'median_price': price_df['period2_avg_price'].median(),
                'price_range': price_df['period2_avg_price'].max() - price_df['period2_avg_price'].min()
            }
        }
        
        # 2. 構建特徵矩陣
        X = []
        y = []
        
        for _, row in sentiment_df.iterrows():
            period = row['olympics_period']
            stats = period_price_stats[period]
            
            features = [
                1 if period == 'After Olympics' else 0,  # 是否奧運後
                stats['avg_price_change'],               # 平均價格變化
                stats['std_price_change'],               # 價格變化標準差
                stats['avg_price'],                      # 平均價格
                stats['median_price'],                   # 中位數價格
                stats['price_range'],                    # 價格範圍
                np.log1p(stats['avg_price']),           # 對數價格
                stats['avg_price_change'] / (stats['avg_price'] + 1e-5),  # 相對價格變化
            ]
            
            X.append(features)
            y.append(row['sentiment_polarity'])
        
        self.X = np.array(X)
        self.y = np.array(y)
        
        # 3. 特徵名稱
        self.feature_names = [
            'is_after_olympics', 'avg_price_change', 'std_price_change',
            'avg_price', 'median_price', 'price_range', 'log_avg_price',
            'relative_price_change'
        ]
        
        print(f"特徵維度: {self.X.shape}")
        print(f"目標變量數量: {len(self.y)}")
        
        # 4. 分割數據集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # 5. 標準化特徵
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def define_models(self):
        """定義所有要測試的模型"""
        self.models = {
            # 線性模型
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(alpha=1.0),
            
            'Lasso Regression': Lasso(alpha=0.1, max_iter=1000),
            
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            
            # 樹模型
            'Decision Tree': DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=20,
                random_state=42
            ),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            
            'AdaBoost': AdaBoostRegressor(
                n_estimators=50,
                learning_rate=1.0,
                random_state=42
            ),
            
            # 其他模型
            'KNN': KNeighborsRegressor(
                n_neighbors=10,
                weights='distance'
            ),
            
            'SVM': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            )
        }
        
        # 如果安裝了 XGBoost 和 LightGBM
        try:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        except:
            print("XGBoost 未安裝")
        
        try:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=-1
            )
        except:
            print("LightGBM 未安裝")
    
    def train_and_evaluate(self):
        """訓練和評估所有模型"""
        print("\n=== 訓練和評估模型 ===")
        
        for name, model in self.models.items():
            print(f"\n訓練 {name}...")
            
            try:
                # 訓練模型
                model.fit(self.X_train_scaled, self.y_train)
                
                # 預測
                y_pred_train = model.predict(self.X_train_scaled)
                y_pred_test = model.predict(self.X_test_scaled)
                
                # 計算指標
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                test_mse = mean_squared_error(self.y_test, y_pred_test)
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                
                # 交叉驗證
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                           cv=5, scoring='r2')
                
                # 保存結果
                self.results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred_test
                }
                
                print(f"  訓練 R²: {train_r2:.4f}")
                print(f"  測試 R²: {test_r2:.4f}")
                print(f"  測試 MSE: {test_mse:.4f}")
                print(f"  測試 MAE: {test_mae:.4f}")
                print(f"  交叉驗證 R² (平均±標準差): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"  錯誤: {e}")

        if self.results:
            best_model_name = max(self.results, key=lambda x: self.results[x]['test_r2'])
            self.best_model = self.results[best_model_name]['model']

    def hyperparameter_tuning(self, model_name='Random Forest'):
        """對最佳模型進行超參數調優"""
        print(f"\n=== {model_name} 超參數調優 ===")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 10, 20],
                'min_samples_leaf': [1, 5, 10]
            }
            model = RandomForestRegressor(random_state=42)
            
        elif model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42)
            
        else:
            print(f"不支援 {model_name} 的超參數調優")
            return
        
        # 網格搜索
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"\n最佳參數: {grid_search.best_params_}")
        print(f"最佳交叉驗證分數: {grid_search.best_score_:.4f}")
        
        # 評估最佳模型
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test_scaled)
        test_r2 = r2_score(self.y_test, y_pred)
        print(f"測試集 R²: {test_r2:.4f}")
        
        return best_model
    
    def feature_importance_analysis(self):
        """分析特徵重要性"""
        print("\n=== 特徵重要性分析 ===")
        
        # 使用支持特徵重要性的模型
        importance_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        plot_idx = 0
        for model_name in importance_models:
            if model_name in self.results:
                model = self.results[model_name]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # 創建特徵重要性 DataFrame
                    feature_imp = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # 繪圖
                    if plot_idx < 4:
                        ax = axes[plot_idx]
                        feature_imp.plot(kind='barh', x='feature', y='importance', 
                                       ax=ax, legend=False)
                        ax.set_title(f'{model_name} 特徵重要性')
                        ax.set_xlabel('重要性')
                        plot_idx += 1
                    
                    print(f"\n{model_name} 特徵重要性:")
                    print(feature_imp)
        
        # plt.tight_layout()
        # plt.show()
    
    def visualize_feature_sentiment_relationship(self):
        """視覺化特徵與情緒的關係"""
        print("\n=== 特徵與情緒關係視覺化 ===")
        
        # 將 X 和 y 轉為 DataFrame 以便視覺化
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df['sentiment_polarity'] = self.y
        
        # 1. 相關性熱圖
        plt.figure(figsize=(10, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('特徵與情緒的相關性熱圖')
        plt.tight_layout()
        plt.show()
        
        # 2. 散點圖：每個特徵與 sentiment_polarity 的關係
        n_features = len(self.feature_names)
        n_cols = 2
        n_rows = (n_features + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.feature_names):
            sns.scatterplot(data=df, x=feature, y='sentiment_polarity', ax=axes[i], alpha=0.5)
            axes[i].set_title(f'{feature} vs Sentiment Polarity')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Sentiment Polarity')
        
        # 隱藏多餘的子圖
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    def generate_insights(self, sentiment_df, price_df):
        """生成分析洞察"""
        print("\n=== 分析洞察 ===")
        
        # 1. 最佳模型總結
        best_model_name = max(self.results, key=lambda x: self.results[x]['test_r2'])
        best_result = self.results[best_model_name]
        
        print(f"\n1. 模型性能總結:")
        print(f"   最佳模型: {best_model_name}")
        print(f"   測試集 R²: {best_result['test_r2']:.4f}")
        print(f"   解釋: 模型可以解釋 {best_result['test_r2']*100:.1f}% 的情緒變異")
        
        # 2. 特徵影響分析
        if hasattr(self.best_model, 'coef_'):
            print(f"\n2. 線性模型係數分析:")
            coef_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': self.best_model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
            print(coef_df)
        
        # 3. 價格變化影響
        print(f"\n3. 價格變化對情緒的影響:")
        avg_price_change = price_df['price_difference'].mean()
        if avg_price_change > 0:
            print(f"   平均價格上漲: ${avg_price_change:.2f}")
            print("   建議: 房東應考慮價格上漲對顧客滿意度的影響")
        
        

# 主程序
def main():
    """執行完整的多模型分析"""
    print("=== 多模型回歸分析 - 奧運對Airbnb價格和情緒的影響 ===\n")
    
    # 載入數據（使用您的實際數據路徑）
    try:
        sentiment_df = pd.read_csv('~/airbnb_paris_translated_sentiment.csv')
        price_df = pd.read_csv('price_comparison_result.csv')  # 或使用 result
    except:
        print("使用模擬數據進行演示...")
        # 創建模擬數據
        np.random.seed(42)
        n_samples = 1000
        
        sentiment_df = pd.DataFrame({
            'olympics_period': ['Before Olympics'] * (n_samples//2) + ['After Olympics'] * (n_samples//2),
            'sentiment_polarity': np.random.normal(0.1, 0.3, n_samples)
        })
        
        price_df = pd.DataFrame({
            'period1_avg_price': np.random.normal(100, 30, 100),
            'period2_avg_price': np.random.normal(150, 40, 100),
            'price_difference': np.random.normal(50, 20, 100)
        })
    # 處理情緒數據
    deal_sentiment_df = MultiModelRegression().deal_sentiment(sentiment_df)
    sentiment_df = deal_sentiment_df

    # 創建分析器
    analyzer = MultiModelRegression()
    
    # 準備數據
    analyzer.prepare_data(sentiment_df, price_df)
    
    # 圖像化特徵與情緒的關係
    analyzer.visualize_feature_sentiment_relationship()
    
    # 定義模型
    analyzer.define_models()
    
    # 訓練和評估
    analyzer.train_and_evaluate()
    
    # 特徵重要性分析
    analyzer.feature_importance_analysis()
    
    # 視覺化結果
    # analyzer.visualize_results()
    
    # 超參數調優（可選）
    # best_tuned_model = analyzer.hyperparameter_tuning('Random Forest')
    
    # 生成洞察
    analyzer.generate_insights(sentiment_df, price_df)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()