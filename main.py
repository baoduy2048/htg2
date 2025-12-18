import pandas as pd
import numpy as np
import os
import warnings

# Import các module đã tạo
from preprocessing import DataProcessor, post_process_cleaning, remove_outliers
from feature_engineering import process_and_transform_data
from modeling import (
    train_catboost_model, 
    calculate_metrics, 
    get_feature_importance, 
    filter_features_by_importance
)
import visualization

# Tắt cảnh báo để output gọn gàng hơn
warnings.filterwarnings('ignore')

def load_and_prepare_data(train_path, test_path):
    """Đọc và gộp dữ liệu Train/Test."""
    print("Loading data...")
    dtypes = dict(
        Month="category",
        Name="category",
        Occupation="category",
        Type_of_Loan="category",
        Credit_History_Age="category",
        Payment_Behaviour="category"
    )
    
    train_df = pd.read_csv(train_path, dtype=dtypes, parse_dates=['Month'])
    train_df["is_train"] = True
    
    test_df = pd.read_csv(test_path, dtype=dtypes, parse_dates=['Month'])
    test_df["is_train"] = False
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Drop cột không cần thiết ngay từ đầu
    df.drop(["Name", "SSN", "ID"], axis=1, inplace=True, errors="ignore")
    
    return df

def identify_feature_types(X_df):
    """Xác định cột Categorical và Text cho CatBoost."""
    # Trong notebook gốc, Type_of_Loan được xử lý như text feature
    text_features = ["Type_of_Loan"] if "Type_of_Loan" in X_df.columns else []
    
    # Các cột còn lại là categorical (trừ text features)
    cat_features = X_df.select_dtypes(exclude="number").columns.drop(
        text_features, errors='ignore'
    ).tolist()
    
    return cat_features, text_features

def main():
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    # Bạn hãy thay đổi đường dẫn này trỏ đến file csv thực tế của bạn
    TRAIN_PATH = "train.csv" 
    TEST_PATH = "test.csv"
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print(f"Error: Không tìm thấy file dữ liệu tại {TRAIN_PATH} hoặc {TEST_PATH}")
        print("Vui lòng cập nhật đường dẫn trong hàm main() của file main.py")
        return

    # 1. LOAD & INITIAL PROCESSING
    # ---------------------------------------------------------
    df = load_and_prepare_data(TRAIN_PATH, TEST_PATH)
    
    print("Preprocessing data (cleaning text, fixing values)...")
    processor = DataProcessor("Customer_ID", df)
    df_clean = processor.preprocess()
    
    # Các bước làm sạch bổ sung
    df_clean = post_process_cleaning(df_clean)
    
    # Xử lý Outlier trên toàn bộ dataset trước khi split (theo logic notebook cũ)
    # Lưu ý: Trong thực tế nên fit trên train và transform trên test để tránh data leakage,
    # nhưng ở đây ta giữ nguyên logic của notebook gốc để đảm bảo kết quả tương đồng.
    print("Handling outliers...")
    df_clean = remove_outliers(df_clean)

    # (Tùy chọn) Vẽ biểu đồ EDA
    # visualization.plot_numerical_boxplots(df_clean[df_clean["is_train"]])
    # visualization.plot_correlation_heatmap(df_clean[df_clean["is_train"]])

    # 2. FEATURE ENGINEERING & SPLIT
    # ---------------------------------------------------------
    print("Transforming data (Imputing, Scaling)...")
    (X_train, X_val, y_train, y_val), X_submission = process_and_transform_data(df_clean, save_pkl=True)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Xác định loại cột cho CatBoost
    cat_cols, text_cols = identify_feature_types(X_train)
    print(f"Categorical features: {len(cat_cols)}")
    print(f"Text features: {text_cols}")

    # 3. INITIAL MODEL TRAINING
    # ---------------------------------------------------------
    print("\n--- Training Initial Model ---")
    model_v1, val_pool_v1 = train_catboost_model(
        X_train, y_train, 
        X_val, y_val, 
        cat_cols, text_cols,
        params={"task_type": "CPU", "iterations": 1000}, # Giảm iterations để test nhanh
        verbose=200
    )
    
    calculate_metrics(model_v1, X_val, y_val)

    # 4. FEATURE SELECTION & REFINEMENT
    # ---------------------------------------------------------
    print("\n--- Feature Refinement ---")
    importance_df = get_feature_importance(model_v1, val_pool_v1)
    
    # (Tùy chọn) Vẽ biểu đồ importance
    # visualization.plot_feature_importance(importance_df)
    
    # Lọc bỏ các feature có độ quan trọng < 0 (hoặc = 0)
    # Trong notebook gốc lọc < 0.000
    features_to_drop = importance_df.query("Importances <= 0.0")["Feature Id"].tolist()
    
    if features_to_drop:
        print(f"Removing {len(features_to_drop)} features: {features_to_drop}")
        X_train_refined = X_train.drop(columns=features_to_drop, errors='ignore')
        X_val_refined = X_val.drop(columns=features_to_drop, errors='ignore')
        X_submission_refined = X_submission.drop(columns=features_to_drop, errors='ignore')
        
        # Cập nhật lại danh sách features
        cat_cols_refined, text_cols_refined = identify_feature_types(X_train_refined)

        # 5. RETRAIN REFINED MODEL
        # ---------------------------------------------------------
        print("\n--- Training Refined Model ---")
        model_final, _ = train_catboost_model(
            X_train_refined, y_train, 
            X_val_refined, y_val, 
            cat_cols_refined, text_cols_refined,
            params={
                "task_type": "CPU", 
                "iterations": 2000, 
                "max_depth": 7 # Tăng depth như notebook gốc
            },
            verbose=200
        )
        
        calculate_metrics(model_final, X_val_refined, y_val)
        
        # Save model
        model_final.save_model("credit_score_model_refined.cbm")
        print("Refined model saved.")
        
    else:
        print("No features to drop. Using initial model as final.")
        model_final = model_v1
        X_submission_refined = X_submission
        model_final.save_model("credit_score_model_v1.cbm")

    # 6. FINAL SUBMISSION PREDICTION
    # ---------------------------------------------------------
    print("\nGenerating submission predictions...")
    submission_preds = model_final.predict(X_submission_refined)
    
    # Giả sử file test gốc có cột Customer_ID để map lại (cần load lại raw test để lấy ID)
    # Ở đây chỉ lưu kết quả thô
    submission_df = pd.DataFrame(submission_preds, columns=["Credit_Score"])
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved to 'submission.csv'")

if __name__ == "__main__":
    main()
