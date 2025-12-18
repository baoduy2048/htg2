import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def create_pool(X, y=None, cat_features=None, text_features=None):
    """
    Tạo đối tượng CatBoost Pool tối ưu cho việc huấn luyện.
    """
    # Đảm bảo features là list hợp lệ hoặc None
    if cat_features is not None and len(cat_features) == 0:
        cat_features = None
    if text_features is not None and len(text_features) == 0:
        text_features = None
        
    return Pool(data=X, label=y, cat_features=cat_features, text_features=text_features)

def train_catboost_model(X_train, y_train, X_val, y_val, cat_features, text_features, params=None, verbose=500):
    """
    Hàm huấn luyện mô hình CatBoost.
    """
    # Cấu hình mặc định (dựa trên notebook gốc)
    default_params = {
        "iterations": 10000,
        "custom_metric": ["F1", "AUC", "Accuracy"],
        "thread_count": -1,
        "random_state": 42,
        "train_dir": "catboost_info",
        "text_processing": ["NaiveBayes+Word|BoW+Word"],
        # Lưu ý: "task_type": "GPU" yêu cầu máy có GPU. 
        # Nếu chạy trên CPU, hãy đổi thành "CPU" hoặc xóa dòng này.
        "task_type": "CPU", 
        "one_hot_max_size": 3,
        "depth": 6,
        "auto_class_weights": "Balanced",
        "bootstrap_type": "Poisson",
        "subsample": 0.5,
        "max_bin": 100
    }

    # Cập nhật tham số nếu người dùng truyền vào
    if params:
        default_params.update(params)

    # Tạo Pools
    train_pool = create_pool(X_train, y_train, cat_features, text_features)
    val_pool = create_pool(X_val, y_val, cat_features, text_features)

    # Khởi tạo và huấn luyện mô hình
    model = CatBoostClassifier(**default_params)
    
    print("Start training CatBoost...")
    model.fit(
        train_pool, 
        eval_set=val_pool, 
        early_stopping_rounds=500, 
        verbose=verbose,
        plot=False
    )
    print("Training finished.")
    
    return model, val_pool

def calculate_metrics(model, X_val, y_val):
    """
    Tính toán và in ra các chỉ số đánh giá mô hình.
    """
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average=None)
    precision = precision_score(y_val, y_pred, average=None)
    recall = recall_score(y_val, y_pred, average=None)
    
    print(f"Accuracy: {acc:.2%}")
    print(f"F1 Score: {dict(zip(model.classes_, [f'{x:.2%}' for x in f1]))}")
    print(f"Precision: {dict(zip(model.classes_, [f'{x:.2%}' for x in precision]))}")
    print(f"Recall: {dict(zip(model.classes_, [f'{x:.2%}' for x in recall]))}")
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred))
    
    return y_pred

def get_feature_importance(model, val_pool):
    """
    Lấy DataFrame chứa mức độ quan trọng của các đặc trưng.
    """
    importance = model.get_feature_importance(val_pool, type="LossFunctionChange", prettified=True)
    return importance

def filter_features_by_importance(df, importance_df, threshold=0.0):
    """
    Loại bỏ các đặc trưng có độ quan trọng thấp hơn ngưỡng (threshold).
    Dùng cho bước Model Refinement.
    """
    low_importance_feats = importance_df.query(f"Importances <= {threshold}")["Feature Id"].tolist()
    
    # Chỉ drop những cột thực sự tồn tại trong df
    cols_to_drop = [col for col in low_importance_feats if col in df.columns]
    
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} features with importance <= {threshold}")
        return df.drop(columns=cols_to_drop)
    
    return df
