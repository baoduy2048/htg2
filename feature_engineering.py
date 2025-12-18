import pandas as pd
import numpy as np
import pickle
import itertools
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# IterativeImputer là tính năng thử nghiệm, cần import enable trước
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency

# Import class từ file preprocessing đã tạo ở bước trước
from preprocessing import ClipOutliersTransformer

def chi_2_test(df: pd.DataFrame, target_col="Credit_Score"):
    """
    Thực hiện kiểm định Chi-Square (cho biến phân loại) và ANOVA F-test (cho biến số)
    để đánh giá mức độ quan trọng của đặc trưng đối với biến mục tiêu.
    """
    # Chỉ lấy dữ liệu train để kiểm định
    df_copy = df.loc[df["is_train"]].copy()
    
    cat_cols = df_copy.select_dtypes(exclude="number").columns.drop(
        ["Customer_ID", "Month", "is_train", target_col], errors='ignore'
    )
    num_cols = df_copy.select_dtypes(include="number").columns
    
    summary = []
    
    # Factorize target variable for ANOVA
    y, *_ = df_copy[target_col].factorize(sort=False)

    # Chi-Square Test for Categorical
    for col in cat_cols:
        cross = pd.crosstab(index=df_copy[col], columns=df_copy[target_col])
        t_stat, pvalue, *_ = chi2_contingency(cross)
        summary.append([col, t_stat, pvalue])

    # ANOVA F-test for Numerical
    for col in num_cols:
        # Xử lý missing value tạm thời để chạy test
        col_data = df_copy[[col]].fillna(df_copy[col].median())
        t_stat, pvalue = f_classif(col_data, y)
        summary.append([col, t_stat[0], pvalue[0]])

    return pd.DataFrame(data=summary, columns=["column", "t-statistic", "p-value"])

def build_transformer_pipeline(data_frame: pd.DataFrame):
    """
    Xây dựng ColumnTransformer chứa các pipeline xử lý:
    - Numerical: Impute Median -> MinMaxScaler
    - Categorical: Impute Most Frequent
    """
    category = data_frame.select_dtypes(exclude="number").columns.tolist()
    number = data_frame.select_dtypes(include="number").columns.tolist()

    def build_pipeline_numb(strategy="median"):
        return Pipeline(steps=[
            ("imputer", IterativeImputer(initial_strategy=strategy, random_state=42)),
            ("scaling", MinMaxScaler())
        ])

    def build_pipeline_cat(strategy="most_frequent"):
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=strategy))
        ])

    transformer = ColumnTransformer(
        [
            ("cat_transformer", build_pipeline_cat(), category),
            ("numb_transformer", build_pipeline_numb(), number)
        ],
        remainder="drop"
    )
    return transformer

def split_data(data: pd.DataFrame, test_size=0.2, target_col="Credit_Score"):
    """
    Chia dữ liệu thành tập Train/Validation và tách tập Test (submission).
    """
    df_copy = data.copy()
    
    # Loại bỏ các cột định danh không dùng cho model
    cols_to_drop = ["Month", "Customer_ID", "Name", "SSN"]
    df_copy.drop(columns=[c for c in cols_to_drop if c in df_copy.columns], inplace=True)
    
    if "Delay_from_due_date" in df_copy.columns:
        df_copy["Delay_from_due_date"] = df_copy["Delay_from_due_date"].abs()

    # Tách tập Train (có nhãn) và Test (không nhãn - để submit)
    train_set_full = df_copy[df_copy["is_train"]].drop(["is_train"], axis=1)
    submission_set = df_copy[~df_copy["is_train"]].drop(["is_train", target_col], axis=1).reset_index(drop=True)

    X = train_set_full.drop(target_col, axis=1)
    y = train_set_full[target_col]

    # Chia Train/Validation
    Xtrain, Xval, ytrain, yval = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42, shuffle=True
    )
    
    return (Xtrain, Xval, ytrain, yval), submission_set

def process_and_transform_data(data: pd.DataFrame, save_pkl=False):
    """
    Hàm chính để thực thi toàn bộ quy trình Feature Engineering:
    1. Split Data
    2. Fit Outlier Transformer
    3. Fit Column Transformer (Impute + Scale)
    4. Transform toàn bộ dữ liệu
    """
    # 1. Split Data
    (Xtrain_raw, Xval_raw, ytrain, yval), submission_raw = split_data(data)

    # 2. Outlier Transformer (Fit trên Train set)
    outlier_remover = ClipOutliersTransformer(0.25, 0.75, multiply_by=1.5, replace_with_median=False)
    # Fit chỉ trên cột số
    num_cols = Xtrain_raw.select_dtypes("number").columns
    outlier_remover.fit(Xtrain_raw[num_cols])
    
    # Transform (lưu ý: logic này cần apply thủ công vào cột số vì Class Clip trả về array)
    # Để đơn giản hóa, ta áp dụng transformer này vào DataFrame
    def apply_outlier_removal(df, transformer, cols):
        df_out = df.copy()
        df_out[cols] = transformer.transform(df[cols])
        return df_out

    Xtrain_out = apply_outlier_removal(Xtrain_raw, outlier_remover, num_cols)
    Xval_out = apply_outlier_removal(Xval_raw, outlier_remover, num_cols)
    submission_out = apply_outlier_removal(submission_raw, outlier_remover, num_cols)

    # 3. Column Transformer (Fit trên Train set đã xử lý outlier)
    transformer = build_transformer_pipeline(Xtrain_out)
    transformer.fit(Xtrain_out)

    # 4. Transform dữ liệu cuối cùng
    # Lấy tên cột sau khi biến đổi để tái tạo DataFrame
    # Lưu ý: ColumnTransformer làm mất tên cột, ta cần logic để lấy lại nếu muốn giữ DataFrame
    # Logic dưới đây tái tạo lại danh sách tên cột từ transformer
    feature_names = []
    
    # Lấy features từ categorical pipeline
    # (SimpleImputer giữ nguyên số lượng cột đầu vào)
    cat_cols = Xtrain_out.select_dtypes(exclude="number").columns.tolist()
    feature_names.extend(cat_cols)
    
    # Lấy features từ numerical pipeline
    num_cols_final = Xtrain_out.select_dtypes(include="number").columns.tolist()
    feature_names.extend(num_cols_final)
    
    # Thực hiện transform
    Xtrain_processed = pd.DataFrame(transformer.transform(Xtrain_out), columns=feature_names)
    Xval_processed = pd.DataFrame(transformer.transform(Xval_out), columns=feature_names)
    submission_processed = pd.DataFrame(transformer.transform(submission_out), columns=feature_names)

    # Convert numeric types
    Xtrain_processed = Xtrain_processed.apply(pd.to_numeric, errors="ignore")
    Xval_processed = Xval_processed.apply(pd.to_numeric, errors="ignore")
    submission_processed = submission_processed.apply(pd.to_numeric, errors="ignore")

    if save_pkl:
        with open("OutlierRemover.pkl", "wb") as f: pickle.dump(outlier_remover, f)
        with open("ColumnsTransformers.pkl", "wb") as f: pickle.dump(transformer, f)

    return (Xtrain_processed, Xval_processed, ytrain, yval), submission_processed
