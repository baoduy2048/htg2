import pandas as pd
import numpy as np
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin

class DataProcessor:
    """
    Class dùng để tiền xử lý và làm sạch dữ liệu thô.
    """
    def __init__(self, groupby_col, data_frame):
        self.groupby = groupby_col
        self.df = data_frame

    def get_month(self, x):
        if not pd.isnull(x):
            year_month = re.findall(r"\d+", x)
            if year_month:
                months = (int(year_month[0]) * 12) + np.int64(year_month[-1])
                return months
            return x
        else:
            return x

    @staticmethod
    def get_numbers(text):
        digits = re.findall(r'\d+', str(text))
        digits = ','.join(digits)
        return digits

    @staticmethod
    def replace_special_character(text):
        if "NM" in str(text):
            return "No"
        if "payments" in str(text) or "_" not in str(text):
            return text
        clean_text = str(text).replace("_", "")
        return np.nan if clean_text == "nan" else clean_text

    @staticmethod
    def preprocess_text(texts: str) -> tuple[dict, list[str]]:
        dictionary = {}
        tokens = [str(text).lower().replace("and", "").split(",") for text in texts]
        tokens = [[token.strip() for token in token_list if token not in string.punctuation] for token_list in tokens]
        for token_list in tokens:
            for token in token_list:
                if token not in dictionary:
                    size = len(dictionary)
                    dictionary[token] = size
        return (dictionary, ["|".join(words) for words in tokens])

    @staticmethod
    def fill_na(df: pd.DataFrame, groupby=None):
        # Loại bỏ các cột không cần thiết khi tính toán điền khuyết
        cols_to_exclude = ["is_train", "Credit_Score", "Type_of_Loan"]
        cat_features = df.select_dtypes(exclude="number").columns.drop(
            [c for c in cols_to_exclude if c in df.columns], errors='ignore'
        )
        num_features = df.select_dtypes(include="number").columns

        df["Type_of_Loan"] = df["Type_of_Loan"].fillna("not specified")

        def fill_na_cat(d):
            if groupby and groupby in d.columns:
                 d[cat_features] = d.groupby(groupby)[cat_features].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
            else:
                 d[cat_features] = d[cat_features].fillna(d[cat_features].mode().iloc[0])
            return d

        def fill_na_num(d):
            if groupby and groupby in d.columns:
                d[num_features] = d.groupby(groupby)[num_features].transform(
                    lambda x: x.fillna(x.median()))
            else:
                 d[num_features] = d[num_features].fillna(d[num_features].median())
            return d

        df = fill_na_cat(df)
        df = fill_na_num(df)
        return df

    def preprocess(self):
        # Extract numbers from Age
        self.df['Age'] = self.df.Age.apply(DataProcessor.get_numbers)
        
        # Clean special characters
        # Note: applymap is deprecated in pandas 2.0+, using map/apply generally preferred but keeping logic
        self.df = self.df.apply(lambda col: col.map(DataProcessor.replace_special_character))
        
        # Convert to numeric where possible
        self.df = self.df.apply(lambda x: pd.to_numeric(x, errors="ignore"))
        
        # Specific column handling
        if "Credit_Mix" in self.df.columns:
            self.df["Credit_Mix"] = self.df.groupby(self.groupby)["Credit_Mix"].transform(
                lambda x: x.replace("", x.mode()[0] if not x.mode().empty else np.nan))
        
        if "Payment_Behaviour" in self.df.columns:
            self.df["Payment_Behaviour"] = self.df.groupby(self.groupby)["Payment_Behaviour"].transform(
                lambda x: x.replace("!@9#%8" if (not x.mode().empty and x.mode()[0] != "!@9#%8") else np.nan)
            )
            
        # Process Type_of_Loan text
        if "Type_of_Loan" in self.df.columns:
            self.df["Type_of_Loan"] = self.df[["Type_of_Loan"]].apply(
                lambda x: DataProcessor.preprocess_text(x.values)[-1]
            )
            self.df["Type_of_Loan"] = self.df["Type_of_Loan"].str.replace(" ", "_").str.replace("|", " ")

        # Process Credit History Age
        if "Credit_History_Age" in self.df.columns:
            self.df["Credit_History_Age"] = self.df["Credit_History_Age"].apply(lambda x: self.get_month(x))

        # Handle Monthly Balance
        if "Monthly_Balance" in self.df.columns:
            self.df["Monthly_Balance"] = pd.to_numeric(self.df.Monthly_Balance, errors="coerce")

        # Fill NA
        self.df = DataProcessor.fill_na(self.df, self.groupby)

        return self.df

class ClipOutliersTransformer(BaseEstimator, TransformerMixin):
    """
    Class dùng để xử lý ngoại lai (Outliers) bằng phương pháp IQR Clipping.
    """
    def __init__(self, lower_quantile, upper_quantile, multiply_by=1.5, replace_with_median: bool = False):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.multiply_by = multiply_by
        self.replace_with_median = replace_with_median
        self.lower_limit = 0
        self.upper_limit = 0

    def fit(self, X, y=None):
        q1 = np.quantile(X, self.lower_quantile)
        q3 = np.quantile(X, self.upper_quantile)
        iqr = q3 - q1
        self.lower_limit = q1 - (self.multiply_by * iqr)
        self.upper_limit = q3 + (self.multiply_by * iqr)
        return self

    def transform(self, X):
        if self.replace_with_median:
            return np.where(
                ((X >= self.lower_limit) & (X <= self.upper_limit)), X,
                np.median(X))
        else:
            return np.clip(X, self.lower_limit, self.upper_limit)

def get_skewness(df, lower=None, upper=None):
    """Tính độ lệch (skewness) để xác định cột nào cần xử lý ngoại lai."""
    skewness = df.skew(numeric_only=True)
    highly_skewed = skewness[(skewness <= lower) | (skewness >= upper)].index.to_list()
    lowly_skewed = skewness[(skewness > lower) & (skewness < upper)].index.to_list()
    return (highly_skewed, lowly_skewed)

def remove_outliers(df: pd.DataFrame):
    """Hàm wrapper để thực hiện loại bỏ ngoại lai trên toàn bộ DataFrame."""
    numbers = df.select_dtypes(include="number").columns
    
    # Xác định ngưỡng skewness
    highly_skewed, lowly_skewed = get_skewness(df[numbers], lower=-0.8, upper=0.8)

    # Xử lý highly skewed: thay bằng median
    if highly_skewed:
        df[highly_skewed] = df[highly_skewed].apply(
            lambda x: ClipOutliersTransformer(
                0.25, 0.75, multiply_by=1.5, replace_with_median=True
            ).fit_transform(x)
        )

    # Xử lý lowly skewed: clip (cắt ngọn)
    if lowly_skewed:
        df[lowly_skewed] = df[lowly_skewed].apply(
            lambda x: ClipOutliersTransformer(
                0.25, 0.75, multiply_by=1.5, replace_with_median=False
            ).fit_transform(x)
        )
    return df

def get_unique_values(df):
    """Hàm tiện ích để kiểm tra các giá trị unique trong các cột categorical."""
    cat_cols = df.select_dtypes("object").columns
    data_info = []
    
    for col in cat_cols:
        if len(df[col].unique()) > 5000:
            continue
        unique_values, counts = np.unique(np.array(df[col], dtype=str), return_counts=True)
        num_of_uv = len(unique_values)
        unique_val_percent = np.round(counts / counts.sum(), 2)
        data_info.append([col, unique_values.tolist(), counts.tolist(), num_of_uv, unique_val_percent])
            
    return pd.DataFrame(data_info, columns=["column", "unique", "counts", "len_unique_values", "%_unique_values"])

def post_process_cleaning(data: pd.DataFrame):
    """Các bước làm sạch thủ công bổ sung sau khi chạy DataProcessor."""
    if "Num_Bank_Accounts" in data.columns:
        data.loc[data["Num_Bank_Accounts"] < 0, "Num_Bank_Accounts"] = 0
    
    cols_to_fix_nan = ["Type_of_Loan", "Occupation", "Credit_Mix"]
    for col in cols_to_fix_nan:
        if col in data.columns:
             data.loc[data[col] == "nan", col] = np.nan
             data.loc[data[col] == "", col] = np.nan
             
    return data
