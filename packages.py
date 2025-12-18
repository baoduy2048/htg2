import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
import pickle
import itertools
import os

# Thư viện mô hình CatBoost
from catboost import CatBoostClassifier, Pool, MetricVisualizer

# Các thư viện tiền xử lý và đánh giá từ Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif

# Thư viện thống kê
from scipy.stats.contingency import chi2_contingency

# Thiết lập mặc định
pd.set_option('display.max_columns', 30)
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Thiết lập vẽ đồ thị
import matplotlib
matplotlib.rc(("xtick", "ytick", "text"), c="k")
matplotlib.rc("figure", dpi=80)
