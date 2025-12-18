import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Thiết lập style chung cho các biểu đồ
plt.style.use('ggplot')
import matplotlib
matplotlib.rc(("xtick", "ytick", "text"), c="k")
matplotlib.rc("figure", dpi=80)

def plot_numerical_boxplots(df: pd.DataFrame, target_col="Credit_Score"):
    """
    Vẽ Boxplot cho tất cả các cột số để xem phân phối theo biến mục tiêu.
    """
    numb_columns = df.select_dtypes(include="number").columns
    
    # Tính toán số lượng hàng/cột cho subplot
    n_cols = 4
    n_rows = (len(numb_columns) + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(18, 4 * n_rows), dpi=300)
    
    for i, column in enumerate(numb_columns):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.boxplot(x=target_col, y=column, data=df, ax=ax, width=0.8, palette="Set2")
        plt.title(column, fontdict={"fontsize": 10})
        plt.xticks(rotation=0)
        plt.tight_layout(pad=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_categorical_counts(df: pd.DataFrame, target_col="Credit_Score"):
    """
    Vẽ biểu đồ Countplot cho các biến phân loại.
    """
    # Lọc các cột categorical, loại trừ các cột không cần thiết
    exclude_cols = [target_col, 'is_train', 'Customer_ID', "Type_of_Loan", "Month", "Name", "SSN", "ID"]
    cat_cols = df.select_dtypes(exclude="number").columns.drop(
        [c for c in exclude_cols if c in df.columns], errors='ignore'
    ).tolist()
    
    # Đưa Payment_Behaviour lên vị trí hiển thị riêng nếu có
    if "Payment_Behaviour" in cat_cols:
        cat_cols.remove("Payment_Behaviour")
        has_payment = True
    else:
        has_payment = False

    fig, axes = plt.subplots(figsize=(12, 6), dpi=300)
    fig.suptitle("Counts of categorical columns")
    axes.grid(visible=False)
    axes.xaxis.set_tick_params(labelbottom=False)
    axes.yaxis.set_tick_params(labelleft=False)

    def __plot_graph(data, col, ax, legend=False):
        sns.countplot(data=data, x=col, ax=ax, hue=target_col)
        ax.set_xlabel(col, fontdict={"size": 9})
        ax.set_title(f"by {col}", fontdict={"size": 9})
        ax.tick_params(labelsize=7, axis="y")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontdict=dict(size=7))
        ax.grid(False)
        if legend:
            ax.legend(shadow=True, loc="best", facecolor="inherit", frameon=True)
        else:
            if ax.legend_: ax.legend_.remove()

    # Vẽ grid các cột thường
    for i, col in enumerate(cat_cols, 1):
        # Giới hạn số lượng subplot nếu cần (logic notebook cũ là 2x3)
        if i > 6: break 
        ax = fig.add_subplot(2, 3, i)
        __plot_graph(df, col=col, ax=ax)

    # Vẽ riêng Payment_Behaviour nếu có
    if has_payment:
        ax2 = fig.add_axes((0.74, 0.527, 0.23, 0.35))
        __plot_graph(df, col="Payment_Behaviour", ax=ax2, legend=True)
    
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Vẽ Heatmap thể hiện tương quan giữa các biến số.
    """
    df_num = df.select_dtypes(include="number")
    if "is_train" in df_num.columns:
        df_num = df_num.drop(["is_train"], axis=1)
        
    corr = df_num.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig = plt.figure(figsize=(10, 6), dpi=150)
    sns.heatmap(corr, annot=True, mask=mask, fmt=".0%", annot_kws={"size":10})
    plt.grid(False)
    plt.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    plt.title("Correlation Matrix")
    plt.show()

def plot_complex_monthly_balance(df: pd.DataFrame):
    """
    Vẽ biểu đồ Pie Chart lồng nhau thể hiện phân phối Monthly_Balance 
    theo Credit Score và Credit Mix.
    """
    # Logic xử lý dữ liệu trước khi vẽ (như trong notebook)
    df_copy = df.copy()
    if "Monthly_Balance" in df_copy.columns:
         # Xử lý outlier tạm thời cho biểu đồ này
         q75 = df_copy["Monthly_Balance"].quantile(0.75)
         median = df_copy["Monthly_Balance"].median()
         df_copy["Monthly_Balance"] = np.where(
             (df_copy["Monthly_Balance"] > q75) | (df_copy["Monthly_Balance"] < q75), 
             median, df_copy["Monthly_Balance"]
         )

    cross_tab = pd.crosstab(
        values=df_copy["Monthly_Balance"], 
        index=[df_copy["Credit_Score"], df_copy["Credit_Mix"]], 
        columns="Monthly_Balance", 
        aggfunc="mean"
    ).reset_index()

    main_group = pd.pivot_table(cross_tab, "Monthly_Balance", "Credit_Score", aggfunc=np.mean)
    
    # Màu sắc
    b = plt.cm.Blues
    a = plt.cm.Accent

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle("Distribution of Monthly_Balance by Credit Score & Credit Mix", fontsize=11, color="k")
    
    # Vòng tròn ngoài (Credit Score)
    pie1, *_, texts = ax.pie(x=main_group["Monthly_Balance"], labels=main_group.index,
                             autopct="%.1f%%", radius=1.3,
                             colors=[a(80, 1), b(100, 1), a(0, 1)],
                             pctdistance=0.8, textprops={"size": 9}, frame=True)
    plt.setp(pie1, width=0.5)

    # Vòng tròn trong (Credit Mix)
    pie2, *_, texts = ax.pie(x=cross_tab["Monthly_Balance"], autopct="%.0f%%", radius=0.8,
                             colors=[a(80, 0.9), a(80, 0.8), a(80, 0.7),
                                     b(100, 0.9), b(100, 0.8), b(100, 0.7),
                                     a(0, 0.8), a(0, 0.65), a(0, 0.5)],
                             textprops={"size": 8})
    plt.setp(pie2, width=0.5)
    
    # Legend
    legend_labels = np.unique(cross_tab["Credit_Mix"].astype(str))
    if len(legend_labels) >= 3:
        legend_handles = [
            plt.plot([], label=legend_labels[0], c="k")[0],
            plt.plot([], label=legend_labels[1], c='b')[0],
            plt.plot([], label=legend_labels[-1], c="g")[0]
        ]
        plt.legend(handles=legend_handles, shadow=True, frameon=True, facecolor="inherit",
                   loc="best", title="Credit Mix", bbox_to_anchor=(1, 1, 0.5, 0.1))
    
    plt.show()

def plot_chi2_results(chi2_summary_df):
    """Vẽ kết quả kiểm định Chi-square/ANOVA."""
    if chi2_summary_df is empty: return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=chi2_summary_df, y="column", x="t-statistic", ax=ax)
    plt.setp([ax.get_xticklabels(), ax.get_yticklabels()], size=8)
    plt.title("Feature Significance (Chi2/ANOVA)")
    plt.show()

def plot_feature_importance(importance_df):
    """Vẽ biểu đồ mức độ quan trọng của các đặc trưng."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, y="Feature Id", x="Importances")
    plt.title("Feature Importance in CatBoost")
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, classes):
    """Vẽ Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    
    annot = np.array([f"{i}\n({g:.1%})" for i, g in zip(cm.flatten(), cm_norm.flatten())])
    annot = annot.reshape(cm.shape)

    fig = plt.figure(dpi=90)
    sns.heatmap(cm, annot=annot, fmt="", xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
