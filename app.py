import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from preprocessing import DataProcessor, post_process_cleaning
from feature_engineering import build_transformer_pipeline # Cần để load object pickle
from preprocessing import ClipOutliersTransformer # Cần để load object pickle

# Cấu hình trang
st.set_page_config(
    page_title="Credit Score Prediction AI",
    page_icon="💳",
    layout="wide"
)

# --- 1. HÀM LOAD RESOURCES (Model & Transformers) ---
@st.cache_resource
def load_resources():
    # Load Model
    model = CatBoostClassifier()
    try:
        model.load_model("credit_score_model_refined.cbm")
    except:
        # Fallback nếu chưa chạy refinement
        model.load_model("credit_score_model_v1.cbm")
        
    # Load Transformers
    with open("OutlierRemover.pkl", "rb") as f:
        outlier_remover = pickle.load(f)
        
    with open("ColumnsTransformers.pkl", "rb") as f:
        column_transformer = pickle.load(f)
        
    return model, outlier_remover, column_transformer

try:
    model, outlier_remover, column_transformer = load_resources()
    st.success("✅ Hệ thống đã sẵn sàng!")
except Exception as e:
    st.error(f"❌ Lỗi khi tải model: {e}")
    st.warning("Hãy chắc chắn bạn đã chạy file `main.py` để tạo ra các file model và pickle trước.")
    st.stop()

# --- 2. GIAO DIỆN NHẬP LIỆU (SIDEBAR) ---
st.sidebar.header("📝 Nhập thông tin khách hàng")

def user_input_features():
    # Nhóm thông tin cá nhân
    st.sidebar.subheader("1. Thông tin cá nhân")
    age = st.sidebar.number_input("Tuổi (Age)", 18, 100, 30)
    occupation = st.sidebar.selectbox("Nghề nghiệp", 
        ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 
         'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 
         'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'])
    annual_income = st.sidebar.number_input("Thu nhập hàng năm ($)", 0.0, 1000000.0, 50000.0)
    monthly_salary = st.sidebar.number_input("Lương thực nhận hàng tháng ($)", 0.0, 100000.0, 4000.0)
    
    # Nhóm thông tin ngân hàng & thẻ
    st.sidebar.subheader("2. Ngân hàng & Tín dụng")
    num_bank_accounts = st.sidebar.number_input("Số tài khoản ngân hàng", 0, 50, 2)
    num_credit_card = st.sidebar.number_input("Số lượng thẻ tín dụng", 0, 50, 4)
    interest_rate = st.sidebar.number_input("Lãi suất thẻ tín dụng (%)", 0, 100, 15)
    num_loan = st.sidebar.number_input("Số khoản vay hiện tại", 0, 50, 1)
    credit_utilization_ratio = st.sidebar.number_input("Tỷ lệ sử dụng tín dụng (%)", 0.0, 100.0, 30.0)
    
    # Nhóm lịch sử thanh toán
    st.sidebar.subheader("3. Lịch sử tài chính")
    delay_from_due_date = st.sidebar.number_input("Số ngày trễ hạn trung bình", 0, 100, 5)
    num_delayed_payment = st.sidebar.number_input("Số lần thanh toán chậm", 0, 100, 3)
    changed_credit_limit = st.sidebar.number_input("Thay đổi hạn mức tín dụng", -100.0, 100.0, 0.0)
    num_credit_inquiries = st.sidebar.number_input("Số lần tra cứu tín dụng (Credit Inquiries)", 0, 50, 4)
    outstanding_debt = st.sidebar.number_input("Dư nợ hiện tại ($)", 0.0, 100000.0, 1000.0)
    
    # Nhóm hành vi & Khác
    st.sidebar.subheader("4. Khác")
    credit_mix = st.sidebar.selectbox("Credit Mix", ['Standard', 'Good', 'Bad'])
    credit_history_age = st.sidebar.text_input("Tuổi lịch sử tín dụng (VD: 20 Years and 3 Months)", "10 Years and 5 Months")
    payment_of_min_amount = st.sidebar.selectbox("Chỉ thanh toán tối thiểu?", ['No', 'Yes', 'NM'])
    total_emi = st.sidebar.number_input("Tiền trả góp hàng tháng (EMI)", 0.0, 10000.0, 500.0)
    amount_invested = st.sidebar.number_input("Số tiền đầu tư hàng tháng", 0.0, 10000.0, 200.0)
    payment_behaviour = st.sidebar.selectbox("Hành vi thanh toán", 
        ['High_spent_Small_value_payments', 'Low_spent_Large_value_payments',
         'Low_spent_Medium_value_payments', 'Low_spent_Small_value_payments',
         'High_spent_Medium_value_payments', 'High_spent_Large_value_payments'])
    monthly_balance = st.sidebar.number_input("Số dư hàng tháng còn lại ($)", 0.0, 10000.0, 300.0)
    
    # Dữ liệu dạng text cần xử lý
    type_of_loan = st.sidebar.text_area("Các loại khoản vay (phân cách bằng dấu phẩy)", "Home Loan, Auto Loan")

    # Tạo DataFrame từ input
    data = {
        'Age': str(age), 
        'Occupation': occupation,
        'Annual_Income': annual_income,
        'Monthly_Inhand_Salary': monthly_salary,
        'Num_Bank_Accounts': num_bank_accounts,
        'Num_Credit_Card': num_credit_card,
        'Interest_Rate': interest_rate,
        'Num_of_Loan': num_loan,
        'Type_of_Loan': type_of_loan,
        'Delay_from_due_date': delay_from_due_date,
        'Num_of_Delayed_Payment': num_delayed_payment,
        'Changed_Credit_Limit': changed_credit_limit,
        'Num_Credit_Inquiries': num_credit_inquiries,
        'Credit_Mix': credit_mix,
        'Outstanding_Debt': outstanding_debt,
        'Credit_Utilization_Ratio': credit_utilization_ratio, 
        'Credit_History_Age': credit_history_age,
        'Payment_of_Min_Amount': payment_of_min_amount,
        'Total_EMI_per_month': total_emi,
        'Amount_invested_monthly': amount_invested,
        'Payment_Behaviour': payment_behaviour,
        'Monthly_Balance': monthly_balance,
        
        # Các cột giả lập (Dummy)
        'Month': 'January',
        'Customer_ID': 'CUS_0000',
        'Name': 'User',
        'SSN': '000',
        'ID': '000',
        'is_train': False 
    }
    
    return pd.DataFrame([data])

# --- 3. LOGIC DỰ ĐOÁN ---
st.title("💳 Credit Score Prediction App")
st.write("Ứng dụng dự đoán điểm tín dụng sử dụng Machine Learning (CatBoost).")

input_df = user_input_features()

# Hiển thị dữ liệu người dùng nhập (Raw)
with st.expander("Xem dữ liệu đầu vào thô"):
    st.dataframe(input_df)

if st.button("🚀 Dự đoán Credit Score", type="primary"):
    with st.spinner("Đang xử lý dữ liệu và dự đoán..."):
        try:
            # BƯỚC 1: Tiền xử lý (Cleaning)
            processor = DataProcessor("Customer_ID", input_df)
            df_clean = processor.preprocess()
            df_clean = post_process_cleaning(df_clean)
            
            # Loại bỏ các cột không dùng cho training
            cols_to_drop = ["Month", "Customer_ID", "Name", "SSN", "is_train", "ID"]
            df_clean_for_transform = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], errors='ignore')

            # BƯỚC 2: Xử lý Outlier (Transform)
            num_cols = df_clean_for_transform.select_dtypes(include="number").columns
            
            # Chỉ transform trên các cột số
            df_outlier = df_clean_for_transform.copy()
            df_outlier[num_cols] = outlier_remover.transform(df_outlier[num_cols])

            # BƯỚC 3: Column Transformer (Impute + Scale)
            # Pipeline trả về numpy array
            X_processed_array = column_transformer.transform(df_outlier)
            
            # Tái tạo tên cột để đưa vào CatBoost
            # Logic này phải khớp 100% với file feature_engineering.py
            feature_names = []
            
            # Lấy features từ categorical pipeline
            # Vì SimpleImputer giữ nguyên cột, ta lấy tên cột categorical từ df đầu vào của pipeline
            cat_cols_input = df_outlier.select_dtypes(exclude="number").columns.tolist()
            feature_names.extend(cat_cols_input)
            
            # Lấy features từ numerical pipeline
            num_cols_input = df_outlier.select_dtypes(include="number").columns.tolist()
            feature_names.extend(num_cols_input)
            
            X_final = pd.DataFrame(X_processed_array, columns=feature_names)
            
            # Convert lại sang numeric
            X_final = X_final.apply(pd.to_numeric, errors="ignore")
            
            # Đảm bảo đúng thứ tự cột nếu cần
            
            # BƯỚC 4: Dự đoán
            prediction = model.predict(X_final)
            proba = model.predict_proba(X_final)
            
            result = prediction[0][0] # Kết quả là array lồng nhau
            
            # Hiển thị kết quả
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if result == "Good":
                    st.success(f"### Kết quả: {result}")
                    st.balloons()
                elif result == "Standard":
                    st.warning(f"### Kết quả: {result}")
                else:
                    st.error(f"### Kết quả: {result}")
            
            with col2:
                st.write("#### Xác suất dự đoán:")
                proba_df = pd.DataFrame(proba, columns=model.classes_)
                st.bar_chart(proba_df.T)
                
        except Exception as e:
            st.error(f"Có lỗi xảy ra trong quá trình xử lý: {e}")
            st.write("Chi tiết lỗi:", e)

# Footer
st.markdown("---")
st.caption("Developed with Streamlit & CatBoost")
