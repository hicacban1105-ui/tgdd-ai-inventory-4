import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="TGDĐ - AI Inventory Management", page_icon="📱", layout="wide")

# SỬA LỖI CACHE: Dùng cache_data thay vì cache_resource khi trả về DataFrame
@st.cache_data
def run_ai_pipeline():
    # 1. Nạp và chuẩn hóa
    df = pd.read_csv('TGDĐ_2025_2026_Inventory_Data (2) (1).csv', sep=None, engine='python')
    df.columns = ['date', 'store_id', 'sku_id', 'sales_qty', 'lead_time_days',
                  'current_stock', 'promo_flag', 'forecast_demand', 'stockout_risk']
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date']).sort_values(by='date').reset_index(drop=True)

    # 2. Đặc trưng thời gian & Lag
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['sales_qty_log1p'] = np.log1p(df['sales_qty'])

    for lag in [1, 3, 7]:
        df[f'sales_qty_lag_{lag}'] = df.groupby(['store_id', 'sku_id'])['sales_qty'].shift(lag)

    df = df.dropna().reset_index(drop=True)

    # Ép kiểu int rõ ràng để tránh lỗi XGBoost
    df['store_id_encoded'] = df['store_id'].astype('category').cat.codes.astype(int)
    df['sku_id_encoded'] = df['sku_id'].astype('category').cat.codes.astype(int)
    df['promo_flag'] = df['promo_flag'].astype(int)

    features = ['store_id_encoded', 'sku_id_encoded', 'lead_time_days', 'current_stock', 'promo_flag',
                'month_sin', 'month_cos', 'day_sin', 'day_cos',
                'sales_qty_lag_1', 'sales_qty_lag_3', 'sales_qty_lag_7']

    X = df[features]

    # 3. Chạy thuật toán để lấy kết quả xuất lên Web
    reg_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    reg_model.fit(X, df['sales_qty_log1p'])
    df['AI_Daily_Forecast'] = np.expm1(reg_model.predict(X)).round()

    clf_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss', random_state=42)
    clf_model.fit(X, df['stockout_risk'])
    df['AI_Risk_Warning'] = clf_model.predict(X)

    return df

try:
    df = run_ai_pipeline()
except Exception as e:
    st.error(f"❌ Lỗi tải dữ liệu hoặc huấn luyện AI: {e}")
    st.stop()

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.sidebar.image("/content/Logo-The-Gioi-Di-Dong-MWG.webp", width=50)
st.sidebar.markdown("### ⚙️ BỘ LỌC DỮ LIỆU")

store_list = df['store_id'].unique().tolist()
selected_store = st.sidebar.selectbox("🏠 Chọn Siêu thị", ["Tất cả"] + store_list)

sku_list = df['sku_id'].unique().tolist()
selected_sku = st.sidebar.selectbox("📦 Chọn Mã sản phẩm", ["Tất cả"] + sku_list)

filtered_df = df.copy()
if selected_store != "Tất cả":
    filtered_df = filtered_df[filtered_df['store_id'] == selected_store]
if selected_sku != "Tất cả":
    filtered_df = filtered_df[filtered_df['sku_id'] == selected_sku]

st.title("📊 Hệ Thống AI Dự Báo & Tối Ưu Tồn Kho TGDĐ")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
total_sales = filtered_df['sales_qty'].sum()
avg_lead_time = filtered_df['lead_time_days'].mean()
total_stockouts = filtered_df[filtered_df['AI_Risk_Warning'] == 1].shape[0]
avg_stock = filtered_df['current_stock'].mean()

col1.metric("Tổng lượng bán", f"{total_sales:,.0f} máy")
col2.metric("Tồn kho trung bình", f"{avg_stock:,.0f} máy")
col3.metric("Lead Time trung bình", f"{avg_lead_time:.1f} ngày")
col4.metric("🚨 Cảnh báo thiếu hụt hàng hoá", f"{total_stockouts} lượt")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📉 Tổng quan Dự báo", "🚨 Cảnh báo Rủi ro", "📦 Khuyến nghị Nhập hàng"])

with tab1:
    st.subheader("Biểu đồ So sánh Nhu cầu: Thực tế vs AI Dự báo")
    daily_trend = filtered_df.groupby('date')[['sales_qty', 'AI_Daily_Forecast']].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_trend['date'], daily_trend['sales_qty'], label='Thực tế bán ra', color='blue', marker='o', alpha=0.6)
    ax.plot(daily_trend['date'], daily_trend['AI_Daily_Forecast'], label='AI Dự báo (LightGBM)', color='orange', linestyle='--', linewidth=2)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

with tab2:
    st.subheader("Danh sách SKU Rủi ro Cháy Hàng")
    risk_df = filtered_df[filtered_df['AI_Risk_Warning'] == 1]
    if risk_df.empty:
        st.success("✅ Hiện tại không phát hiện rủi ro đứt gãy cung ứng.")
    else:
        st.error(f"⚠️ Phát hiện {len(risk_df)} điểm nghẽn cung ứng!")
        st.dataframe(risk_df[['date', 'store_id', 'sku_id', 'sales_qty', 'current_stock', 'lead_time_days']].sort_values(by='date', ascending=False), use_container_width=True)

with tab3:
    st.subheader("Hệ thống Tự động Đề xuất Nhập hàng")
    suggest_df = filtered_df.groupby(['store_id', 'sku_id']).agg({
        'current_stock': 'last',
        'AI_Daily_Forecast': 'mean',
        'lead_time_days': 'mean'
    }).reset_index()

    suggest_df['forecast_demand'] = np.ceil(suggest_df['AI_Daily_Forecast'] * suggest_df['lead_time_days'])
    suggest_df['Safety_Stock'] = np.ceil(suggest_df['forecast_demand'] * 0.5)
    suggest_df['Reorder_Qty'] = np.maximum(0, suggest_df['forecast_demand'] + suggest_df['Safety_Stock'] - suggest_df['current_stock'])
    action_df = suggest_df[suggest_df['Reorder_Qty'] > 0].sort_values(by='Reorder_Qty', ascending=False)

    if action_df.empty:
        st.success("Tất cả hàng hóa đang ở mức an toàn.")
    else:
        st.dataframe(action_df[['store_id', 'sku_id', 'forecast_demand', 'Safety_Stock', 'current_stock', 'Reorder_Qty']].style.highlight_max(subset=['Reorder_Qty'], color='lightcoral'), use_container_width=True)
        # Đọc nội dung file app.py hiện tại để sửa triệt để Tab 3
with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

# Đoạn xử lý hiển thị Tab 3 cũ
old_tab3_code = """    suggest_df['forecast_demand'] = np.ceil(suggest_df['AI_Daily_Forecast'] * suggest_df['lead_time_days'])
    suggest_df['Safety_Stock'] = np.ceil(suggest_df['forecast_demand'] * 0.5)
    suggest_df['Reorder_Qty'] = np.maximum(0, suggest_df['forecast_demand'] + suggest_df['Safety_Stock'] - suggest_df['current_stock'])
    action_df = suggest_df[suggest_df['Reorder_Qty'] > 0].sort_values(by='Reorder_Qty', ascending=False)"""

# Đoạn xử lý mới ép toàn bộ các cột tính toán về số nguyên integer
new_tab3_code = """    suggest_df['forecast_demand'] = np.ceil(suggest_df['AI_Daily_Forecast'] * suggest_df['lead_time_days']).astype(int)
    suggest_df['Safety_Stock'] = np.ceil(suggest_df['forecast_demand'] * 0.5).astype(int)
    suggest_df['Reorder_Qty'] = np.maximum(0, suggest_df['forecast_demand'] + suggest_df['Safety_Stock'] - suggest_df['current_stock']).astype(int)
    action_df = suggest_df[suggest_df['Reorder_Qty'] > 0].sort_values(by='Reorder_Qty', ascending=False)"""

if old_tab3_code in code:
    code = code.replace(old_tab3_code, new_tab3_code)
else:
    # Đoạn phòng hờ nếu cấu trúc chữ viết hơi khác bản cũ của bạn
    code = code.replace("st.dataframe(action_df", """action_df['forecast_demand'] = action_df['forecast_demand'].round(0).astype(int)
        action_df['Safety_Stock'] = action_df['Safety_Stock'].round(0).astype(int)
        action_df['Reorder_Qty'] = action_df['Reorder_Qty'].round(0).astype(int)
        st.dataframe(action_df""")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(code)

print("✅ Đã ép kiểu số nguyên thành công cho toàn bộ bảng Khuyến nghị nhập hàng!")
