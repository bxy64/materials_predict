import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# 设置宽屏模式
st.set_page_config(layout="wide")

# 读取数据
file_path = "./sample.csv"
data = pd.read_csv(file_path)

# 数据预处理
# num_features = data.shape[1] - 1
X = data.drop(columns=['Adsorption capacity'])
y = data['Adsorption capacity']
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# **增强 Contact time 影响**
X['Contact time^2'] = X['Contact time'] ** 2
X['Contact time^3'] = X['Contact time'] ** 3
X['Contact_Dosage'] = X['Contact time'] * X['Dosage']  # 交互特征

# **特征权重**
feature_weights = {
    'Contact time': 2.0,  # 提高权重
    'Dosage': 1.5,  # 让剂量影响更大
}
for feature, weight in feature_weights.items():
    if feature in X.columns:
        X[feature] *= weight

# **标准化数据**
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# 训练XGBoost模型
best_params = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'colsample_bytree': 1.0,
    'gamma': 0.5,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'scale_pos_weight': 2,  # 增强剂量对吸附容量的影响
}

xgb_regressor = xgb.XGBRegressor(**best_params)
xgb_regressor.fit(X_std, y)

# **预测函数**
def predict_external_data(external_data):
    external_data = external_data.reindex(columns=scaler.feature_names_in_, fill_value=0)
    external_data_std = scaler.transform(external_data)
    return xgb_regressor.predict(external_data_std)

# **Streamlit 交互界面**
st.title('Adsorption Capacity Prediction')
col1, col2, col3 = st.columns(3)


# 创建与模型输入格式匹配的 DataFrame（X_input）
def create_input_dataframe(feature_values):
    """
    创建输入数据框，保证与模型训练时的特征匹配
    :param feature_values: 字典，包含所有特征名称及其对应的值
    :return: DataFrame 格式的输入数据
    """
    X_input = pd.DataFrame([feature_values])
    return X_input


def predict_external_data(external_data):
    print("🚀 开始预测...")

    # 重新排序，使其特征顺序与训练时一致
    external_data = external_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    print("✅ 输入数据预处理完成，形状:", external_data.shape)

    # 尝试标准化数据
    try:
        external_data_std = scaler.transform(external_data)
        print("✅ 标准化完成")
    except Exception as e:
        print(f"❌ Scaler transform 出错: {e}")
        return None  # 返回 None 避免后续错误

    # 尝试进行预测
    try:
        predictions = xgb_regressor.predict(external_data_std)
        print("✅ 预测完成，结果:", predictions)
    except Exception as e:
        print(f"❌ XGBoost 预测出错: {e}")
        return None  # 返回 None 避免后续错误

    return predictions


# 选项列表
heavy_metal_ions_choices = ['As(Ⅲ)',
'As(Ⅴ)',
'Cd(Ⅱ)',
'Co(Ⅱ)',
'Cr(Ⅲ)',
'Cr(Ⅵ)',
'Cu(Ⅱ)',
'Fe(Ⅱ)',
'Hg(Ⅱ)',
'Mn(Ⅱ)',
'Ni(Ⅱ)',
'Pb(Ⅱ)',
'Sb(Ⅲ)',
'Sb(Ⅴ)',
'Zn(Ⅱ)'
                            ]
modified_choices = ['Chemical modification', 'Composite modification', 'Magnetic properties', 'Nano modification',
                    'Others']
substrate_choices = ['Fe-Mn binary compound', 'MnO₂', 'MnOₓ']

# 主界面
st.title('Adsorption Capacity Prediction')

# 创建三列
col1, col2, col3 = st.columns(3)

# 第一列（红色标题）
with col1:
    st.markdown("<h3 style='color: #FF4B4B;'>Mn-Based Nanomaterials Properties</h3>", unsafe_allow_html=True)
    specific_surface = st.number_input("Specific surface area (m²/g)", min_value=0.0, value=100.0, key='surf_area')
    pore_volume = st.number_input("Pore volume (cm³/g)", min_value=0.0, value=0.2, step=0.01, key='pore_vol')
    avg_pore = st.number_input("Average pore (nm)", min_value=0.0, value=100.0, key='avg_pore')
    substrate = st.selectbox('Substrate', substrate_choices)
    modified = st.selectbox('Modified', modified_choices)
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)  # 添加空白占位符


# 第二列（绿色标题）
with col2:
    st.markdown("<h3 style='color: #00C853;'>Adsorption conditions</h3>", unsafe_allow_html=True)
    dosage = st.number_input("Dosage (g/L)", min_value=0.0, value=0.5, step=0.1, key='dosage')
    initial_concentration = st.number_input("Initial concentration. (mg/L)", min_value=0.0, value=10.0, key='init_conc')
    temperature = st.number_input("Temperature (°C)", value=25, key='temp')
    contact_time = st.number_input("Contact time (h)", min_value=0.00, value=30.00, step=0.01, format="%.2f", key='contact_time')
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.0, step=0.1, key='ph')

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)  # 添加空白占位符


# 第三列（蓝色标题）
with col3:
    st.markdown("<h3 style='color: #2979FF;'>Heavy metal ions properties</h3>", unsafe_allow_html=True)
    heavy_metal_ions = st.selectbox('Heavy metal ions', heavy_metal_ions_choices)
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)  # 添加空白占位符





# 预测按钮
submitted = st.button("Predict", type="primary")


if submitted:
    # 确保 contact_time 为 float
    contact_time = float(contact_time)
    
    # 生成one-hot特征
    heavy_metal_values = {f'Heavy metal ions_{v}': 0 for v in heavy_metal_ions_choices}
    modified_values = {f'Modified_{v}': 0 for v in modified_choices}
    substrate_values = {f'Substrate_{v}': 0 for v in substrate_choices}

    heavy_metal_values[f'Heavy metal ions_{heavy_metal_ions}'] = 1
    modified_values[f'Modified_{modified}'] = 1
    substrate_values[f'Substrate_{substrate}'] = 1

    # 组合特征值
    feature_values = {
        'Specific surface area': specific_surface,
        'Pore volume': pore_volume,
        'Average pore': avg_pore,
        'Dosage': dosage,
        'Initial concentration': initial_concentration,
        'Temperature': temperature,
        'Contact time': contact_time * 2.0,  # 应用权重
        'Contact time^2': contact_time ** 2,
        'Contact time^3': contact_time ** 3,
        'Contact_Dosage': contact_time * dosage,
        'pH': ph,
    }
    feature_values.update(heavy_metal_values)
    feature_values.update(modified_values)
    feature_values.update(substrate_values)
    # st.write(feature_values)
    # 创建输入数据并预测
    X_input = create_input_dataframe(feature_values)
    try:
        # 调用预测函数
        y_pred = predict_external_data(X_input)

        # 只有 y_pred 不是 None 时才打印
        if y_pred is not None:
            print(f"Predicted Adsorption Capacity: {y_pred[0]:.4f} mg/g")
        else:
            print("❌ 预测失败，请检查错误信息！")

        st.success(f"Predicted Adsorption Capacity: **{y_pred[0]:.2f} mg/g**")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
