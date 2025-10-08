import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pymc as pm
import arviz as az
from pygam import LinearGAM, s
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="広告宣伝費最適化ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #3498db;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #f39c12;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# セッションステートの初期化
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None
if 'all_channel_results' not in st.session_state:
    st.session_state.all_channel_results = []
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'optimization_models' not in st.session_state:
    st.session_state.optimization_models = {}

# ========================================
# ヘルパー関数群（重複なし・1回のみ定義）
# ========================================

def train_hill_model(x_data, y_data):
    """Hillモデルの学習"""
    with pm.Model() as model:
        slope = pm.HalfNormal('slope', sigma=1)
        EC50 = pm.HalfNormal('EC50', sigma=np.median(x_data[x_data > 0]) if np.any(x_data > 0) else 10000)
        Vmax = pm.Deterministic('Vmax', slope * EC50)
        mu = Vmax * x_data / (EC50 + x_data)
        sigma = pm.HalfNormal('sigma', sigma=y_data.std())
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)
        trace = pm.sample(10000, tune=1000, chains=4, cores=4, target_accept=0.95, progressbar=False, random_seed=42)
    return trace

def train_linear_model(x_data, y_data):
    """線形回帰モデルの学習"""
    with pm.Model() as linear_model:
        alpha = pm.Normal('alpha', mu=y_data.mean(), sigma=y_data.std() * 2)
        beta = pm.Normal('beta', mu=0, sigma=1)
        mu = alpha + beta * x_data
        sigma = pm.HalfNormal('sigma', sigma=y_data.std())
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)
        trace = pm.sample(10000, tune=1000, chains=4, cores=4, target_accept=0.95, progressbar=False, random_seed=42)
    return trace

def train_gam_model(X_data, y_data):
    """GAMモデルの学習"""
    gam = LinearGAM(s(0, constraints='monotonic_inc')).fit(X_data, y_data)
    return gam

def plot_hill_curve(x_data, y_data, trace, dates, channel_name, target_var):
    """Hill曲線の可視化"""
    vmax_mean = trace.posterior['Vmax'].mean().item()
    ec50_mean = trace.posterior['EC50'].mean().item()
    y_pred = vmax_mean * x_data / (ec50_mean + x_data)
    r2 = r2_score(y_data, y_pred)
    
    x_range = np.linspace(0, x_data.max() * 1.1, 100)
    post_curves = trace.posterior['Vmax'].values.flatten()[:, None] * x_range / (trace.posterior['EC50'].values.flatten()[:, None] + x_range)
    y_mean = post_curves.mean(axis=0)
    hdi_data = az.hdi(post_curves, hdi_prob=0.95)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([hdi_data[:, 1], hdi_data[:, 0][::-1]]),
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% 信用区間',
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_range, y=y_mean,
        mode='lines',
        line=dict(color='rgba(0,176,246,0.8)', width=3),
        name='Hill関数曲線'
    ))
    
    latest_date = pd.to_datetime(dates).max()
    four_weeks_ago = latest_date - pd.Timedelta(days=27)
    recent_mask = pd.to_datetime(dates) > four_weeks_ago
    past_mask = ~recent_mask
    
    if past_mask.any():
        fig.add_trace(go.Scatter(
            x=x_data[past_mask], y=y_data[past_mask],
            mode='markers',
            marker=dict(size=10, color='#636EFA', opacity=0.7),
            name='過去の実績'
        ))
    
    if recent_mask.any():
        fig.add_trace(go.Scatter(
            x=x_data[recent_mask], y=y_data[recent_mask],
            mode='markers',
            marker=dict(symbol='star', size=15, color='gold', line=dict(width=1, color='black')),
            name='直近4週間の実績'
        ))
    
    fig.update_layout(
        title=f"{channel_name}: 広告費と{target_var}の関係",
        xaxis_title='広告宣伝費',
        yaxis_title=target_var,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    fig.add_annotation(
        x=0.98, y=0.05,
        xref='paper', yref='paper',
        text=f'<b>R²: {r2:.3f}</b>',
        showarrow=False,
        font=dict(size=14),
        bgcolor='rgba(255, 255, 255, 0.7)',
        bordercolor='#3498db',
        borderwidth=1
    )
    
    return fig, r2

def plot_linear_curve(x_data, y_data, trace, dates, channel_name, target_var):
    """線形回帰曲線の可視化"""
    alpha_mean = trace.posterior['alpha'].mean().item()
    beta_mean = trace.posterior['beta'].mean().item()
    y_pred = alpha_mean + beta_mean * x_data
    r2 = r2_score(y_data, y_pred)
    
    x_range = np.linspace(0, x_data.max() * 1.1, 100)
    alpha_post = trace.posterior['alpha'].values.flatten()
    beta_post = trace.posterior['beta'].values.flatten()
    post_curves = alpha_post[:, None] + beta_post[:, None] * x_range
    y_mean = post_curves.mean(axis=0)
    hdi_data = az.hdi(post_curves, hdi_prob=0.95)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([hdi_data[:, 1], hdi_data[:, 0][::-1]]),
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% 信用区間',
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_range, y=y_mean,
        mode='lines',
        line=dict(color='rgba(0,176,246,0.8)', width=3),
        name='線形回帰直線'
    ))
    
    latest_date = pd.to_datetime(dates).max()
    four_weeks_ago = latest_date - pd.Timedelta(days=27)
    recent_mask = pd.to_datetime(dates) > four_weeks_ago
    past_mask = ~recent_mask
    
    if past_mask.any():
        fig.add_trace(go.Scatter(
            x=x_data[past_mask], y=y_data[past_mask],
            mode='markers',
            marker=dict(size=10, color='#636EFA', opacity=0.7),
            name='過去の実績'
        ))
    
    if recent_mask.any():
        fig.add_trace(go.Scatter(
            x=x_data[recent_mask], y=y_data[recent_mask],
            mode='markers',
            marker=dict(symbol='star', size=15, color='gold', line=dict(width=1, color='black')),
            name='直近4週間の実績'
        ))
    
    fig.update_layout(
        title=f"{channel_name}: 広告費と{target_var}の関係",
        xaxis_title='広告宣伝費',
        yaxis_title=target_var,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    fig.add_annotation(
        x=0.98, y=0.05,
        xref='paper', yref='paper',
        text=f'<b>R²: {r2:.3f}</b>',
        showarrow=False,
        font=dict(size=14),
        bgcolor='rgba(255, 255, 255, 0.7)',
        bordercolor='#3498db',
        borderwidth=1
    )
    
    return fig, r2

def plot_combined_hill_curves(all_results, x_range):
    """全媒体のHill曲線を統合表示"""
    fig = go.Figure()
    
    colors = ['#9b59b6', '#3498db', '#2ecc71', '#e67e22', '#34495e', '#5dade2', '#1abc9c', '#e74c3c']
    
    for i, result in enumerate(all_results):
        channel = result['channel']
        vmax = result['vmax_mean']
        ec50 = result['ec50_mean']
        
        y_pred = vmax * x_range / (ec50 + x_range)
        
        fig.add_trace(go.Scatter(
            x=x_range, y=y_pred,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=3),
            name=channel
        ))
    
    fig.update_layout(
        title="全媒体比較(Hill関数曲線)",
        xaxis_title='広告宣伝費',
        yaxis_title='予測応募数',
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    return fig

def analyze_channel(df, channel_name, config):
    """媒体分析の実行（現状把握用）"""
    start_date = pd.to_datetime(config['start_date'])
    end_date = pd.to_datetime(config['end_date'])
    target_var = config['target_variable']
    model_type = config['model_type']
    
    filtered_df = df[(df['channel'] == channel_name) & 
                     (df['week_start_date'] >= start_date) & 
                     (df['week_start_date'] <= end_date)].copy()
    
    if len(filtered_df) < 5:
        return None, None, None
    
    x_data = filtered_df['total_spend'].values
    y_data = filtered_df[target_var].values
    dates = filtered_df['week_start_date'].values
    
    try:
        if model_type == 'hill':
            trace = train_hill_model(x_data, y_data)
            fig, r2 = plot_hill_curve(x_data, y_data, trace, dates, channel_name, target_var)
            result = {
                'channel': channel_name,
                'model_type': 'hill',
                'vmax_mean': trace.posterior['Vmax'].mean().item(),
                'ec50_mean': trace.posterior['EC50'].mean().item(),
                'r2': r2
            }
        else:  # linear
            trace = train_linear_model(x_data, y_data)
            fig, r2 = plot_linear_curve(x_data, y_data, trace, dates, channel_name, target_var)
            result = {
                'channel': channel_name,
                'model_type': 'linear',
                'alpha_mean': trace.posterior['alpha'].mean().item(),
                'beta_mean': trace.posterior['beta'].mean().item(),
                'r2': r2
            }
        
        return fig, r2, result
    except Exception as e:
        st.error(f"媒体 {channel_name} の分析中にエラー: {e}")
        return None, None, None

def train_optimization_models(df, config):
    """投資費用最適化用のモデル学習（Hill/Linear/GAM対応）"""
    model_params = {}
    
    for channel_name, cfg in config.items():
        model_type = cfg.get('model_type', 'linear')
        
        start_date = pd.to_datetime(cfg['start_date'])
        end_date = pd.to_datetime(cfg['end_date'])
        filtered_df = df[(df['channel'] == channel_name) & 
                        (df['week_start_date'] >= start_date) & 
                        (df['week_start_date'] <= end_date)].copy()
        
        if len(filtered_df) < 5:
            continue
        
        x_data = filtered_df['total_spend'].values
        y_data = filtered_df[cfg['target_variable']].values
        X_gam = filtered_df[['total_spend']]
        
        try:
            if model_type == 'hill':
                trace = train_hill_model(x_data, y_data)
                model_params[channel_name] = {
                    'model_type': 'hill',
                    'vmax_mean': trace.posterior['Vmax'].mean().item(),
                    'ec50_mean': trace.posterior['EC50'].mean().item(),
                    'min_spend': filtered_df['total_spend'].min(),
                    'max_spend': filtered_df['total_spend'].max()
                }
            elif model_type == 'linear':
                trace = train_linear_model(x_data, y_data)
                model_params[channel_name] = {
                    'model_type': 'linear',
                    'alpha_mean': trace.posterior['alpha'].mean().item(),
                    'beta_mean': trace.posterior['beta'].mean().item(),
                    'min_spend': filtered_df['total_spend'].min(),
                    'max_spend': filtered_df['total_spend'].max()
                }
            elif model_type == 'gam':
                if len(filtered_df) < 10:
                    continue
                gam = train_gam_model(X_gam, y_data)
                model_params[channel_name] = {
                    'model_type': 'gam',
                    'gam_model': gam,
                    'min_spend': filtered_df['total_spend'].min(),
                    'max_spend': filtered_df['total_spend'].max()
                }
        except Exception as e:
            st.warning(f"媒体 {channel_name} の学習中にエラー: {e}")
            continue
    
    return model_params

def optimize_budget_allocation(model_params, total_budget, priority_channels, priority_ratio, n_starts=1000):
    """予算配分最適化（マルチスタート法）"""
    channels = list(model_params.keys())
    n_channels = len(channels)
    
    def objective_function(budgets):
        total_revenue = 0
        for i, channel in enumerate(channels):
            params = model_params[channel]
            budget = budgets[i]
            revenue = 0
            
            if params['model_type'] == 'hill':
                vmax = params['vmax_mean']
                ec50 = params['ec50_mean']
                revenue = vmax * budget / (ec50 + budget + 1e-9)
            elif params['model_type'] == 'linear':
                alpha = params['alpha_mean']
                beta = params['beta_mean']
                revenue = alpha + beta * budget
            elif params['model_type'] == 'gam':
                revenue = params['gam_model'].predict(np.array([[budget]]))[0]
            
            total_revenue += max(0, revenue)
        return -total_revenue
    
    cons1 = {'type': 'eq', 'fun': lambda budgets: np.sum(budgets) - total_budget}
    priority_indices = [i for i, ch in enumerate(channels) if ch in priority_channels]
    cons2 = {'type': 'eq', 'fun': lambda budgets: np.sum(budgets[priority_indices]) - total_budget * priority_ratio}
    constraints = [cons1, cons2] if priority_channels else [cons1]
    
    bounds = []
    for channel in channels:
        params = model_params[channel]
        bounds.append((params['min_spend'], params['max_spend']))
    
    best_result = None
    best_fun_value = float('inf')
    
    progress_bar = st.progress(0)
    for i in range(n_starts):
        if (i + 1) % (n_starts // 10) == 0:
            progress_bar.progress((i + 1) / n_starts)
        
        random_values = np.random.rand(n_channels)
        initial_guess = (random_values / random_values.sum()) * total_budget
        
        try:
            result = minimize(objective_function, initial_guess, method='SLSQP', 
                            bounds=bounds, constraints=constraints, options={'disp': False})
            if result.success and result.fun < best_fun_value:
                best_fun_value = result.fun
                best_result = result
        except:
            pass
    
    progress_bar.empty()
    return best_result, channels

# ========================================
# サイドバー（1回のみ定義）
# ========================================
st.sidebar.title("📊 広告最適化ツール")

# データ読み込み
data_path = st.secrets.get("data_path", None)

if data_path:
    if st.session_state.combined_df is None:
        try:
            st.session_state.combined_df = pd.read_csv(data_path)
            st.session_state.combined_df['week_start_date'] = pd.to_datetime(st.session_state.combined_df['week_start_date'])
            st.sidebar.success(f"✅ データ読み込み完了 ({len(st.session_state.combined_df)}行)")
        except Exception as e:
            st.sidebar.error(f"データ読み込みエラー: {e}")
else:
    uploaded_file = st.sidebar.file_uploader(
        "CSVデータをアップロード",
        type=['csv'],
        help="combined_dfのCSVファイルをアップロードしてください"
    )
    
    if uploaded_file is not None:
        try:
            st.session_state.combined_df = pd.read_csv(uploaded_file)
            st.session_state.combined_df['week_start_date'] = pd.to_datetime(st.session_state.combined_df['week_start_date'])
            st.sidebar.success(f"✅ データ読み込み完了 ({len(st.session_state.combined_df)}行)")
        except Exception as e:
            st.sidebar.error(f"データ読み込みエラー: {e}")

page = st.sidebar.radio(
    "分析メニュー",
    ["📈 現状把握", "🎯 投資費用最適化", "🔍 事前効果検証(前半)", "📊 事前効果検証(後半)"],
    key="main_navigation"
)

st.sidebar.markdown("---")
st.sidebar.info("データ更新日: 2025年10月1日\nデータ期間: 2024年1月 - 2025年9月")

# データチェック
if st.session_state.combined_df is None:
    st.warning("⚠️ データがアップロードされていません。サイドバーからCSVファイルをアップロードしてください。")
    st.stop()

df = st.session_state.combined_df
available_channels = df['channel'].unique().tolist()

# ========================================
# メインコンテンツ
# ========================================

if page == "📈 現状把握":
    st.markdown('<div class="main-header">現状把握</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">媒体別パフォーマンス分析</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 各媒体ごとに学習期間や目的変数を個別に設定して、パフォーマンスを確認できます</div>', unsafe_allow_html=True)
    
    # 媒体選択
    selected_channels = st.multiselect(
        "分析する媒体を選択",
        available_channels,
        default=available_channels[:5] if len(available_channels) >= 5 else available_channels,
        key="status_channels"
    )
    
    if not selected_channels:
        st.warning("媒体を選択してください")
        st.stop()
    
    st.subheader("媒体別パラメータ設定")
    
    # 一括設定
    with st.expander("📝 一括設定", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            bulk_start = st.date_input("学習期間(開始)", value=pd.to_datetime("2024-01-01"), key="bulk_start")
        with col2:
            bulk_end = st.date_input("学習期間(終了)", value=pd.to_datetime("2025-09-30"), key="bulk_end")
        with col3:
            available_targets = [col for col in df.columns if '応募' in col or 'コンバージョン' in col]
            bulk_target = st.selectbox("目的変数", available_targets, key="bulk_target")
        with col4:
            bulk_model = st.selectbox("回帰モデル", ["Hill Model", "線形回帰"], key="bulk_model")
        
        if st.button("全媒体に適用", key="apply_bulk"):
            for ch in selected_channels:
                st.session_state[f"status_{ch}_start"] = bulk_start
                st.session_state[f"status_{ch}_end"] = bulk_end
                st.session_state[f"status_{ch}_target"] = bulk_target
                st.session_state[f"status_{ch}_model"] = bulk_model
            st.success("設定を全媒体に適用しました!")
            st.rerun()
    
    # 各媒体の設定
    config = {}
    for channel in selected_channels:
        with st.expander(f"📌 {channel}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                start_date = st.date_input(
                    "学習期間(開始)",
                    value=st.session_state.get(f"status_{channel}_start", pd.to_datetime("2024-01-01")),
                    key=f"status_{channel}_start"
                )
            
            with col2:
                end_date = st.date_input(
                    "学習期間(終了)",
                    value=st.session_state.get(f"status_{channel}_end", pd.to_datetime("2025-09-30")),
                    key=f"status_{channel}_end"
                )
            
            with col3:
                target_var = st.selectbox(
                    "目的変数",
                    available_targets,
                    index=0,
                    key=f"status_{channel}_target"
                )
            
            with col4:
                model_type = st.selectbox(
                    "回帰モデル",
                    ["Hill Model", "線形回帰"],
                    key=f"status_{channel}_model"
                )
            
            config[channel] = {
                'start_date': start_date,
                'end_date': end_date,
                'target_variable': target_var,
                'model_type': 'hill' if model_type == "Hill Model" else 'linear'
            }
    
    # 分析実行
    if st.button("全媒体の分析を実行", type="primary", key="run_analysis"):
        st.session_state.all_channel_results = []
        st.session_state.trained_models = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, channel in enumerate(selected_channels):
            status_text.text(f"分析中: {channel} ({i+1}/{len(selected_channels)})")
            
            fig, r2, result = analyze_channel(df, channel, config[channel])
            
            if result:
                st.session_state.trained_models[channel] = {
                    'fig': fig,
                    'r2': r2,
                    'config': config[channel]
                }
                if result['model_type'] == 'hill':
                    st.session_state.all_channel_results.append(result)
            
            progress_bar.progress((i + 1) / len(selected_channels))
        
        status_text.text("分析完了!")
        st.success("✅ 全媒体の分析が完了しました!")
        st.rerun()
    
    # 結果表示
    if st.session_state.trained_models:
        avg_r2 = np.mean([v['r2'] for v in st.session_state.trained_models.values()])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均モデル精度 (R²)", f"{avg_r2:.3f}", help="全媒体の平均値")
        
        st.subheader("媒体別パフォーマンス")
        tabs = st.tabs(selected_channels)
        
        for i, tab in enumerate(tabs):
            with tab:
                channel = selected_channels[i]
                if channel in st.session_state.trained_models:
                    model_info = st.session_state.trained_models[channel]
                    st.plotly_chart(model_info['fig'], use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R²スコア", f"{model_info['r2']:.3f}")
                    with col2:
                        st.metric("モデルタイプ", model_info['config']['model_type'].upper())
                else:
                    st.info("この媒体の分析結果がありません")
        
        # 全媒体比較グラフ
        if st.session_state.all_channel_results:
            st.subheader("全媒体比較(Hill関数曲線)")
            st.markdown('<div class="info-box">💡 各媒体のHill関数を同一空間にプロットし、費用対効果を一目で比較できます</div>', unsafe_allow_html=True)
            
            max_spend = df[df['channel'].isin([r['channel'] for r in st.session_state.all_channel_results])]['total_spend'].max()
            x_range = np.linspace(0, max_spend * 1.1, 200)
            
            combined_fig = plot_combined_hill_curves(st.session_state.all_channel_results, x_range)
            st.plotly_chart(combined_fig, use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
            <strong>📊 解釈:</strong> 曲線が左上にあるほど、低予算で高い効果を得られる媒体です。
            最も費用対効果が高い媒体から順に投資を検討してください。
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("「全媒体の分析を実行」ボタンをクリックして分析を開始してください")

elif page == "🎯 投資費用最適化":
    st.markdown('<div class="main-header">投資費用最適化</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">予算配分最適化</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 Hill/Linear/GAMモデルに対応した予算配分最適化を行います</div>', unsafe_allow_html=True)
    
    # 最適化用の媒体選択と設定
    st.subheader("最適化対象媒体の選択と設定")
    
    opt_channels = st.multiselect(
        "最適化対象の媒体を選択",
        available_channels,
        default=available_channels[:6] if len(available_channels) >= 6 else available_channels,
        key="opt_channels"
    )
    
    if not opt_channels:
        st.warning("最適化対象の媒体を選択してください")
        st.stop()
    
    # 各媒体のモデル設定
    st.subheader("各媒体のモデル設定")
    opt_config = {}
    
    for channel in opt_channels:
        with st.expander(f"📌 {channel}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                start_date = st.date_input(
                    "学習期間(開始)",
                    value=pd.to_datetime("2024-01-01"),
                    key=f"opt_{channel}_start"
                )
            
            with col2:
                end_date = st.date_input(
                    "学習期間(終了)",
                    value=pd.to_datetime("2025-09-30"),
                    key=f"opt_{channel}_end"
                )
            
            with col3:
                available_targets = [col for col in df.columns if '応募' in col or 'コンバージョン' in col]
                target_var = st.selectbox(
                    "目的変数",
                    available_targets,
                    index=0,
                    key=f"opt_{channel}_target"
                )
            
            with col4:
                model_type = st.selectbox(
                    "モデルタイプ",
                    ["Hill Model", "線形回帰", "GAM"],
                    key=f"opt_{channel}_model"
                )
            
            model_map = {"Hill Model": "hill", "線形回帰": "linear", "GAM": "gam"}
            opt_config[channel] = {
                'start_date': start_date,
                'end_date': end_date,
                'target_variable': target_var,
                'model_type': model_map[model_type]
            }
    
    # 最適化パラメータ
    st.subheader("最適化パラメータ")
    col1, col2 = st.columns(2)
    
    with col1:
        total_budget = st.number_input(
            "週当たりの総予算 (円)",
            min_value=0,
            value=30000000,
            step=1000000,
            format="%d",
            key="opt_budget"
        )
    
    with col2:
        n_starts = st.slider(
            "マルチスタート試行回数",
            10, 5000, 1000,
            help="多いほど精度が上がりますが時間がかかります",
            key="opt_nstarts"
        )
    
    # 優先媒体設定
    with st.expander("🎯 優先媒体設定（オプション）", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            priority_channels = st.multiselect(
                "優先媒体を選択",
                opt_channels,
                key="opt_priority_channels"
            )
        with col2:
            priority_ratio = st.slider(
                "優先媒体への配分比率",
                0.0, 1.0, 0.70, 0.001,
                key="opt_priority_ratio"
            )
    
    # 最適化実行
    if st.button("最適配分を計算", type="primary", key="run_optimization"):
        with st.spinner("モデル学習中..."):
            model_params = train_optimization_models(df, opt_config)
        
        if not model_params:
            st.error("モデルの学習に失敗しました。設定を確認してください。")
            st.stop()
        
        st.success(f"✅ {len(model_params)}媒体のモデル学習完了")
        
        with st.spinner(f"{n_starts}回の最適化を実行中..."):
            result, channels = optimize_budget_allocation(
                model_params, 
                total_budget, 
                priority_channels if priority_channels else [],
                priority_ratio if priority_channels else 0,
                n_starts
            )
        
        if result and result.success:
            st.session_state.optimization_result = {
                'budgets': result.x,
                'channels': channels,
                'total_revenue': -result.fun,
                'model_params': model_params
            }
            st.success("✅ 最適化が完了しました!")
            st.rerun()
        else:
            st.error("最適化に失敗しました。パラメータを調整してください。")
    
    # 結果表示
    if st.session_state.optimization_result:
        st.subheader("最適予算配分結果")
        
        result_data = st.session_state.optimization_result
        optimal_budgets = result_data['budgets']
        channels = result_data['channels']
        model_params = result_data['model_params']
        
        # 予測成果を計算
        predicted_revenues = []
        for i, ch in enumerate(channels):
            budget = optimal_budgets[i]
            params = model_params[ch]
            
            if params['model_type'] == 'hill':
                revenue = params['vmax_mean'] * budget / (params['ec50_mean'] + budget + 1e-9)
            elif params['model_type'] == 'linear':
                revenue = params['alpha_mean'] + params['beta_mean'] * budget
            elif params['model_type'] == 'gam':
                revenue = params['gam_model'].predict(np.array([[budget]]))[0]
            
            predicted_revenues.append(max(0, revenue))
        
        result_df = pd.DataFrame({
            '順位': range(len(channels)),
            '媒体': channels,
            'モデル': [model_params[ch]['model_type'].upper() for ch in channels],
            '最適配分予算': optimal_budgets,
            '予算比率': optimal_budgets / total_budget,
            '予測成果': predicted_revenues
        })
        
        result_df = result_df.sort_values('最適配分予算', ascending=False).reset_index(drop=True)
        result_df['順位'] = range(len(result_df))
        
        # フォーマット
        display_df = result_df.copy()
        display_df['最適配分予算'] = display_df['最適配分予算'].apply(lambda x: f"¥{x:,.0f}")
        display_df['予算比率'] = display_df['予算比率'].apply(lambda x: f"{x:.1%}")
        display_df['予測成果'] = display_df['予測成果'].apply(lambda x: f"{x:,.1f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # サマリー
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("予測合計成果", f"{result_data['total_revenue']:,.0f}")
        with col2:
            st.metric("配分媒体数", len(channels))
        with col3:
            st.metric("総予算", f"¥{total_budget:,.0f}")

elif page == "🔍 事前効果検証(前半)":
    st.markdown('<div class="main-header">事前効果検証(前半)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">シナリオ比較 - 点推定予測</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 システム提案と現場担当者の予算配分を比較し、どちらが優れているか事前に検証できます</div>', unsafe_allow_html=True)
    
    st.info("この機能は準備中です。「投資費用最適化」で最適化を実行してから、手動の予算配分と比較してください。")

elif page == "📊 事前効果検証(後半)":
    st.markdown('<div class="main-header">事前効果検証(後半)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">シナリオ比較 - 区間推定(モンテカルロシミュレーション)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 ベイズ的シミュレーションにより、不確実性を考慮した比較が可能です</div>', unsafe_allow_html=True)
    
    st.info("この機能は準備中です。")

# フッター
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 広告最適化ツール")