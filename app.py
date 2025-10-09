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
        
        # ★★★ 新機能: 0円を除外した下位3つの平均を最低予算として設定 ★★★
        non_zero_spends = filtered_df[filtered_df['total_spend'] > 0]['total_spend'].values
        
        if len(non_zero_spends) >= 3:
            # 下位3つの平均を計算
            sorted_spends = np.sort(non_zero_spends)
            bottom_3_avg = np.mean(sorted_spends[:3])
            min_budget_constraint = bottom_3_avg
        elif len(non_zero_spends) > 0:
            # データが3件未満の場合は最小値を使用
            min_budget_constraint = np.min(non_zero_spends)
        else:
            # 0円しかない場合は最小値として1000円を設定
            min_budget_constraint = 1000
        
        # 最低でも1000円は確保
        min_budget_constraint = max(min_budget_constraint, 1000)
        
        try:
            if model_type == 'hill':
                trace = train_hill_model(x_data, y_data)
                model_params[channel_name] = {
                    'model_type': 'hill',
                    'vmax_mean': trace.posterior['Vmax'].mean().item(),
                    'ec50_mean': trace.posterior['EC50'].mean().item(),
                    'min_spend': min_budget_constraint,
                    'max_spend': filtered_df['total_spend'].max(),
                    'bottom_3_avg': bottom_3_avg if len(non_zero_spends) >= 3 else min_budget_constraint
                }
            elif model_type == 'linear':
                trace = train_linear_model(x_data, y_data)
                model_params[channel_name] = {
                    'model_type': 'linear',
                    'alpha_mean': trace.posterior['alpha'].mean().item(),
                    'beta_mean': trace.posterior['beta'].mean().item(),
                    'min_spend': min_budget_constraint,
                    'max_spend': filtered_df['total_spend'].max(),
                    'bottom_3_avg': bottom_3_avg if len(non_zero_spends) >= 3 else min_budget_constraint
                }
            elif model_type == 'gam':
                if len(filtered_df) < 10:
                    continue
                gam = train_gam_model(X_gam, y_data)
                model_params[channel_name] = {
                    'model_type': 'gam',
                    'gam_model': gam,
                    'min_spend': min_budget_constraint,
                    'max_spend': filtered_df['total_spend'].max(),
                    'bottom_3_avg': bottom_3_avg if len(non_zero_spends) >= 3 else min_budget_constraint
                }
        except Exception as e:
            st.warning(f"媒体 {channel_name} の学習中にエラー: {e}")
            continue
    
    return model_params

def train_models_with_uncertainty(df, config):
    """不確実性情報を含むモデル学習（Hill/Linear/GAM対応）"""
    model_results = {}
    
    for channel_name, cfg in config.items():
        model_type = cfg.get('model_type', 'linear')
        
        start_date = pd.to_datetime(cfg['start_date'])
        end_date = pd.to_datetime(cfg['end_date'])
        filtered_df = df[(df['channel'] == channel_name) & 
                        (df['week_start_date'] >= start_date) & 
                        (df['week_start_date'] <= end_date)].copy()
        
        if len(filtered_df) < 10:
            continue
        
        x_data = filtered_df['total_spend'].values
        y_data = filtered_df[cfg['target_variable']].values
        X_gam = filtered_df[['total_spend']]
        
        try:
            if model_type == 'hill':
                trace = train_hill_model(x_data, y_data)
                model_results[channel_name] = {
                    'model_type': 'hill',
                    'trace': trace
                }
            elif model_type == 'linear':
                trace = train_linear_model(x_data, y_data)
                model_results[channel_name] = {
                    'model_type': 'linear',
                    'trace': trace
                }
            elif model_type == 'gam':
                # GAMはブートストラップで信頼区間を計算
                spend_range = np.linspace(X_gam.values.min(), X_gam.values.max(), 100)
                n_bootstraps = 200  # Streamlit用に少なめに設定
                bootstrap_preds = []
                
                progress = st.progress(0, text=f"{channel_name}: ブートストラップ中...")
                for i in range(n_bootstraps):
                    if (i + 1) % 20 == 0:
                        progress.progress((i + 1) / n_bootstraps)
                    resampled_indices = np.random.choice(X_gam.index, size=len(X_gam), replace=True)
                    X_boot = X_gam.loc[resampled_indices]
                    y_boot = y_data[X_gam.index.get_indexer(resampled_indices)]
                    gam_boot = train_gam_model(X_boot, y_boot)
                    pred_boot = gam_boot.predict(spend_range.reshape(-1, 1))
                    bootstrap_preds.append(pred_boot)
                
                progress.empty()
                
                bootstrap_preds = np.array(bootstrap_preds)
                lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
                upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)
                intervals = np.c_[lower_bound, upper_bound]
                final_gam = train_gam_model(X_gam, y_data)
                
                model_results[channel_name] = {
                    'model_type': 'gam',
                    'gam_model': final_gam,
                    'spend_range': spend_range,
                    'intervals': intervals
                }
        except Exception as e:
            st.warning(f"媒体 {channel_name} の学習中にエラー: {e}")
            continue
    
    return model_results

def simulate_scenarios(trained_models, scenario1_ratios, scenario2_ratios, total_budget, n_samples=5000):
    """2つのシナリオをモンテカルロシミュレーションで比較"""
    channels = list(trained_models.keys())
    
    # 比率の正規化
    def normalize_ratios(ratios):
        total_ratio = sum(ratios.get(ch, 0) for ch in channels)
        if total_ratio > 0 and not np.isclose(total_ratio, 1.0):
            return {ch: ratios.get(ch, 0) / total_ratio for ch in channels}
        return ratios
    
    scenario1_ratios = normalize_ratios(scenario1_ratios)
    scenario2_ratios = normalize_ratios(scenario2_ratios)
    
    s1_revenues = []
    s2_revenues = []
    
    progress_bar = st.progress(0, text=f"{n_samples}回のシミュレーション実行中...")
    
    for i in range(n_samples):
        if (i + 1) % (n_samples // 10) == 0:
            progress_bar.progress((i + 1) / n_samples)
        
        s1_total = 0
        s2_total = 0
        
        for ch in channels:
            params = trained_models[ch]
            s1_budget = scenario1_ratios.get(ch, 0) * total_budget
            s2_budget = scenario2_ratios.get(ch, 0) * total_budget
            
            if params['model_type'] == 'hill':
                trace = params['trace']
                posterior_samples = az.extract(trace, num_samples=1)
                vmax = posterior_samples['Vmax'].item()
                ec50 = posterior_samples['EC50'].item()
                s1_total += max(0, vmax * s1_budget / (ec50 + s1_budget + 1e-9))
                s2_total += max(0, vmax * s2_budget / (ec50 + s2_budget + 1e-9))
                
            elif params['model_type'] == 'linear':
                trace = params['trace']
                posterior_samples = az.extract(trace, num_samples=1)
                alpha = posterior_samples['alpha'].item()
                beta = posterior_samples['beta'].item()
                s1_total += max(0, alpha + beta * s1_budget)
                s2_total += max(0, alpha + beta * s2_budget)
                
            elif params['model_type'] == 'gam':
                gam_model = params['gam_model']
                spend_range = params['spend_range']
                intervals = params['intervals']
                
                # GAMの不確実性をサンプリング
                for budget, total in [(s1_budget, s1_total), (s2_budget, s2_total)]:
                    idx = np.argmin(np.abs(spend_range - budget))
                    mean_pred = gam_model.predict(np.array([[budget]]))[0]
                    lower, upper = intervals[idx]
                    std_dev = (upper - lower) / 4.0
                    sample = np.random.normal(mean_pred, max(std_dev, 1e-9))
                    if budget == s1_budget:
                        s1_total += max(0, sample)
                    else:
                        s2_total += max(0, sample)
        
        s1_revenues.append(s1_total)
        s2_revenues.append(s2_total)
    
    progress_bar.empty()
    
    return np.array(s1_revenues), np.array(s2_revenues)
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
            10, 3000, 1000,
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
                0.0, 1.0, 0.70, 0.05,
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
            '最低予算制約': [model_params[ch].get('bottom_3_avg', model_params[ch]['min_spend']) for ch in channels],
            '予算比率': optimal_budgets / total_budget,
            '予測成果': predicted_revenues
        })
        
        result_df = result_df.sort_values('最適配分予算', ascending=False).reset_index(drop=True)
        result_df['順位'] = range(len(result_df))
        
        # フォーマット
        display_df = result_df.copy()
        display_df['最適配分予算'] = display_df['最適配分予算'].apply(lambda x: f"¥{x:,.0f}")
        display_df['最低予算制約'] = display_df['最低予算制約'].apply(lambda x: f"¥{x:,.0f}")
        display_df['予算比率'] = display_df['予算比率'].apply(lambda x: f"{x:.1%}")
        display_df['予測成果'] = display_df['予測成果'].apply(lambda x: f"{x:,.1f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # 制約の説明
        st.info("💡 **最低予算制約**: 各媒体の過去データから0円を除いた下位3つの平均金額。すべての媒体でこの金額以上が配分されます。")
        
        # サマリー
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("予測合計成果", f"{result_data['total_revenue']:,.0f}")
        with col2:
            st.metric("配分媒体数", len(channels))
        with col3:
            st.metric("総予算", f"¥{total_budget:,.0f}")
        
        # 制約充足の確認
        st.subheader("制約充足の確認")
        constraint_check = []
        for i, ch in enumerate(channels):
            min_constraint = model_params[ch]['min_spend']
            actual_budget = optimal_budgets[i]
            is_satisfied = actual_budget >= min_constraint
            constraint_check.append({
                '媒体': ch,
                '最低予算制約': f"¥{min_constraint:,.0f}",
                '配分予算': f"¥{actual_budget:,.0f}",
                '制約充足': '✅' if is_satisfied else '❌'
            })
        
        check_df = pd.DataFrame(constraint_check)
        st.dataframe(check_df, use_container_width=True, hide_index=True)
        
        if all([c['制約充足'] == '✅' for c in constraint_check]):
            st.success("✅ すべての媒体で最低予算制約を満たしています")
        else:
            st.warning("⚠️ 一部の媒体で最低予算制約を満たしていません。総予算を増やすか、優先媒体の設定を調整してください。")

elif page == "🔍 事前効果検証(前半)":
    st.markdown('<div class="main-header">事前効果検証(前半)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">シナリオ比較 - ベイズ的シミュレーション</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 2つの予算配分案をモンテカルロシミュレーションで比較し、どちらが優れているか確率的に検証します</div>', unsafe_allow_html=True)
    
    # 媒体選択
    st.subheader("比較対象媒体の選択")
    comparison_channels = st.multiselect(
        "シミュレーション対象の媒体を選択",
        available_channels,
        default=available_channels[:6] if len(available_channels) >= 6 else available_channels,
        key="comparison_channels"
    )
    
    if not comparison_channels:
        st.warning("媒体を選択してください")
        st.stop()
    
    # 各媒体のモデル設定
    st.subheader("各媒体のモデル設定")
    comparison_config = {}
    
    for channel in comparison_channels:
        with st.expander(f"📌 {channel}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                start_date = st.date_input(
                    "学習期間(開始)",
                    value=pd.to_datetime("2025-05-01"),
                    key=f"comp_{channel}_start"
                )
            
            with col2:
                end_date = st.date_input(
                    "学習期間(終了)",
                    value=pd.to_datetime("2025-09-30"),
                    key=f"comp_{channel}_end"
                )
            
            with col3:
                available_targets = [col for col in df.columns if '応募' in col or 'コンバージョン' in col]
                target_var = st.selectbox(
                    "目的変数",
                    available_targets,
                    index=0,
                    key=f"comp_{channel}_target"
                )
            
            with col4:
                model_type = st.selectbox(
                    "モデルタイプ",
                    ["Hill Model", "線形回帰", "GAM"],
                    key=f"comp_{channel}_model"
                )
            
            model_map = {"Hill Model": "hill", "線形回帰": "linear", "GAM": "gam"}
            comparison_config[channel] = {
                'start_date': start_date,
                'end_date': end_date,
                'target_variable': target_var,
                'model_type': model_map[model_type]
            }
    
    # 総予算設定
    st.subheader("総予算設定")
    total_budget = st.number_input(
        "週当たりの総予算 (円)",
        min_value=0,
        value=30000000,
        step=1000000,
        format="%d",
        key="comparison_budget"
    )
    
    # シナリオ入力
    st.subheader("2つのシナリオの予算配分")
    
    col1, col2 = st.columns(2)
    
    scenario1_ratios = {}
    scenario2_ratios = {}
    
    with col1:
        st.markdown("#### 🤖 シナリオ1: システム提案（最適化案）")
        for ch in comparison_channels:
            default_ratio = round(1.0 / len(comparison_channels), 2)
            ratio = st.number_input(
                f"{ch} の配分比率",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(comparison_channels),
                step=0.01,
                format="%.2f",
                key=f"s1_ratio_{ch}"
            )
            scenario1_ratios[ch] = ratio
        
        s1_total = sum(scenario1_ratios.values())
        if not np.isclose(s1_total, 1.0):
            st.warning(f"⚠️ 合計: {s1_total:.2%} (100%になるよう調整されます)")
    
    with col2:
        st.markdown("#### 👤 シナリオ2: 現場担当者提案")
        for ch in comparison_channels:
            default_ratio = round(1.0 / len(comparison_channels), 2)
            ratio = st.number_input(
                f"{ch} の配分比率",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(comparison_channels),
                step=0.01,
                format="%.2f",
                key=f"s2_ratio_{ch}"
            )
            scenario2_ratios[ch] = ratio
        
        s2_total = sum(scenario2_ratios.values())
        if not np.isclose(s2_total, 1.0):
            st.warning(f"⚠️ 合計: {s2_total:.2%} (100%になるよう調整されます)")
    
    # シミュレーション設定
    st.subheader("シミュレーション設定")
    n_samples = st.slider(
        "モンテカルロシミュレーション試行回数",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=500,
        key="n_samples"
    )
    
    # 実行ボタン
    if st.button("シナリオ比較分析を実行", type="primary", key="run_comparison"):
        # モデル学習
        with st.spinner("モデル学習中（不確実性情報を抽出）..."):
            trained_models = train_models_with_uncertainty(df, comparison_config)
        
        if not trained_models:
            st.error("モデルの学習に失敗しました。設定を確認してください。")
            st.stop()
        
        st.success(f"✅ {len(trained_models)}媒体のモデル学習完了")
        
        # シミュレーション実行
        with st.spinner(f"{n_samples}回のシミュレーション実行中..."):
            s1_revenues, s2_revenues = simulate_scenarios(
                trained_models,
                scenario1_ratios,
                scenario2_ratios,
                total_budget,
                n_samples
            )
        
        st.success("✅ シミュレーションが完了しました!")
        
        # 結果をセッションステートに保存
        st.session_state.comparison_result = {
            's1_revenues': s1_revenues,
            's2_revenues': s2_revenues,
            's1_ratios': scenario1_ratios,
            's2_ratios': scenario2_ratios,
            'total_budget': total_budget,
            'n_samples': n_samples
        }
        
        st.rerun()
    
    # 結果表示
    if 'comparison_result' in st.session_state:
        result = st.session_state.comparison_result
        s1_revenues = result['s1_revenues']
        s2_revenues = result['s2_revenues']
        
        # 統計量計算
        s1_mean = np.mean(s1_revenues)
        s1_median = np.median(s1_revenues)
        s2_mean = np.mean(s2_revenues)
        s2_median = np.median(s2_revenues)
        prob_s1_wins = np.mean(s1_revenues > s2_revenues)
        
        # サマリー表示
        st.subheader("シミュレーション結果サマリー")
        
        summary_df = pd.DataFrame({
            'シナリオ': ['シナリオ1 (システム提案) 🏆' if prob_s1_wins > 0.5 else 'シナリオ1 (システム提案)', 
                        'シナリオ2 (現場提案) 🏆' if prob_s1_wins <= 0.5 else 'シナリオ2 (現場提案)',
                        '改善効果'],
            '予測成果(期待値)': [f'{s1_mean:,.0f}', f'{s2_mean:,.0f}', 
                               f'+{s1_mean - s2_mean:,.0f} ({(s1_mean - s2_mean) / s2_mean * 100:+.1f}%)'],
            '予測成果(中央値)': [f'{s1_median:,.0f}', f'{s2_median:,.0f}',
                               f'+{s1_median - s2_median:,.0f} ({(s1_median - s2_median) / s2_median * 100:+.1f}%)']
        })
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # 確率表示
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 2rem 0;'>
            <div style='font-size: 1.2rem; color: #2c3e50; margin-bottom: 1rem;'>
                <strong>シナリオ1（システム提案）がシナリオ2（現場提案）を上回る確率</strong>
            </div>
            <div style='font-size: 4rem; font-weight: 700; color: {"#2ecc71" if prob_s1_wins > 0.5 else "#e74c3c"}; margin: 1rem 0;'>
                {prob_s1_wins * 100:.1f}%
            </div>
            <div style='font-size: 1rem; color: #7f8c8d;'>
                {result['n_samples']:,}回のモンテカルロシミュレーション結果
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ヒストグラム
        st.subheader("予測成果の確率分布")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=s1_revenues,
            name=f'シナリオ1 (期待値: {s1_mean:,.0f})',
            opacity=0.7,
            marker_color='#3498db',
            nbinsx=50
        ))
        
        fig.add_trace(go.Histogram(
            x=s2_revenues,
            name=f'シナリオ2 (期待値: {s2_mean:,.0f})',
            opacity=0.7,
            marker_color='#e67e22',
            nbinsx=50
        ))
        
        fig.add_vline(x=s1_mean, line_dash="dash", line_color="#3498db", 
                     annotation_text="S1平均", annotation_position="top")
        fig.add_vline(x=s2_mean, line_dash="dash", line_color="#e67e22",
                     annotation_text="S2平均", annotation_position="top")
        
        fig.update_layout(
            title='合計成果の予測分布',
            xaxis_title='予測成果',
            yaxis_title='頻度',
            barmode='overlay',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 詳細な統計情報
        with st.expander("📊 詳細な統計情報"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**シナリオ1の統計量**")
                st.write(f"- 平均: {s1_mean:,.0f}")
                st.write(f"- 中央値: {s1_median:,.0f}")
                st.write(f"- 標準偏差: {np.std(s1_revenues):,.0f}")
                st.write(f"- 5%点: {np.percentile(s1_revenues, 5):,.0f}")
                st.write(f"- 95%点: {np.percentile(s1_revenues, 95):,.0f}")
            
            with col2:
                st.write("**シナリオ2の統計量**")
                st.write(f"- 平均: {s2_mean:,.0f}")
                st.write(f"- 中央値: {s2_median:,.0f}")
                st.write(f"- 標準偏差: {np.std(s2_revenues):,.0f}")
                st.write(f"- 5%点: {np.percentile(s2_revenues, 5):,.0f}")
                st.write(f"- 95%点: {np.percentile(s2_revenues, 95):,.0f}")

elif page == "📊 事前効果検証(後半)":
    st.markdown('<div class="main-header">事前効果検証(後半)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">シナリオ比較 - 区間推定(モンテカルロシミュレーション)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 ベイズ的シミュレーションにより、不確実性を考慮した比較が可能です</div>', unsafe_allow_html=True)
    
    st.info("この機能は準備中です。")

# フッター
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 広告最適化ツール")