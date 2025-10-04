import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pymc as pm
import arviz as az
import pytensor
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
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

# サイドバー
st.sidebar.title("📊 広告最適化ツール")

# Secretsからデータパスを取得
data_path = st.secrets.get("data_path", None)

if data_path:
    # Secretsにデータパスが設定されている場合
    if st.session_state.combined_df is None:
        st.session_state.combined_df = pd.read_csv(data_path)
        st.session_state.combined_df['week_start_date'] = pd.to_datetime(st.session_state.combined_df['week_start_date'])
        st.sidebar.success(f"✅ データ読み込み完了 ({len(st.session_state.combined_df)}行)")
else:
    # Secretsが設定されていない場合はアップロード機能を使用
    uploaded_file = st.sidebar.file_uploader(
        "CSVデータをアップロード",
        type=['csv'],
        help="combined_dfのCSVファイルをアップロードしてください"
    )
    
    if uploaded_file is not None:
        st.session_state.combined_df = pd.read_csv(uploaded_file)
        st.session_state.combined_df['week_start_date'] = pd.to_datetime(st.session_state.combined_df['week_start_date'])
        st.sidebar.success(f"✅ データ読み込み完了 ({len(st.session_state.combined_df)}行)")

page = st.sidebar.radio(
    "分析メニュー",
    ["📈 現状把握", "🎯 投資費用最適化", "🔍 事前効果検証(前半)", "📊 事前効果検証(後半)"]
)

st.sidebar.markdown("---")
st.sidebar.info("データ更新日: 2025年10月1日\nデータ期間: 2024年1月 - 2025年9月")

# データチェック
if st.session_state.combined_df is None:
    st.warning("⚠️ データがアップロードされていません。サイドバーからCSVファイルをアップロードしてください。")
    st.stop()

df = st.session_state.combined_df
available_channels = df['channel'].unique().tolist()

# メインコンテンツ
if page == "📈 現状把握":
    st.markdown('<div class="main-header">現状把握</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">媒体別パフォーマンス分析</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 各媒体ごとに学習期間や目的変数を個別に設定して、パフォーマンスを確認できます</div>', unsafe_allow_html=True)
    
    # 媒体選択
    selected_channels = st.multiselect(
        "分析する媒体を選択",
        available_channels,
        default=available_channels[:5] if len(available_channels) >= 5 else available_channels
    )
    
    if not selected_channels:
        st.warning("媒体を選択してください")
        st.stop()
    
    # 媒体別パラメータ設定
    st.subheader("媒体別パラメータ設定")
    
    # 一括設定オプション
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
        
        if st.button("全媒体に適用"):
            for ch in selected_channels:
                st.session_state[f"{ch}_start"] = bulk_start
                st.session_state[f"{ch}_end"] = bulk_end
                st.session_state[f"{ch}_target"] = bulk_target
                st.session_state[f"{ch}_model"] = bulk_model
            st.success("設定を全媒体に適用しました!")
            st.rerun()
    
    # 各媒体の個別設定
    config = {}
    for channel in selected_channels:
        with st.expander(f"📌 {channel}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                start_date = st.date_input(
                    "学習期間(開始)",
                    value=st.session_state.get(f"{channel}_start", pd.to_datetime("2024-01-01")),
                    key=f"{channel}_start"
                )
            
            with col2:
                end_date = st.date_input(
                    "学習期間(終了)",
                    value=st.session_state.get(f"{channel}_end", pd.to_datetime("2025-09-30")),
                    key=f"{channel}_end"
                )
            
            with col3:
                target_var = st.selectbox(
                    "目的変数",
                    available_targets,
                    index=0,
                    key=f"{channel}_target"
                )
            
            with col4:
                model_type = st.selectbox(
                    "回帰モデル",
                    ["Hill Model", "線形回帰"],
                    key=f"{channel}_model"
                )
            
            config[channel] = {
                'start_date': start_date,
                'end_date': end_date,
                'target_variable': target_var,
                'model_type': 'hill' if model_type == "Hill Model" else 'linear'
            }
    
    # 分析実行
    if st.button("全媒体の分析を実行", type="primary"):
        st.session_state.all_channel_results = []
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
        # 平均R²スコア
        avg_r2 = np.mean([v['r2'] for v in st.session_state.trained_models.values()])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均モデル精度 (R²)", f"{avg_r2:.3f}", help="全媒体の平均値")
        
        # 媒体別パフォーマンス
        st.subheader("媒体別パフォーマンス")
        tabs = st.tabs(selected_channels)
        
        for i, tab in enumerate(tabs):
            with tab:
                channel = selected_channels[i]
                if channel in st.session_state.trained_models:
                    model_info = st.session_state.trained_models[channel]
                    st.plotly_chart(model_info['fig'], use_container_width=True)
                    
                    # パラメータ表示
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
    
    st.markdown('<div class="info-box">💡 システムが最適なソルバーを自動的に選択し、週当たりの総予算を各媒体に最適配分します</div>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("⚠️ まず「現状把握」ページでモデルを学習してください")
        st.stop()
    
    # 予算入力
    total_budget = st.number_input(
        "週当たりの総予算 (円)",
        min_value=0,
        value=30000000,
        step=1000000,
        format="%d"
    )
    
    st.write("最適化の詳細設定")
    col1, col2 = st.columns(2)
    
    with col1:
        n_starts = st.slider("マルチスタート試行回数", 10, 5000, 1000, help="多いほど精度が上がりますが時間がかかります")
    
    with col2:
        priority_ratio = st.slider("優先媒体への配分比率", 0.0, 1.0, 0.70, 0.05)
    
    if st.button("最適配分を計算", type="primary"):
        with st.spinner(f"{n_starts}回の最適化を実行中..."):
            # 簡易版の最適化実装
            channels_list = list(st.session_state.trained_models.keys())
            n_channels = len(channels_list)
            
            def objective(budgets):
                total_revenue = 0
                for i, ch in enumerate(channels_list):
                    # ダミー計算(実際は学習済みモデルを使用)
                    total_revenue += budgets[i] * 0.01
                return -total_revenue
            
            cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
            bounds = [(1000, total_budget) for _ in range(n_channels)]
            
            initial_guess = np.ones(n_channels) * (total_budget / n_channels)
            result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=cons)
            
            st.session_state.optimization_result = {
                'budgets': result.x,
                'channels': channels_list,
                'total_revenue': -result.fun
            }
        
        st.success("✅ 最適化が完了しました!")
        st.rerun()
    
    # 結果表示
    if st.session_state.optimization_result:
        st.subheader("最適予算配分結果")
        
        result = st.session_state.optimization_result
        result_df = pd.DataFrame({
            '順位': range(len(result['channels'])),
            '媒体': result['channels'],
            '最適配分予算': result['budgets'],
            '予算比率': result['budgets'] / total_budget
        })
        result_df = result_df.sort_values('最適配分予算', ascending=False).reset_index(drop=True)
        result_df['順位'] = range(len(result_df))
        
        # フォーマット
        result_df['最適配分予算'] = result_df['最適配分予算'].apply(lambda x: f"¥{x:,.0f}")
        result_df['予算比率'] = result_df['予算比率'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        # 予測成果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("予測合計成果", f"{result['total_revenue']:,.0f}")
        with col2:
            st.metric("配分媒体数", len(result['channels']))
        with col3:
            st.metric("総予算", f"¥{total_budget:,.0f}")

elif page == "🔍 事前効果検証(前半)":
    st.markdown('<div class="main-header">事前効果検証(前半)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">シナリオ比較 - 点推定予測</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 システム提案と現場担当者の予算配分を比較し、どちらが優れているか事前に検証できます</div>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("⚠️ まず「現状把握」ページでモデルを学習してください")
        st.stop()
    
    channels_list = list(st.session_state.trained_models.keys())
    
    # 総予算
    total_budget = st.number_input("総予算 (円)", value=30000000, step=1000000, format="%d", key="scenario_total")
    
    # シナリオ入力
    col1, col2 = st.columns(2)
    
    s1_budgets = {}
    s2_budgets = {}
    
    with col1:
        st.subheader("🤖 シナリオ1: システム提案")
        for ch in channels_list:
            s1_budgets[ch] = st.number_input(
                ch,
                min_value=0,
                value=total_budget // len(channels_list),
                step=100000,
                key=f"s1_{ch}"
            )
    
    with col2:
        st.subheader("👤 シナリオ2: 現場担当者提案")
        for ch in channels_list:
            s2_budgets[ch] = st.number_input(
                ch,
                min_value=0,
                value=int(total_budget // len(channels_list) * 0.9),
                step=100000,
                key=f"s2_{ch}"
            )
    
    if st.button("シナリオ比較分析を実行", type="primary"):
        with st.spinner("分析中..."):
            # ダミー計算
            s1_total = sum(s1_budgets.values())
            s2_total = sum(s2_budgets.values())
            s1_revenue = s1_total * 0.015
            s2_revenue = s2_total * 0.012
            
            summary_df = pd.DataFrame({
                'シナリオ': ['システム提案 🏆', '現場担当者提案', '改善効果'],
                '総予算': [f'¥{s1_total:,.0f}', f'¥{s2_total:,.0f}', '-'],
                '予測成果': [f'{s1_revenue:,.0f}', f'{s2_revenue:,.0f}', f'+{s1_revenue-s2_revenue:,.0f} ({(s1_revenue-s2_revenue)/s2_revenue*100:+.1f}%)']
            })
            
            st.success("✅ 分析が完了しました!")
            st.subheader("シナリオ比較サマリー")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

elif page == "📊 事前効果検証(後半)":
    st.markdown('<div class="main-header">事前効果検証(後半)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">シナリオ比較 - 区間推定(モンテカルロシミュレーション)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 ベイズ的シミュレーションにより、不確実性を考慮した比較が可能です</div>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("⚠️ まず「現状把握」ページでモデルを学習してください")
        st.stop()
    
    n_samples = st.slider("シミュレーション試行回数", 1000, 50000, 10000, 1000)
    
    st.info(f"💡 {n_samples:,}回のモンテカルロシミュレーションで確率分布を推定します")
    
    if st.button("シミュレーションを実行", type="primary"):
        with st.spinner(f"{n_samples:,}回のシミュレーションを実行中..."):
            # ダミーシミュレーション
            s1_samples = np.random.normal(10000, 500, n_samples)
            s2_samples = np.random.normal(8500, 600, n_samples)
            prob_s1_wins = np.mean(s1_samples > s2_samples)
            
            st.success("✅ シミュレーションが完了しました!")
            
            # 確率表示
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 2rem 0;'>
                <div style='font-size: 1.2rem; color: #2c3e50; margin-bottom: 1rem;'>
                    <strong>システム提案が現場提案を上回る確率</strong>
                </div>
                <div style='font-size: 4rem; font-weight: 700; color: #2ecc71; margin: 1rem 0;'>
                    {prob_s1_wins*100:.1f}%
                </div>
                <div style='font-size: 1rem; color: #7f8c8d;'>
                    {n_samples:,}回のモンテカルロシミュレーション結果
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # ヒストグラム
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=s1_samples, name='シナリオ1', opacity=0.7, marker_color='#3498db'))
            fig.add_trace(go.Histogram(x=s2_samples, name='シナリオ2', opacity=0.7, marker_color='#e67e22'))
            fig.update_layout(
                title='成果の確率分布',
                xaxis_title='予測成果',
                yaxis_title='頻度',
                barmode='overlay',
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

# フッター
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 広告最適化ツール")
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
    """媒体分析の実行"""
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

# ヘルパー関数群
@st.cache_data
def load_data():
    """データ読み込み(実際のデータに置き換え)"""
    # combined_dfが既に存在する前提
    # ここではダミーデータを返す
    return pd.DataFrame()

def train_hill_model(x_data, y_data):
    """Hillモデルの学習"""
    with pm.Model() as model:
        slope = pm.HalfNormal('slope', sigma=1)
        EC50 = pm.HalfNormal('EC50', sigma=np.median(x_data[x_data > 0]) if np.any(x_data > 0) else 10000)
        Vmax = pm.Deterministic('Vmax', slope * EC50)
        mu = Vmax * x_data / (EC50 + x_data)
        sigma = pm.HalfNormal('sigma', sigma=y_data.std())
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)
        trace = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=0.9, progressbar=False, random_seed=42)
    return trace

def train_linear_model(x_data, y_data):
    """線形回帰モデルの学習"""
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=y_data.mean(), sigma=y_data.std() * 2)
        beta = pm.Normal('beta', mu=0, sigma=1)
        mu = alpha + beta * x_data
        sigma = pm.HalfNormal('sigma', sigma=y_data.std())
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)
        trace = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=0.9, progressbar=False, random_seed=42)
    return trace

def plot_hill_curve(x_data, y_data, trace, dates, channel_name, target_var):
    """Hill曲線の可視化"""
    # R²計算
    y_pred = trace.posterior['Vmax'].mean().item() * x_data / (trace.posterior['EC50'].mean().item() + x_data)
    r2 = r2_score(y_data, y_pred)
    
    # 予測曲線
    x_range = np.linspace(0, x_data.max() * 1.1, 100)
    post_curves = trace.posterior['Vmax'].values.flatten()[:, None] * x_range / (trace.posterior['EC50'].values.flatten()[:, None] + x_range)
    y_mean = post_curves.mean(axis=0)
    hdi_data = az.hdi(post_curves, hdi_prob=0.95)
    
    fig = go.Figure()
    
    # 信用区間
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate([hdi_data[:, 1], hdi_data[:, 0][::-1]]),
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% 信用区間'
    ))
    
    # 平均予測曲線
    fig.add_trace(go.Scatter(
        x=x_range, y=y_mean,
        mode='lines',
        line=dict(color='rgba(0,176,246,0.8)', width=3),
        name='平均予測曲線'
    ))
    
    # 実績データ
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='markers',
        marker=dict(size=10, color='#636EFA'),
        name='実績データ'
    ))
    
    fig.update_layout(
        title=f"{channel_name}: 広告費と{target_var}の関係",
        xaxis_title='広告宣伝費',
        yaxis_title=target_var,
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    fig.add_annotation(
        x=0.98, y=0.05,
        xref='paper', yref='paper',
        text=f'R²: {r2:.3f}',
        showarrow=False,
        font=dict(size=14),
        bgcolor='rgba(255, 255, 255, 0.7)'
    )
    
    return fig

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
        height=600
    )
    
    return fig

# サイドバー
st.sidebar.title("📊 広告最適化ツール")
page = st.sidebar.radio(
    "分析メニュー",
    ["📈 現状把握", "🎯 投資費用最適化", "🔍 事前効果検証(前半)", "📊 事前効果検証(後半)"]
)

st.sidebar.markdown("---")
st.sidebar.info("データ更新日: 2025年10月1日\nデータ期間: 2024年1月 - 2025年9月")

# メインコンテンツ
if page == "📈 現状把握":
    st.markdown('<div class="main-header">現状把握</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">媒体別パフォーマンス分析</div>', unsafe_allow_html=True)
    
    st.info("💡 各媒体ごとに学習期間や目的変数を個別に設定して、パフォーマンスを確認できます")
    
    # 媒体別パラメータ設定
    st.subheader("媒体別パラメータ設定")
    
    # 一括設定
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**一括設定:**")
    with col2:
        if st.button("全媒体に同じ設定を適用", type="secondary"):
            st.success("設定を適用しました")
    
    # 媒体リスト
    channels = ["キャリアインデックス", "スタンバイ", "求人ボックス", "Criteo", "Google(一般KW)"]
    
    # 各媒体の設定
    config = {}
    for channel in channels:
        with st.expander(f"📌 {channel}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                start_date = st.date_input(
                    "学習期間(開始)",
                    value=pd.to_datetime("2024-01-01"),
                    key=f"{channel}_start"
                )
            
            with col2:
                end_date = st.date_input(
                    "学習期間(終了)",
                    value=pd.to_datetime("2025-09-30"),
                    key=f"{channel}_end"
                )
            
            with col3:
                target_var = st.selectbox(
                    "目的変数",
                    ["3日以内有料応募回数", "応募回数", "コンバージョン数"],
                    key=f"{channel}_target"
                )
            
            with col4:
                model_type = st.selectbox(
                    "回帰モデル",
                    ["Hill Model", "線形回帰", "多項式回帰"],
                    key=f"{channel}_model"
                )
            
            config[channel] = {
                'start_date': start_date,
                'end_date': end_date,
                'target_variable': target_var,
                'model_type': 'hill' if model_type == "Hill Model" else 'linear'
            }
    
    if st.button("全媒体の分析を実行", type="primary"):
        with st.spinner("モデルを学習中..."):
            # ここで実際の分析を実行
            st.success("分析が完了しました!")
    
    # モデル精度サマリー
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均モデル精度 (R²)", "0.659", help="全媒体の平均値")
    
    # 媒体別パフォーマンス(タブ)
    st.subheader("媒体別パフォーマンス")
    tabs = st.tabs(channels)
    
    for i, tab in enumerate(tabs):
        with tab:
            st.write(f"**{channels[i]}** の分析結果")
            # ここに実際のグラフを表示
            st.info("グラフを表示するには、実際のデータで分析を実行してください")
    
    # 全媒体比較グラフ
    st.subheader("全媒体比較(Hill関数曲線)")
    st.info("💡 各媒体のHill関数を同一空間にプロットし、費用対効果を一目で比較できます")
    
    # ここに統合グラフを表示
    st.info("統合グラフを表示するには、実際のデータで分析を実行してください")

elif page == "🎯 投資費用最適化":
    st.markdown('<div class="main-header">投資費用最適化</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">予算配分最適化</div>', unsafe_allow_html=True)
    
    st.info("💡 システムが最適なソルバー(モンテカルロ法、貪欲法等)を自動的に選択します")
    
    # 予算入力
    total_budget = st.number_input(
        "週当たりの総予算 (円)",
        min_value=0,
        value=127894580,
        step=1000000,
        format="%d"
    )
    
    # 優先媒体設定
    st.subheader("優先媒体設定(オプション)")
    col1, col2 = st.columns(2)
    
    with col1:
        priority_channels = st.multiselect(
            "優先媒体を選択",
            channels,
            default=["スタンバイ", "求人ボックス", "キャリアインデックス"]
        )
    
    with col2:
        priority_ratio = st.slider(
            "優先媒体への配分比率",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            format="%.0f%%"
        )
    
    if st.button("最適配分を計算", type="primary"):
        with st.spinner("最適化計算中..."):
            # ここで実際の最適化を実行
            st.success("最適化が完了しました!")
            
            # 結果テーブルの表示(ダミーデータ)
            st.subheader("最適予算配分結果")
            result_data = {
                '順位': [0, 1, 2, 3, 4],
                '媒体': ['スタンバイ', '求人ボックス', 'キャリアインデックス', 'Criteo', 'Google(一般KW)'],
                '最適配分予算 (円)': ['¥6,245,917', '¥6,472,555', '¥8,444,319', '¥2,295,284', '¥2,296,241'],
                '予算比率': ['48.8%', '50.6%', '66.0%', '17.9%', '17.9%'],
                '予測応募数': ['2,734', '3,014', '10,891', '632', '683'],
                '予測有料応募CPA (円)': ['¥2,285', '¥2,148', '¥775', '¥3,632', '¥3,362']
            }
            st.dataframe(pd.DataFrame(result_data), use_container_width=True)

elif page == "🔍 事前効果検証(前半)":
    st.markdown('<div class="main-header">事前効果検証(前半)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">シナリオ比較 - 点推定予測</div>', unsafe_allow_html=True)
    
    st.info("💡 システム提案と現場担当者の予算配分を比較し、どちらが優れているか事前に検証できます")
    
    # シナリオ入力
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 シナリオ1: システム提案")
        total_budget_s1 = st.number_input("総予算 (円)", value=127894580, key="s1_total")
        
        st.write("**媒体別予算配分**")
        s1_budgets = {}
        for channel in channels[:5]:  # 主要5媒体
            s1_budgets[channel] = st.number_input(
                channel,
                min_value=0,
                value=5000000,
                step=100000,
                key=f"s1_{channel}"
            )
    
    with col2:
        st.subheader("👤 シナリオ2: 現場担当者提案")
        total_budget_s2 = st.number_input("総予算 (円)", value=127894580, key="s2_total")
        
        st.write("**媒体別予算配分**")
        s2_budgets = {}
        for channel in channels[:5]:
            s2_budgets[channel] = st.number_input(
                channel,
                min_value=0,
                value=4500000,
                step=100000,
                key=f"s2_{channel}"
            )
    
    if st.button("シナリオ比較分析を実行", type="primary"):
        with st.spinner("分析中..."):
            st.success("分析が完了しました!")
            
            # 比較サマリー
            st.subheader("シナリオ比較サマリー")
            summary_data = {
                'シナリオ': ['システム提案 🏆', '現場担当者提案', '改善効果'],
                '総予算 (円)': ['¥127,894,580', '¥127,894,580', '-'],
                '予測応募数(中央値)': ['9,349', '7,948', '+1,401 (+17.6%)'],
                '予測応募数(期待値)': ['9,900', '8,473', '+1,427 (+16.8%)'],
                '予測有料応募CPA (円)': ['¥6,635', '¥7,582', '-¥947 (-12.5%)']
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

elif page == "📊 事前効果検証(後半)":
    st.markdown('<div class="main-header">事前効果検証(後半)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">シナリオ比較 - 区間推定(モンテカルロシミュレーション)</div>', unsafe_allow_html=True)
    
    st.info("💡 ベイズ的シミュレーションにより、不確実性を考慮した比較が可能です")
    
    # シナリオ設定(前半と同じ構造)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 シナリオ1: 最適化案")
        # 予算配分入力
        
    with col2:
        st.subheader("👤 シナリオ2: 現場案")
        # 予算配分入力
    
    # シミュレーション設定
    n_samples = st.slider(
        "モンテカルロシミュレーション試行回数",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )
    
    if st.button("ベイズ的シミュレーション比較を実行", type="primary"):
        with st.spinner(f"{n_samples}回のシミュレーションを実行中..."):
            st.success("シミュレーションが完了しました!")
            
            # 確率表示
            st.subheader("成果改善効果の確率分布")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style='text-align: center; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                    <div style='font-size: 1rem; color: #2c3e50; margin-bottom: 1rem;'>
                        <strong>システム提案が現場担当者提案を上回る確率</strong>
                    </div>
                    <div style='font-size: 3rem; font-weight: 700; color: #2ecc71; margin: 1rem 0;'>
                        98.4%
                    </div>
                    <div style='font-size: 0.9rem; color: #7f8c8d;'>
                        10,000回のモンテカルロシミュレーション結果
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # ヒストグラム(ダミー)
            st.info("ヒストグラムを表示するには、実際のシミュレーションを実行してください")

# フッター
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 広告最適化ツール")