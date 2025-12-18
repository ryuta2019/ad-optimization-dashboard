# ==============================================================================  
# パート1: モデル学習とパラメータ抽出  
# ==============================================================================  
def train_and_extract_models(df, config):  
    """  
    configで指定されたモデルタイプに応じて、各媒体のモデルを学習する関数。  
    - 'hill': ベイズ・ヒル関数モデル  
    - 'linear': ベイズ線形回帰モデル  
    - 'gam': GAM (一般化加法モデル)  
    """  
    print(f"{'='*30}\n パート1: モデル学習を開始 \n{'='*30}")  
    model_params = {}  
    for channel_name, cfg in config.items():  
        model_type = cfg.get('model_type', 'gam')  
        print(f"\n--- 媒体「{channel_name}」のモデル({model_type})を学習中... ---")  
        start_date, end_date = pd.to_datetime(cfg['start_date']), pd.to_datetime(cfg['end_date'])  
        filtered_df = df[(df['channel'] == channel_name) & (df['week_start_date'] >= start_date) & (df['week_start_date'] <= end_date)].copy()

        if len(filtered_df) < 10:  
            print(f"データ不足のためスキップします。")  
            continue

        x_data = filtered_df['total_spend'].values  
        y_data = filtered_df[cfg['target_variable']].values

        try:  
            if model_type == 'hill':  
                with pm.Model() as model:  
                    slope = pm.HalfNormal('slope', sigma=1)  
                    EC50 = pm.HalfNormal('EC50', sigma=np.median(x_data[x_data > 0]) if np.any(x_data > 0) else 10000)  
                    Vmax = pm.Deterministic('Vmax', slope * EC50)  
                    mu = Vmax * x_data / (EC50 + x_data)  
                    sigma = pm.HalfNormal('sigma', sigma=y_data.std())  
                    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)  
                    trace = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=0.9, progressbar=False, random_seed=42)  
                summary = az.summary(trace, round_to=3)  
                if 'rhat' in summary.columns and (summary['rhat'] > 1.05).any():  
                    print(f"警告: 収束不良の可能性があります。")  
                model_params[channel_name] = {  
                    'model_type': 'hill',  
                    'vmax_mean': trace.posterior['Vmax'].mean().item(),  
                    'ec50_mean': trace.posterior['EC50'].mean().item()  
                }  
                print(f"学習完了。")

            # ★★★ 変更点1: ベイズ線形回帰モデルの学習ロジックを追加 ★★★  
            elif model_type == 'linear':  
                with pm.Model() as linear_model:  
                    alpha = pm.Normal('alpha', mu=y_data.mean(), sigma=y_data.std() * 2) # 切片  
                    beta = pm.Normal('beta', mu=0, sigma=1) # 傾き  
                    mu = alpha + beta * x_data  
                    sigma = pm.HalfNormal('sigma', sigma=y_data.std())  
                    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)  
                    trace = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=0.9, progressbar=False, random_seed=42)  
                  
                summary = az.summary(trace, round_to=3)  
                if 'rhat' in summary.columns and (summary['rhat'] > 1.05).any():  
                    print(f"警告: 収束不良の可能性があります。")  
                  
                model_params[channel_name] = {  
                    'model_type': 'linear',  
                    'alpha_mean': trace.posterior['alpha'].mean().item(),  
                    'beta_mean': trace.posterior['beta'].mean().item()  
                }  
                print(f"ベイズ線形回帰の学習完了。")

            elif model_type == 'gam':  
                X_gam, y_gam = filtered_df[['total_spend']], filtered_df[cfg['target_variable']]  
                gam = LinearGAM(s(0, constraints='monotonic_inc')).fit(X_gam, y_gam)  
                model_params[channel_name] = {  
                    'model_type': 'gam', 'gam_model': gam  
                }  
                print(f"GAM学習完了。")

        except Exception as e:  
            print(f"学習中にエラー: {e}")

    print(f"\n{'='*30}\n パート1: モデル学習完了 \n{'='*30}")  
    return model_params

# ==============================================================================  
# パート2: 2つのシナリオの比較シミュレーション  
# ==============================================================================  
def compare_scenarios(trained_models, scenario1_ratios, scenario2_ratios, total_budget):  
    print(f"\n{'='*30}\n パート2: シナリオ比較レポート \n{'='*30}")  
    if not trained_models:  
        print("モデルが学習されていないため、比較できません。")  
        return

    channels = list(trained_models.keys())

    def normalize_ratios(ratios):  
        total_ratio = sum(ratios.get(ch, 0) for ch in channels)  
        if not np.isclose(total_ratio, 1.0) and total_ratio > 0:  
            print(f"警告: 配分比率の合計が {total_ratio:.2%} のため、100%になるよう正規化します。")  
            return {ch: ratio / total_ratio for ch, ratio in ratios.items()}  
        return ratios

    scenario1_ratios = normalize_ratios(scenario1_ratios)  
    scenario2_ratios = normalize_ratios(scenario2_ratios)

    # ★★★ 変更点2: 予測成果の計算部分に 'linear' モデルを追加 ★★★  
    def get_predicted_revenue(budget, params):  
        if params['model_type'] == 'hill':  
            return params['vmax_mean'] * budget / (params['ec50_mean'] + budget + 1e-9)  
        elif params['model_type'] == 'linear':  
            return params['alpha_mean'] + params['beta_mean'] * budget  
        elif params['model_type'] == 'gam':  
            return params['gam_model'].predict(np.array([[budget]]))[0]  
        return 0

    comparison_data = []  
    for ch in channels:  
        params = trained_models.get(ch)  
        if not params: continue # モデルが学習されなかった媒体はスキップ

        s1_budget = scenario1_ratios.get(ch, 0) * total_budget  
        s2_budget = scenario2_ratios.get(ch, 0) * total_budget  
        s1_revenue = get_predicted_revenue(s1_budget, params)  
        s2_revenue = get_predicted_revenue(s2_budget, params)  
          
        # 成果がマイナスにならないようにクリップ  
        s1_revenue = max(0, s1_revenue)  
        s2_revenue = max(0, s2_revenue)

        comparison_data.append({  
            '媒体': ch, 'シナリオ1_予算': s1_budget, 'シナリオ2_予算': s2_budget,  
            'シナリオ1_予測成果': s1_revenue, 'シナリオ2_予測成果': s2_revenue,  
            'シナリオ1_予測CPA': s1_budget / s1_revenue if s1_revenue > 0 else np.inf,  
            'シナリオ2_予測CPA': s2_budget / s2_revenue if s2_revenue > 0 else np.inf,  
        })

    comp_df = pd.DataFrame(comparison_data)  
    if comp_df.empty:  
        print("比較データがありません。")  
        return  
          
    total_row = pd.DataFrame([{'媒体': '【全体合計】',  
        'シナリオ1_予算': comp_df['シナリオ1_予算'].sum(), 'シナリオ2_予算': comp_df['シナリオ2_予算'].sum(),  
        'シナリオ1_予測成果': comp_df['シナリオ1_予測成果'].sum(), 'シナリオ2_予測成果': comp_df['シナリオ2_予測成果'].sum(),  
        'シナリオ1_予測CPA': comp_df['シナリオ1_予算'].sum() / comp_df['シナリオ1_予測成果'].sum() if comp_df['シナリオ1_予測成果'].sum() > 0 else np.inf,  
        'シナリオ2_予測CPA': comp_df['シナリオ2_予算'].sum() / comp_df['シナリオ2_予測成果'].sum() if comp_df['シナリオ2_予測成果'].sum() > 0 else np.inf,  
    }])  
    comp_df = pd.concat([comp_df, total_row], ignore_index=True)  
    comp_df['成果の差(S1-S2)'] = comp_df['シナリオ1_予測成果'] - comp_df['シナリオ2_予測成果']  
    comp_df['成果改善率'] = (comp_df['成果の差(S1-S2)'] / comp_df['シナリオ2_予測成果']).replace([np.inf, -np.inf], np.nan)

    display_df = comp_df.copy()  
    for col in ['シナリオ1_予算', 'シナリオ2_予算']: display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f} 円")  
    for col in ['シナリオ1_予測成果', 'シナリオ2_予測成果', '成果の差(S1-S2)']: display_df[col] = display_df[col].apply(lambda x: f"{x:,.1f}")  
    for col in ['シナリオ1_予測CPA', 'シナリオ2_予測CPA']: display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f} 円" if np.isfinite(x) else "N/A")  
    display_df['成果改善率'] = display_df['成果改善率'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "N/A")

    display(display_df[['媒体', 'シナリオ1_予算', 'シナリオ2_予算', 'シナリオ1_予測成果', 'シナリオ2_予測成果', '成果の差(S1-S2)', '成果改善率', 'シナリオ1_予測CPA', 'シナリオ2_予測CPA']])

# ==============================================================================  
# パート3: 実行とシナリオ設定  
# ==============================================================================

# ★★★ 1. 全体の総予算を設定 ★★★  
TOTAL_WEEKLY_BUDGET = 30232558

# ★★★ 2. 各媒体の学習条件を個別に設定 ★★★  
# ★★★ 変更点3: 'linear' モデルをconfigに追加 ★★★  
analysis_config = {  
    "スタンバイ": {"target_variable": "_10日以内有料応募回数", "start_date": "2024-01-01", "end_date": "2025-08-31", "model_type": "hill"},  
    "求人ボックス": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "hill"},  
    "キャリアインデックス": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "hill"},  
    "Criteo": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "linear"}, # ★変更例  
    "Googleリスティング（一般KW）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "linear"}, # ★変更例  
    "Googleリスティング（インハウス/動的）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-03-01", "end_date": "2025-08-31", "model_type": "gam"},  
    "GDN（インハウス）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "gam"},  
    "Yahoo!リスティング（インハウス/動的）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "gam"},  
    "チャットブースト": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "linear"}, # ★変更例  
}

# ★★★ 3. 比較したい2つのシナリオの配分比率を手動で入力 ★★★

# 【シナリオ1】最適化案の配分比率  
scenario1_allocation_ratios = {  
    "スタンバイ": 0.28,  
    "求人ボックス": 0.22,  
    "キャリアインデックス": 0.20,  
    "Criteo": 0.04,  
    "Googleリスティング（一般KW）": 0.11,  
    "Googleリスティング（インハウス/動的）": 0.05,  
    "GDN（インハウス）": 0.03,  
    "Yahoo!リスティング（インハウス/動的）": 0.05,  
    "チャットブースト": 0.02,  
}

# 【シナリオ2】現場案の配分比率  
scenario2_allocation_ratios = {  
    "スタンバイ": 0.25,  
    "求人ボックス": 0.25,  
    "キャリアインデックス": 0.20,  
    "Criteo": 0.05,  
    "Googleリスティング（一般KW）": 0.10,  
    "Googleリスティング（インハウス/動的）": 0.05,  
    "GDN（インハウス）": 0.03,  
    "Yahoo!リスティング（インハウス/動T）": 0.05,  
    "チャットブースト": 0.02,  
}

# --- 実行プロセス ---  
# 1. モデル学習  
trained_models = train_and_extract_models(combined_df, analysis_config)

# 2. 2つのシナリオを比較  
if trained_models:  
    compare_scenarios(  
        trained_models,  
        scenario1_allocation_ratios, # シナリオ1: 最適化案  
        scenario2_allocation_ratios, # シナリオ2: 現場案  
        TOTAL_WEEKLY_BUDGET  
    )  
else:  
    print("モデルが一つも学習されなかったため、比較を実行できませんでした。")  