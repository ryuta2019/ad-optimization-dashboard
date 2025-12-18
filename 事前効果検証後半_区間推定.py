# ==============================================================================  
# パート1: モデル学習と不確実性情報の抽出 (ベイズ/GAM ハイブリッド版)  
# ==============================================================================  
def train_and_extract_models_with_uncertainty(df, config, n_splines=15, lam=0.6, n_bootstraps=500):  
    """  
    configで指定されたモデルタイプに応じて、各媒体のモデルを学習し、不確実性情報を抽出する関数。  
    - 'hill': ベイズ・ヒル関数モデル  
    - 'linear': ベイズ線形回帰モデル  
    - 'gam': GAM (一般化加法モデル) + 手動ブートストラップ  
    """  
    print(f"{'='*30}\n パート1: モデル学習を開始 \n{'='*30}")  
    model_results = {}

    for channel_name, cfg in config.items():  
        model_type = cfg.get('model_type', 'gam')  
        print(f"\n--- 媒体「{channel_name}」のモデル({model_type})を学習中... ---")  
        start_date, end_date = pd.to_datetime(cfg['start_date']), pd.to_datetime(cfg['end_date'])  
        filtered_df = df[(df['channel'] == channel_name) & (df['week_start_date'] >= start_date) & (df['week_start_date'] <= end_date)].copy()

        if len(filtered_df) < 10:  
            print(f"データ不足のためスキップします。")  
            continue

        x_data, y_data = filtered_df['total_spend'].values, filtered_df[cfg['target_variable']].values

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
                model_results[channel_name] = {'model_type': 'hill', 'trace': trace}  
                print(f"学習完了。")

            # ★★★ 変更点1: ベイズ線形回帰モデルの学習ロジックを追加 ★★★  
            elif model_type == 'linear':  
                print(f"ベイズ線形回帰モデルの学習を開始します。")  
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
                  
                model_results[channel_name] = {  
                    'model_type': 'linear',  
                    'trace': trace # traceオブジェクトをそのまま保存  
                }  
                print(f"学習完了。")

            elif model_type == 'gam':  
                print(f"GAM学習と信頼区間計算を開始します。")  
                X_gam, y_gam = filtered_df[['total_spend']], filtered_df[cfg['target_variable']]  
                print(f"手動ブートストラップ法で信頼区間を計算中 (試行回数: {n_bootstraps})...")  
                spend_range = np.linspace(X_gam.min(), X_gam.max(), 100)  
                bootstrap_preds = []

                for i in tqdm(range(n_bootstraps), desc=f"Bootstrapping {channel_name}", leave=False):  
                    resampled_indices = np.random.choice(X_gam.index, size=len(X_gam), replace=True)  
                    X_boot, y_boot = X_gam.loc[resampled_indices], y_gam.loc[resampled_indices]  
                    gam_boot = LinearGAM(s(0, n_splines=n_splines, constraints='monotonic_inc'), lam=lam).fit(X_boot, y_boot)  
                    pred_boot = gam_boot.predict(spend_range.reshape(-1, 1))  
                    bootstrap_preds.append(pred_boot)

                bootstrap_preds = np.array(bootstrap_preds)  
                lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)  
                upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)  
                intervals = np.c_[lower_bound, upper_bound]  
                final_gam = LinearGAM(s(0, n_splines=n_splines, constraints='monotonic_inc'), lam=lam).fit(X_gam, y_gam)

                model_results[channel_name] = {  
                    'model_type': 'gam', 'gam_model': final_gam,  
                    'spend_range': spend_range, 'intervals': intervals  
                }  
                print(f"GAM学習と信頼区間計算が完了。")

        except Exception as e:  
            print(f"学習中にエラー: {e}")

    print(f"\n{'='*30}\n パート1: モデル学習完了 \n{'='*30}")  
    return model_results

# ==============================================================================  
# パート2: ベイズ的シミュレーションによるシナリオ比較  
# ==============================================================================  
def simulate_and_compare_scenarios(trained_models, scenario1_ratios, scenario2_ratios, total_budget, n_samples=10000):  
    print(f"\n{'='*30}\n パート2: ベイズ的シミュレーション比較レポート \n{'='*30}")  
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

    s1_total_revenues, s2_total_revenues = [], []  
    print(f"{n_samples}回のモンテカルロシミュレーションを実行します...")  
    for _ in tqdm(range(n_samples), desc="シミュレーション実行中"):  
        s1_run_revenue, s2_run_revenue = 0, 0  
        for ch in channels:  
            params = trained_models.get(ch)  
            if not params: continue

            s1_budget = scenario1_ratios.get(ch, 0) * total_budget  
            s2_budget = scenario2_ratios.get(ch, 0) * total_budget

            if params['model_type'] == 'hill':  
                trace = params['trace']  
                posterior_samples = az.extract(trace, num_samples=1)  
                vmax_sample, ec50_sample = posterior_samples['Vmax'].item(), posterior_samples['EC50'].item()  
                s1_run_revenue += vmax_sample * s1_budget / (ec50_sample + s1_budget + 1e-9)  
                s2_run_revenue += vmax_sample * s2_budget / (ec50_sample + s2_budget + 1e-9)

            # ★★★ 変更点2: シミュレーションにベイズ線形回帰のロジックを追加 ★★★  
            elif params['model_type'] == 'linear':  
                trace = params['trace']  
                posterior_samples = az.extract(trace, num_samples=1)  
                alpha_sample = posterior_samples['alpha'].item()  
                beta_sample = posterior_samples['beta'].item()  
                  
                rev1 = alpha_sample + beta_sample * s1_budget  
                s1_run_revenue += max(0, rev1) # 成果がマイナスにならないように  
                  
                rev2 = alpha_sample + beta_sample * s2_budget  
                s2_run_revenue += max(0, rev2) # 成果がマイナスにならないように

            elif params['model_type'] == 'gam':  
                gam_model, spend_range, intervals = params['gam_model'], params['spend_range'], params['intervals']  
                def get_gam_sample(budget):  
                    idx = np.argmin(np.abs(spend_range - budget))  
                    mean_pred = gam_model.predict(np.array([[budget]]))[0]  
                    lower, upper = intervals[idx, 0], intervals[idx, 1]  
                    std_dev = (upper - lower) / 4.0 # 95%区間を約4σと近似  
                    return np.random.normal(loc=mean_pred, scale=max(std_dev, 1e-9))  
                  
                s1_run_revenue += max(0, get_gam_sample(s1_budget))  
                s2_run_revenue += max(0, get_gam_sample(s2_budget))

        s1_total_revenues.append(s1_run_revenue)  
        s2_total_revenues.append(s2_run_revenue)

    s1_total_revenues, s2_total_revenues = np.array(s1_total_revenues), np.array(s2_total_revenues)

    print("\n--- シミュレーション結果 ---")  
    s1_expected, s1_median = np.mean(s1_total_revenues), np.median(s1_total_revenues)  
    s2_expected, s2_median = np.mean(s2_total_revenues), np.median(s2_total_revenues)  
    prob_s1_beats_s2 = np.mean(s1_total_revenues > s2_total_revenues)

    summary_df = pd.DataFrame({  
        '指標': ['予測成果の期待値 (平均)', '予測成果の中央値 (50%点)'],  
        'シナリオ1 (最適化案)': [s1_expected, s1_median],  
        'シナリオ2 (現場案)': [s2_expected, s2_median]  
    })  
    summary_df['シナリオ1 (最適化案)'] = summary_df['シナリオ1 (最適化案)'].apply(lambda x: f"{x:,.1f}")  
    summary_df['シナリオ2 (現場案)'] = summary_df['シナリオ2 (現場案)'].apply(lambda x: f"{x:,.1f}")  
    display(summary_df)

    print(f"\n>>> シナリオ1 (最適化案) の成果がシナリオ2 (現場案) を上回る確率: {prob_s1_beats_s2:.2%}")

    plt.figure(figsize=(10, 6))  
    plt.hist(s1_total_revenues, bins=50, alpha=0.7, label=f'シナリオ1 (最適化案)\n期待値: {s1_expected:,.0f}')  
    plt.hist(s2_total_revenues, bins=50, alpha=0.7, label=f'シナリオ2 (現場案)\n期待値: {s2_expected:,.0f}')  
    plt.axvline(s1_expected, color='blue', linestyle='--')  
    plt.axvline(s2_expected, color='orange', linestyle='--')  
    plt.title('合計成果の予測分布 (シミュレーション結果)', fontsize=16)  
    plt.xlabel('予測される合計成果', fontsize=12)  
    plt.ylabel('頻度', fontsize=12)  
    plt.legend()  
    plt.grid(True, linestyle='--', alpha=0.5)  
    plt.show()

# ==============================================================================  
# パート3: 実行とシナリオ設定  
# ==============================================================================

# ★★★ 1. 全体の総予算を設定 ★★★  
TOTAL_WEEKLY_BUDGET = 30232558

# ★★★ 2. 各媒体の学習条件を個別に設定 ★★★  
# ★★★ 変更点3: 'linear' モデルをconfigに追加 ★★★  
analysis_config = {  
    "スタンバイ": {"target_variable": "_10日以内有料応募回数", "start_date": "2024-01-01", "end_date": "2025-08-31", "model_type": "linear"},  
    "求人ボックス": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "linear"},  
    "キャリアインデックス": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "linear"},  
    "Criteo": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "linear"}, # ★変更例  
    "Googleリスティング（一般KW）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "gam"}, # ★変更例  
    "Googleリスティング（インハウス/動的）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-03-01", "end_date": "2025-08-31", "model_type": "gam"},  
    "GDN（インハウス）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "gam"},  
    "Yahoo!リスティング（インハウス/動的）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "gam"},  
    "チャットブースト": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "gam"}, # ★変更例  
}

# ★★★ 3. 比較したい2つのシナリオの配分比率を手動で入力 ★★★  
scenario1_allocation_ratios = {  
    "スタンバイ": 0.28, "求人ボックス": 0.22, "キャリアインデックス": 0.20, "Criteo": 0.04,  
    "Googleリスティング（一般KW）": 0.11, "Googleリスティング（インハウス/動的）": 0.05,  
    "GDN（インハウス）": 0.03, "Yahoo!リスティング（インハウス/動的）": 0.05, "チャットブースト": 0.02,  
}  
scenario2_allocation_ratios = {  
    "スタンバイ": 0.25, "求人ボックス": 0.25, "キャリアインデックス": 0.20, "Criteo": 0.05,  
    "Googleリスティング（一般KW）": 0.10, "Googleリスティング（インハウス/動的）": 0.05,  
    "GDN（インハウス）": 0.03, "Yahoo!リスティング（インハウス/動的）": 0.05, "チャットブースト": 0.02,  
}

# --- 実行プロセス ---  
# 1. モデル学習  
trained_models = train_and_extract_models_with_uncertainty(combined_df, analysis_config)

# 2. 2つのシナリオをシミュレーションで比較  
if trained_models:  
    simulate_and_compare_scenarios(  
        trained_models,  
        scenario1_allocation_ratios,  
        scenario2_allocation_ratios,  
        TOTAL_WEEKLY_BUDGET  
    )  
else:  
    print("モデルが一つも学習されなかったため、比較を実行できませんでした。") 
    