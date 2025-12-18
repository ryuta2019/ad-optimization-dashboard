# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★  
# ★★★ 設定ファイル ★★★  
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★  
# model_type: 'linear' はベイズ線形回帰モデルを意味します。  
analysis_config = {  
    "スタンバイ": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "linear"},  
    "求人ボックス": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "linear"},  
    "キャリアインデックス": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "linear"},  
    "Criteo": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "linear"}, # ★ベイズ線形回帰  
    "Googleリスティング（一般KW）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "linear"}, # ★ベイズ線形回帰  
    "Googleリスティング（インハウス/動的）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-03-01", "end_date": "2025-08-31", "model_type": "gam"},  
    "GDN（インハウス）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "gam"},  
    "Yahoo!リスティング（インハウス/動的）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "gam"},  
    "チャットブースト": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "gam"}, # ★ベイズ線形回帰  
}

# ==============================================================================  
# パート1: モデル学習とパラメータ抽出 (ベイズHill/ベイズLinear/GAM ハイブリッド版)  
# ==============================================================================  
def train_and_extract_models(df, config):  
    """  
    configで指定されたモデルタイプに応じて、各媒体のモデルを学習する関数。  
    - 'hill': ベイズ・ヒル関数モデル  
    - 'linear': ベイズ線形回帰モデル  
    - 'gam': GAM (一般化加法モデル)  
    """  
    print(f"{'='*30}\n パート1: モデル/データ抽出を開始 \n{'='*30}")  
    model_params = {}

    for channel_name, cfg in config.items():  
        model_type = cfg.get('model_type', 'gam')  
        print(f"\n--- 媒体「{channel_name}」の処理を開始 (モデル: {model_type}) ---")

        # データフィルタリング  
        start_date = pd.to_datetime(cfg['start_date'])  
        end_date = pd.to_datetime(cfg['end_date'])  
        filtered_df = df[(df['channel'] == channel_name) & (df['week_start_date'] >= start_date) & (df['week_start_date'] <= end_date)].copy()

        if len(filtered_df) < 5:  
            print(f"データ不足のため、媒体「{channel_name}」をスキップします。")  
            continue

        # pymcは1D配列、pygamは2D配列を要求するため、両方用意  
        x_pymc = filtered_df['total_spend'].values  
        y_pymc = filtered_df[cfg['target_variable']].values  
        X_gam = filtered_df[['total_spend']]  
        y_gam = filtered_df[cfg['target_variable']]

        try:  
            if model_type == 'hill':  
                print(f"ベイズ・ヒル関数モデルの学習を開始します。")  
                with pm.Model() as model:  
                    slope = pm.HalfNormal('slope', sigma=1)  
                    EC50 = pm.HalfNormal('EC50', sigma=np.median(x_pymc[x_pymc > 0]) if np.any(x_pymc > 0) else 10000)  
                    Vmax = pm.Deterministic('Vmax', slope * EC50)  
                    mu = Vmax * x_pymc / (EC50 + x_pymc)  
                    sigma = pm.HalfNormal('sigma', sigma=y_pymc.std())  
                    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_pymc)  
                    trace = pm.sample(20000, tune=1000, chains=4, cores=4, target_accept=0.9, progressbar=False, random_seed=42)

                summary = az.summary(trace, round_to=3)  
                if 'rhat' in summary.columns and (summary['rhat'] > 1.05).any():  
                    print(f"警告: 媒体「{channel_name}」で収束不良の可能性があります。")

                model_params[channel_name] = {  
                    'model_type': 'hill',  
                    'vmax_mean': trace.posterior['Vmax'].mean().item(),  
                    'ec50_mean': trace.posterior['EC50'].mean().item(),  
                    'min_spend': filtered_df['total_spend'].min(),  
                    'max_spend': filtered_df['total_spend'].max()  
                }

            # ★★★ 変更点1: 線形回帰をベイズモデルに置き換え ★★★  
            elif model_type == 'linear':  
                print(f"ベイズ線形回帰モデルの学習を開始します。")  
                with pm.Model() as linear_model:  
                    alpha = pm.Normal('alpha', mu=y_pymc.mean(), sigma=y_pymc.std() * 2) # 切片  
                    beta = pm.Normal('beta', mu=0, sigma=1) # 傾き  
                    mu = alpha + beta * x_pymc  
                    sigma = pm.HalfNormal('sigma', sigma=y_pymc.std())  
                    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_pymc)  
                    trace = pm.sample(20000, tune=1000, chains=4, cores=4, target_accept=0.9, progressbar=False, random_seed=42)

                summary = az.summary(trace, round_to=3)  
                if 'rhat' in summary.columns and (summary['rhat'] > 1.05).any():  
                    print(f"警告: 媒体「{channel_name}」で収束不良の可能性があります。")

                model_params[channel_name] = {  
                    'model_type': 'linear',  
                    'alpha_mean': trace.posterior['alpha'].mean().item(),  
                    'beta_mean': trace.posterior['beta'].mean().item(),  
                    'min_spend': filtered_df['total_spend'].min(),  
                    'max_spend': filtered_df['total_spend'].max()  
                }

            elif model_type == 'gam':  
                if len(filtered_df) < 10:  
                    print(f"データ不足(10件未満)のため、GAMの学習をスキップします。")  
                    continue  
                print(f"GAMの学習を開始します。")  
                gam = LinearGAM(s(0, constraints='monotonic_inc')).fit(X_gam, y_gam)  
                model_params[channel_name] = {  
                    'model_type': 'gam',  
                    'gam_model': gam,  
                    'min_spend': filtered_df['total_spend'].min(),  
                    'max_spend': filtered_df['total_spend'].max()  
                }

            print(f"媒体「{channel_name}」の学習完了。")

        except Exception as e:  
            print(f"媒体「{channel_name}」の学習中にエラーが発生しました: {e}")

    print(f"\n{'='*30}\n パート1: モデル/データ抽出完了 \n{'='*30}")  
    return model_params


# ==============================================================================  
# パート2: 数理最適化による予算配分 (ベイズ/GAM ハイブリッド版)  
# ==============================================================================  
def optimize_budget_allocation(model_params, total_budget, priority_channels, priority_ratio, n_starts=50):  
    """  
    ベイズモデル(Hill/Linear)とGAMを使い分け、マルチスタート法で最適化を行う関数。  
    """  
    print(f"\n{'='*30}\n パート2: 予算最適化を開始 (マルチスタート法: {n_starts}回試行) \n{'='*30}")

    channels = list(model_params.keys())  
    n_channels = len(channels)

    # ★★★ 変更点2: 目的関数をベイズ線形回帰に対応 ★★★  
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
              
            # 成果がマイナスにならないように0でクリップ  
            total_revenue += max(0, revenue)  
        return -total_revenue

    # 制約、境界、マルチスタートのロジック (変更なし)  
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
    best_fun_history = []

    print(f"\n{n_starts}個のランダムな出発点から最適化を実行します...")  
    for i in range(n_starts):  
        if (i + 1) % (n_starts // 10 or 1) == 0:  
            print(f"--- 試行 {i + 1}/{n_starts} ---")  
        random_values = np.random.rand(n_channels)  
        initial_guess = (random_values / random_values.sum()) * total_budget  
        try:  
            result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False})  
            if result.success and result.fun < best_fun_value:  
                best_fun_value = result.fun  
                best_result = result  
        except Exception:  
            pass  
        best_fun_history.append(-best_fun_value if best_fun_value != float('inf') else 0)

    print(f"\n{'='*30}\n パート2: 予算最適化完了 \n{'='*30}")  
    if best_result is not None:  
        result = best_result  
        optimal_budgets = result.x

        # ★★★ 変更点3: 予測成果の計算部分をベイズ線形回帰に対応 ★★★  
        def get_predicted_revenue(budget, params):  
            if params['model_type'] == 'hill':  
                return params['vmax_mean'] * budget / (params['ec50_mean'] + budget + 1e-9)  
            elif params['model_type'] == 'linear':  
                return params['alpha_mean'] + params['beta_mean'] * budget  
            elif params['model_type'] == 'gam':  
                return params['gam_model'].predict(np.array([[budget]]))[0]  
            return 0

        results_df = pd.DataFrame({  
            '媒体': channels,  
            '最適配分予算': optimal_budgets,  
            '予測成果': [get_predicted_revenue(optimal_budgets[i], model_params[ch]) for i, ch in enumerate(channels)]  
        })  
        results_df['予算比率'] = results_df['最適配分予算'] / total_budget  
        results_df = results_df[['媒体', '最適配分予算', '予算比率', '予測成果']]  
        results_df['最適配分予算'] = results_df['最適配分予算'].apply(lambda x: f"{x:,.0f} 円")  
        results_df['予算比率'] = results_df['予算比率'].apply(lambda x: f"{x:.1%}")  
        results_df['予測成果'] = results_df['予測成果'].apply(lambda x: f"{max(0, x):,.1f}")

        print("\n>>> 最適化成功！ (全試行中のベスト結果)")  
        display(results_df)  
        total_allocated = np.sum(optimal_budgets)  
        priority_allocated = np.sum(optimal_budgets[priority_indices])  
        print("\n--- 検算 ---")  
        print(f"総予算: {total_budget:,.0f} 円")  
        print(f"配分された合計予算: {total_allocated:,.0f} 円")  
        if priority_channels:  
            print(f"主要媒体への配分合計: {priority_allocated:,.0f} 円 (目標: {total_budget * priority_ratio:,.0f} 円)")  
        print(f"予測される合計成果: {-result.fun:,.1f}")  
    else:  
        print("\n>>> 最適化失敗...")  
        print("全ての出発点から最適解を見つけることができませんでした。")

    return best_result, best_fun_history


# ==============================================================================  
# パート3: 実行と収束プロットの描画  
# ==============================================================================

# --- 最適化の条件を設定 ---  
TOTAL_WEEKLY_BUDGET = 30232558  
PRIORITY_CHANNELS = ["スタンバイ", "求人ボックス", "キャリアインデックス"]  
PRIORITY_BUDGET_RATIO = 0.70  
HILL_MODEL_CHANNELS = ["スタンバイ", "求人ボックス", "キャリアインデックス"]

# 1. モデル学習/データ抽出  
trained_models = train_and_extract_models(combined_df, analysis_config)

# 2. 最適化の実行  
if trained_models:  
    # ★★★ 変更点：履歴を受け取る ★★★  
    optimization_result, history = optimize_budget_allocation(  
        trained_models,  
        TOTAL_WEEKLY_BUDGET,  
        PRIORITY_CHANNELS,  
        PRIORITY_BUDGET_RATIO,  
        n_starts=3000 # 試行回数  
    )

    # ★★★ 変更点：収束プロットを描画 ★★★  
    if history:  
        plt.figure(figsize=(10, 6))  
        plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-')  
        plt.title('マルチスタート法の収束プロット', fontsize=16)  
        plt.xlabel('試行回数 (Number of Starts)', fontsize=12)  
        plt.ylabel('その時点で見つかった最良の合計成果', fontsize=12)  
        plt.grid(True, linestyle='--', alpha=0.6)  
        plt.xticks(range(0, len(history) + 1, 5))  
        plt.show()  
else:  
    print("モデル/データが一つも抽出されなかったため、最適化を実行できませんでした。")