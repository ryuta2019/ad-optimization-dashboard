# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★  
# ★★★ 設定ファイル ★★★  
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★  
# model_type に 'linear' を追加。分析したい媒体に合わせて 'hill' または 'linear' を指定してください。  
analysis_config = {  
    "スタンバイ": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "linear"},  
    "求人ボックス": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "linear"},  
    "キャリアインデックス": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-05-01", "end_date": "2025-08-31", "model_type": "linear"},  
    "Criteo": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "linear"}, # ★変更例: 線形回帰  
    "Googleリスティング（一般KW）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "hill"},  
    "Googleリスティング（インハウス/動的）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "linear"}, # ★変更例: 線形回帰  
    "GDN（インハウス）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "hill"},  
    "Yahoo!リスティング（インハウス/動的）": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "hill"},  
    "チャットブースト": {"target_variable": "_10日以内有料応募回数", "start_date": "2025-01-01", "end_date": "2025-08-31", "model_type": "linear"}, # ★変更例: 線形回帰  
}

# --- 線形回帰モデルを追加した分析関数 ---  
def run_configured_analysis_with_diagnostics(df, channel_name, config):  
    print(f"\n{'='*80}\n--- 媒体「{channel_name}」の分析を開始 (モデル: {config['model_type']}) ---\n{'='*80}")

    # 1. データフィルタリング  
    target_variable = config['target_variable']  
    start_date = pd.to_datetime(config['start_date'])  
    end_date = pd.to_datetime(config['end_date'])  
    model_type = config['model_type']  
    filtered_df = df[(df['channel'] == channel_name) & (df['week_start_date'] >= start_date) & (df['week_start_date'] <= end_date)].copy().sort_values('week_start_date')

    if len(filtered_df) < 5:  
        print(f"データ不足（{len(filtered_df)}件）。モデルを構築できません。")  
        return

    x_data = filtered_df['total_spend'].values  
    y_data = filtered_df[target_variable].values  
    dates = filtered_df['week_start_date'].values

    # 2. ベイズモデリング  
    try:  
        with pm.Model() as model:  
            if model_type == 'hill':  
                slope = pm.HalfNormal('slope', sigma=1)  
                EC50 = pm.HalfNormal('EC50', sigma=np.median(x_data[x_data > 0]) if np.any(x_data > 0) else 10000)  
                Vmax = pm.Deterministic('Vmax', slope * EC50)  
                mu = Vmax * x_data / (EC50 + x_data)  
            elif model_type == 'adstock_hill':  
                decay_rate = pm.Beta('decay_rate', alpha=2.0, beta=2.0)  
                def adstock_step(spend_t, adstock_tm1, decay):  
                    return spend_t + decay * adstock_tm1  
                adstock_results, _ = pytensor.scan(  
                    fn=adstock_step,  
                    sequences=[pytensor.tensor.as_tensor_variable(x_data)],  
                    outputs_info=[pytensor.tensor.as_tensor_variable(np.array(0.0, dtype=np.float64))],  
                    non_sequences=[decay_rate]  
                )  
                adstocked_spend = pm.Deterministic('adstocked_spend', adstock_results)  
                slope = pm.HalfNormal('slope', sigma=1)  
                EC50 = pm.HalfNormal('EC50', sigma=np.median(x_data[x_data > 0]) if np.any(x_data > 0) else 10000)  
                Vmax = pm.Deterministic('Vmax', slope * EC50)  
                mu = Vmax * adstocked_spend / (EC50 + adstocked_spend)  
            # ★★★ 変更点1: 線形回帰モデルの定義を追加 ★★★  
            elif model_type == 'linear':  
                alpha = pm.Normal('alpha', mu=y_data.mean(), sigma=y_data.std() * 2) # 切片  
                beta = pm.Normal('beta', mu=0, sigma=1) # 傾き  
                mu = alpha + beta * x_data

            sigma = pm.HalfNormal('sigma', sigma=y_data.std())  
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_data)  
            trace = pm.sample(10000, tune=1000, chains=4, cores=4, target_accept=0.95, progressbar=True, random_seed=42)

        # 3. 収束診断  
        print("\n[診断1] 収束サマリー (rhatの確認)")  
        rhat_check = False  
        try:  
            summary = az.summary(trace, round_to=3)  
            display(summary)  
            if (summary['rhat'] > 1.01).any():  
                rhat_check = True  
                print("\n★★★ 警告: rhat > 1.01 のパラメータがあります。サンプリングが収束していない可能性があります。 ★★★")  
            else:  
                print("\n>>> 全てのパラメータで rhat <= 1.01 であり、収束は良好と判断されます。")  
        except (KeyError, ValueError):  
            rhat_check = True  
            print("\n★★★ エラー: rhatの計算に失敗しました。発散が多すぎる可能性があります。 ★★★")

        print("\n[診断2] トレースプロットによる視覚的診断")  
        az.plot_trace(trace)  
        plt.show()

        # 4. 常にグラフを描画  
        print("\n[分析結果] 広告効果の可視化")

        if model_type == 'adstock_hill':  
            vmax_post = trace.posterior['Vmax'].values  
            ec50_post = trace.posterior['EC50'].values  
            adstock_post = trace.posterior['adstocked_spend'].values  
            vmax_mean = vmax_post.mean()  
            ec50_mean = ec50_post.mean()  
            adstock_mean = adstock_post.mean(axis=(0,1))  
            y_pred_mean = vmax_mean * adstock_mean / (ec50_mean + adstock_mean)  
            r2 = r2_score(y_data, y_pred_mean)

            fig1 = go.Figure()  
            fig1.add_trace(go.Scatter(x=dates, y=y_data, mode='lines+markers', name='実測値', line=dict(color='gray')))  
            fig1.add_trace(go.Scatter(x=dates, y=y_pred_mean, mode='lines', name='モデル予測値', line=dict(color='rgba(0,176,246,0.8)', width=3)))  
            title_text1 = f"<b>{channel_name}</b>: 実績値とモデル予測値の時系列比較<br>(モデル: {model_type.capitalize()})"  
            if rhat_check:  
                title_text1 += " <b style='color:red;'>[警告: 収束に問題あり]</b>"  
            fig1.update_layout(title=title_text1, xaxis_title='日付', yaxis_title=target_variable, legend_title='凡例', template='plotly_white', annotations=[dict(x=0.98, y=0.95, xref='paper', yref='paper', text=f'<b>R²: {r2:.3f}</b>', showarrow=False, font=dict(size=14), bgcolor='rgba(255, 255, 255, 0.7)')])  
            fig1.show()

            print("\n[分析結果] 推定されたパラメータの分布")  
            az.plot_posterior(trace, var_names=['decay_rate', 'Vmax', 'EC50'], round_to=3)  
            plt.suptitle(f'{channel_name}: パラメータ事後分布', y=1.02)  
            plt.show()

        # ★★★ 変更点2: 'hill' と 'linear' の可視化ロジックを分岐 ★★★  
        elif model_type == 'hill':  
            y_pred_for_r2 = trace.posterior['Vmax'].mean().item() * x_data / (trace.posterior['EC50'].mean().item() + x_data)  
            r2 = r2_score(y_data, y_pred_for_r2)

            x_range = np.linspace(0, x_data.max() * 1.1, 100)  
            post_curves = trace.posterior['Vmax'].values.flatten()[:, None] * x_range / (trace.posterior['EC50'].values.flatten()[:, None] + x_range)  
            y_mean_pred_curve = post_curves.mean(axis=0)  
            hdi_data = az.hdi(post_curves, hdi_prob=0.95)  
            y_hdi_lower, y_hdi_upper = hdi_data.T

            fig = go.Figure()  
            fig.add_trace(go.Scatter(x=np.concatenate([x_range, x_range[::-1]]), y=np.concatenate([y_hdi_upper, y_hdi_lower[::-1]]), fill='toself', fillcolor='rgba(0,176,246,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="none", name='95% 信用区間'))  
            fig.add_trace(go.Scatter(x=x_range, y=y_mean_pred_curve, mode='lines', line=dict(color='rgba(0,176,246,0.8)', width=3), name='平均予測曲線'))

            # (散布図プロットのロジックは共通)  
            filtered_df['year'] = filtered_df['week_start_date'].dt.year  
            latest_date = filtered_df['week_start_date'].max()  
            four_weeks_ago = latest_date - pd.to_timedelta(27, unit='d')  
            recent_data = filtered_df[filtered_df['week_start_date'] > four_weeks_ago]  
            past_data = filtered_df[filtered_df['week_start_date'] <= four_weeks_ago]  
            color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']  
            for i, year in enumerate(sorted(past_data['year'].unique())):  
                df_year = past_data[past_data['year'] == year]  
                fig.add_trace(go.Scatter(x=df_year['total_spend'], y=df_year[target_variable], mode='markers', marker=dict(symbol='circle', size=10, opacity=0.7, color=color_palette[i % len(color_palette)]), name=f'過去の実績 ({year})', customdata=df_year['week_start_date'].dt.strftime('%Y-%m-%d'), hovertemplate=f"<b>週開始日</b>: %{{customdata}}<br><b>広告費</b>: %{{x:,.0f}}<br><b>{target_variable}</b>: %{{y:,.0f}}<br><extra></extra>"))  
            if not recent_data.empty:  
                fig.add_trace(go.Scatter(x=recent_data['total_spend'], y=recent_data[target_variable], mode='markers', marker=dict(symbol='star', color='gold', size=15, line=dict(width=1, color='black')), name='直近4週間の実績', customdata=recent_data['week_start_date'].dt.strftime('%Y-%m-%d'), hovertemplate=f"<b>週開始日</b>: %{{customdata}}<br><b>広告費</b>: %{{x:,.0f}}<br><b>{target_variable}</b>: %{{y:,.0f}}<br><extra></extra>"))

            title_text = f"<b>{channel_name}</b>: 広告費と{target_variable}の関係<br>(モデル: {model_type.capitalize()})"  
            if rhat_check:  
                title_text += " <b style='color:red;'>[警告: 収束に問題あり]</b>"  
            fig.update_layout(title=title_text, xaxis_title='広告宣伝費', yaxis_title=target_variable, legend_title='凡例', hovermode='x unified', template='plotly_white', annotations=[dict(x=0.98, y=0.05, xref='paper', yref='paper', text=f'<b>R²: {r2:.3f}</b>', showarrow=False, font=dict(size=14), bgcolor='rgba(255, 255, 255, 0.7)')])  
            fig.show()

        # ★★★ 変更点3: 線形回帰モデル用の可視化ロジックを追加 ★★★  
        elif model_type == 'linear':  
            # R2スコアの計算  
            alpha_mean = trace.posterior['alpha'].mean().item()  
            beta_mean = trace.posterior['beta'].mean().item()  
            y_pred_for_r2 = alpha_mean + beta_mean * x_data  
            r2 = r2_score(y_data, y_pred_for_r2)

            # 予測直線と信用区間の計算  
            x_range = np.linspace(0, x_data.max() * 1.1, 100)  
            alpha_post = trace.posterior['alpha'].values.flatten()  
            beta_post = trace.posterior['beta'].values.flatten()  
            post_curves = alpha_post[:, None] + beta_post[:, None] * x_range  
            y_mean_pred_curve = post_curves.mean(axis=0)  
            hdi_data = az.hdi(post_curves, hdi_prob=0.95)  
            y_hdi_lower, y_hdi_upper = hdi_data.T

            # 可視化  
            fig = go.Figure()  
            fig.add_trace(go.Scatter(x=np.concatenate([x_range, x_range[::-1]]), y=np.concatenate([y_hdi_upper, y_hdi_lower[::-1]]), fill='toself', fillcolor='rgba(0,176,246,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="none", name='95% 信用区間'))  
            fig.add_trace(go.Scatter(x=x_range, y=y_mean_pred_curve, mode='lines', line=dict(color='rgba(0,176,246,0.8)', width=3), name='平均予測線'))

            # (散布図プロットのロジックは共通)  
            filtered_df['year'] = filtered_df['week_start_date'].dt.year  
            latest_date = filtered_df['week_start_date'].max()  
            four_weeks_ago = latest_date - pd.to_timedelta(27, unit='d')  
            recent_data = filtered_df[filtered_df['week_start_date'] > four_weeks_ago]  
            past_data = filtered_df[filtered_df['week_start_date'] <= four_weeks_ago]  
            color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']  
            for i, year in enumerate(sorted(past_data['year'].unique())):  
                df_year = past_data[past_data['year'] == year]  
                fig.add_trace(go.Scatter(x=df_year['total_spend'], y=df_year[target_variable], mode='markers', marker=dict(symbol='circle', size=10, opacity=0.7, color=color_palette[i % len(color_palette)]), name=f'過去の実績 ({year})', customdata=df_year['week_start_date'].dt.strftime('%Y-%m-%d'), hovertemplate=f"<b>週開始日</b>: %{{customdata}}<br><b>広告費</b>: %{{x:,.0f}}<br><b>{target_variable}</b>: %{{y:,.0f}}<br><extra></extra>"))  
            if not recent_data.empty:  
                fig.add_trace(go.Scatter(x=recent_data['total_spend'], y=recent_data[target_variable], mode='markers', marker=dict(symbol='star', color='gold', size=15, line=dict(width=1, color='black')), name='直近4週間の実績', customdata=recent_data['week_start_date'].dt.strftime('%Y-%m-%d'), hovertemplate=f"<b>週開始日</b>: %{{customdata}}<br><b>広告費</b>: %{{x:,.0f}}<br><b>{target_variable}</b>: %{{y:,.0f}}<br><extra></extra>"))

            title_text = f"<b>{channel_name}</b>: 広告費と{target_variable}の関係<br>(モデル: {model_type.capitalize()})"  
            if rhat_check:  
                title_text += " <b style='color:red;'>[警告: 収束に問題あり]</b>"  
            fig.update_layout(title=title_text, xaxis_title='広告宣伝費', yaxis_title=target_variable, legend_title='凡例', hovermode='x unified', template='plotly_white', annotations=[dict(x=0.98, y=0.05, xref='paper', yref='paper', text=f'<b>R²: {r2:.3f}</b>', showarrow=False, font=dict(size=14), bgcolor='rgba(255, 255, 255, 0.7)')])  
            fig.show()

            # パラメータの事後分布プロット  
            print("\n[分析結果] 推定されたパラメータの分布")  
            az.plot_posterior(trace, var_names=['alpha', 'beta'], round_to=3)  
            plt.suptitle(f'{channel_name}: パラメータ事後分布 (alpha: 切片, beta: 傾き)', y=1.02)  
            plt.show()


        print(f"--- グラフ描画完了 ---")

    except Exception as e:  
        import traceback  
        print(f"分析中に予期せぬエラーが発生しました: {e}")  
        traceback.print_exc()

# --- メイン処理 ---  
# (変更なし)  
for channel_name, config in analysis_config.items():  
    run_configured_analysis_with_diagnostics(  
        df=combined_df,  
        channel_name=channel_name,  
        config=config  
    )
            