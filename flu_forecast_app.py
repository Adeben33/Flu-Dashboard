import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
import streamlit as st
import pydeck as pdk
import os
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

st.set_page_config(layout="wide")
st.title("🦠 Influenza in Canada")

# SEIRV Model section with improved error handling
try:
    from cmdstanpy import CmdStanModel, cmdstan_path
    
    # Debug information for CmdStan installation
    if not st.session_state.get('cmdstan_checked', False):
        try:
            st.sidebar.write(f"CmdStan path: {cmdstan_path()}")
        except:
            st.sidebar.warning("⚠️ CmdStan not installed properly")
        
        # Check if SEIRV model exists
        seirv_exists = os.path.exists('./seirv_model')
        seirv_executable = os.access('./seirv_model', os.X_OK) if seirv_exists else False
        st.sidebar.write(f"SEIRV model exists: {seirv_exists}")
        st.sidebar.write(f"SEIRV model is executable: {seirv_executable}")
        
        st.session_state['cmdstan_checked'] = True
        st.session_state['seirv_available'] = seirv_exists and seirv_executable
    
    # Only offer SEIRV if it appears to be available
    use_seirv = st.sidebar.checkbox("Use SEIRV Model", value=st.session_state.get('seirv_available', False),
                                  disabled=not st.session_state.get('seirv_available', False))
    
    if use_seirv:
        @st.cache_resource
        def load_seirv_model():
            return CmdStanModel(exe_file="./seirv_model")

        @st.cache_resource
        def run_seirv_sampling(_model, stan_data):
            return _model.sample(data=stan_data, chains=1, iter_sampling=1000, iter_warmup=500, 
                               seed=123, show_console=True)

        @st.cache_data
        def extract_predictions(fit):
            pred_cases = fit.stan_variable("pred_cases")
            median_pred = np.median(pred_cases, axis=0)
            ci95_low = np.percentile(pred_cases, 2.5, axis=0)
            ci95_high = np.percentile(pred_cases, 97.5, axis=0)
            return median_pred, ci95_low, ci95_high

        @st.cache_data
        def prepare_stan_data(observed):
            T = len(observed)
            time = np.arange(1, T + 1)
            return {
                "T": T,
                "ts": time,
                "t0": 0 - 1e-6,
                "initial_state": [40097761 - 110, 0, 0, 110, 0, 0],
                "parameter": [0.000000053, 0.33, 0.0004, 0.1, 0.001, 0.2, 0.14],
                "incidence": observed
            }
except ImportError:
    st.sidebar.warning("⚠️ CmdStanPy not installed. SEIRV model will not be available.")
    use_seirv = False

# Sidebar controls
st.sidebar.header("Filter")
region = st.sidebar.selectbox("Region", ["Canada (National)", "Ontario", "Quebec", "BC", "Alberta"])
season = st.sidebar.selectbox("Season", ["2023-2024", "2024-2025"])

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV (with surveillance data)", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Create Date from Surveillance Week and Year
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + df['Surveilaince Week'].astype(str) + '1', format='%G%V%u')
    df.set_index('Date', inplace=True)

    # Extract relevant columns
    df['Influenza A'] = df['A(H1N1)'] + df['A(H3N2)'] + df['A(Sub typed)']
    df['Influenza Total'] = df['Influenza A'] + df['Influenza B']

    ah1n1 = df['A(H1N1)']
    ah3n2 = df['A(H3N2)']
    bunsubtyped = df['A(Sub typed)']
    flu_b = df['Influenza B']
    influenza_a = df['Influenza A']
    influenza_total = df['Influenza Total']
    perc_a = df['Percent Positive A']
    perc_b = df['Percent Positive B']

    # KPIs
    total_cases = int(ah1n1.sum() + ah3n2.sum() + bunsubtyped.sum() + flu_b.sum())
    last_week = df.index.max()
    current_week_data = df.loc[df.index == last_week]
    weekly_cases = int(current_week_data[['A(H1N1)', 'A(H3N2)', 'A(Sub typed)', 'Influenza B']].sum(axis=1).values[0])

    avg_perc_a = round(perc_a.mean(), 2)
    avg_perc_b = round(perc_b.mean(), 2)
    latest_perc_a = round(current_week_data['Percent Positive A'].values[0], 2)
    latest_perc_b = round(current_week_data['Percent Positive B'].values[0], 2)

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Total Cases (Uploaded)", f"{total_cases:,}")
    kpi2.metric("New Cases (Most Recent Week)", f"{weekly_cases:,}")
    kpi3.metric("Avg. Positivity Rate A (%)", f"{avg_perc_a}%")
    kpi4.metric("Avg. Positivity Rate B (%)", f"{avg_perc_b}%")
    kpi5.metric("This Week's Positivity A (%)", f"{latest_perc_a}%")
    kpi6.metric("This Week's Positivity B (%)", f"{latest_perc_b}%")

    # Create layout for three plots
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Influenza Activity Overview")
        fig1, ax1 = plt.subplots(figsize=(11.5, 6), dpi=150)
        ax1.bar(df.index, ah1n1, label='A(H1N1)', color='firebrick')
        ax1.bar(df.index, ah3n2, bottom=ah1n1, label='A(H3N2)', color='lightcoral')
        ax1.bar(df.index, bunsubtyped, bottom=ah1n1 + ah3n2, label='A(Unsubtyped)', color='mistyrose', hatch='//')
        ax1.bar(df.index, flu_b, bottom=ah1n1 + ah3n2 + bunsubtyped, label='Influenza B', color='cornflowerblue')

        ax2 = ax1.twinx()
        ax2.plot(df.index, perc_a, label='percent positive A', color='darkred', linewidth=2)
        ax2.plot(df.index, perc_b, label='percent positive B', color='darkblue', linewidth=2)

        ax1.set_title("Strain Composition & Positivity Rate", fontsize=16)
        ax1.set_xlabel("Surveillance Week", fontsize=16)
        ax1.set_ylabel("Number of positive influenza tests", fontsize=14)
        ax2.set_ylabel("% Tests positive", fontsize=14)
        ax1.set_xticks(df.index[::2])
        ax1.set_xticklabels(df['Surveilaince Week'].iloc[::2], rotation=45, ha='right')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.legend(['A(H1N1)', 'A(H3N2)', 'A(Unsubtyped)', 'Influenza B'], loc='upper left')
        ax2.legend(loc='upper right')
        st.pyplot(fig1)

    with col2:
        st.markdown("### Distribution of Cases by Type")
        fig2, ax_box = plt.subplots(figsize=(12, 6), dpi=150)
        df_box = pd.DataFrame({
            'A(H1N1)': ah1n1,
            'A(H3N2)': ah3n2,
            'A(Unsubtyped)': bunsubtyped,
            'Influenza B': flu_b
        })
        df_box.plot(kind='box', ax=ax_box, patch_artist=True)
        ax_box.set_title("Boxplot of Influenza Cases (2023–2024)", fontsize=16)
        ax_box.set_xlabel("Influenza Strain", fontsize=16)
        ax_box.set_ylabel("Number of Cases", fontsize=14)
        ax_box.tick_params(axis='both', labelsize=14)
        ax_box.grid(False)
        st.pyplot(fig2)

   
    # --- Prepare for modeling ---
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-W' + df['Surveilaince Week'].astype(str) + '-6', format='%G-W%V-%u')
    df.set_index('Date', inplace=True)

    df['Cases'] = df['A(Sub typed)'] + df['A(H3N2)'] + df['A(H1N1)'] + df['Influenza B']
    df['Virus'] = 'Influenza'
    df_single = df[df['Virus'] == 'Influenza'].sort_index()
    df_weekly = df_single['Cases'].asfreq('W-SAT')
    df_weekly = df_weekly.interpolate(method='linear')

    # --- Fourier Terms ---
    def add_fourier_terms(df, period=52, order=2):
        t = np.arange(len(df))
        for k in range(1, order + 1):
            df[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
            df[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
        return df

    df_fourier = df_weekly.to_frame(name='Cases')
    df_fourier = add_fourier_terms(df_fourier.copy(), period=52, order=2)
    X = df_fourier[[col for col in df_fourier.columns if 'sin' in col or 'cos' in col]]
    X = sm.add_constant(X)
    y = df_fourier['Cases']
    model = sm.OLS(y, X).fit()

    pred_train = model.get_prediction(X)
    pred_train_summary = pred_train.summary_frame(alpha=0.05)
    df_fourier['Fitted'] = pred_train_summary['mean']
    df_fourier['Lower'] = pred_train_summary['mean_ci_lower']
    df_fourier['Upper'] = pred_train_summary['mean_ci_upper']

    # --- Fix Negative Fourier Fits ---
    df_fourier['Fitted'] = df_fourier['Fitted'].clip(lower=0)
    df_fourier['Lower'] = df_fourier['Lower'].clip(lower=0)
    df_fourier['Upper'] = df_fourier['Upper'].clip(lower=0)

    # --- Forecast ---
    future_index = pd.date_range(start=df_fourier.index[-1] + pd.Timedelta(weeks=1), periods=12, freq='W-SAT')
    df_future = pd.DataFrame(index=future_index)
    df_future = add_fourier_terms(df_future, period=52, order=2)
    df_future = sm.add_constant(df_future)

    pred_future = model.get_prediction(df_future)
    pred_future_summary = pred_future.summary_frame(alpha=0.05)
    df_future['Forecast'] = pred_future_summary['mean']
    df_future['Lower'] = pred_future_summary['mean_ci_lower']
    df_future['Upper'] = pred_future_summary['mean_ci_upper']

    # --- STL decomposition ---
    df_log = np.log(df_weekly + 1e-6)
    stl_log = STL(df_log, period=52)
    result_log = stl_log.fit()
    stl_fitted = np.exp(result_log.trend + result_log.seasonal)
    residuals = result_log.resid
    residual_std = np.std(residuals)
    stl_upper = stl_fitted * np.exp(1.96 * residual_std)
    stl_lower = stl_fitted * np.exp(-1.96 * residual_std)

    # --- SARIMA ---
    with st.spinner("Fitting SARIMA model..."):
        try:
            sarima_model = SARIMAX(df_weekly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            sarima_result = sarima_model.fit(disp=False)
            fitted_sarima = sarima_result.fittedvalues
            sarima_success = True
        except Exception as e:
            st.warning(f"SARIMA model fitting failed: {str(e)}")
            sarima_success = False
            fitted_sarima = pd.Series(index=df_weekly.index, data=np.nan)

    # Initialize columns for visualizations
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Statistical Forecasting")
        fig_total, ax_total = plt.subplots(figsize=(12, 6))
        ax_total.plot(df_weekly.index, df_weekly, label='Actual Cases', color='black', marker='o')
        ax_total.plot(df_fourier.index, df_fourier['Fitted'], label='Fourier Fit', color='red')
        ax_total.fill_between(df_fourier.index, df_fourier['Lower'], df_fourier['Upper'], color='red', alpha=0.2, label='95% CI (Fourier Fit)')
        ax_total.plot(df_weekly.index, stl_fitted, label='STL (log) Trend + Seasonality', color='green')
        
        if sarima_success:
            ax_total.plot(df_weekly.index, fitted_sarima, label='SARIMA Fit', color='blue', linestyle='--')
        
        ax_total.set_xticks(df.index[::2])
        ax_total.set_xticklabels(df['Surveilaince Week'].iloc[::2], rotation=45, ha='right', fontsize=10)
        ax_total.set_title('Flu Case Fitting with Fourier, STL, and SARIMA', fontsize=18)
        ax_total.set_xlabel('Surveillance Week', fontsize=16)
        ax_total.set_ylabel("Number of Cases", fontsize=14)
        ax_total.tick_params(axis='both', labelsize=14)
        ax_total.legend()
        ax_total.grid(False)
        plt.tight_layout()
        st.pyplot(fig_total)

    # SEIRV model if available and enabled
    if use_seirv:
        with col4:
            st.markdown("### Forecasting with SEIRV Model")
            
            observed = df['Influenza Total'].values
            dates = df.index
            weeks = df['Surveilaince Week'].values
            
            stan_data = prepare_stan_data(observed)
            
            try:
                with st.spinner("Running SEIRV model sampling (cached)..."):
                    seirv_model = load_seirv_model()
                    fit = run_seirv_sampling(seirv_model, stan_data)
                    median_pred, ci95_low, ci95_high = extract_predictions(fit)
                    
                    fig_seirv, ax_seirv = plt.subplots(figsize=(12, 6))
                    ax_seirv.fill_between(np.arange(len(dates)), ci95_low, ci95_high, color='lightblue', alpha=0.3)
                    ax_seirv.plot(np.arange(len(dates)), median_pred, label="Predicted Median (SEIRV)", color='blue')
                    ax_seirv.scatter(np.arange(len(dates)), observed, label="Observed", color='black', s=40)
                    ax_seirv.set_xticks(np.arange(0, len(dates), 2))
                    ax_seirv.set_xticklabels(weeks[::2], rotation=45, ha='right')
                    ax_seirv.set_xlabel('Surveillance Week', fontsize=16)
                    ax_seirv.set_ylabel("Number of Cases", fontsize=14)
                    ax_seirv.tick_params(axis='both', labelsize=14)
                    ax_seirv.set_title('Observed vs Predicted Influenza Incidence (SEIRV Model)', fontsize=18)
                    ax_seirv.legend()
                    ax_seirv.grid(False)
                    plt.tight_layout()
                    st.pyplot(fig_seirv)
            except Exception as e:
                st.error(f"SEIRV model error: {str(e)}")
                st.warning("Consider checking if the SEIRV model is properly compiled and if CmdStan is correctly installed.")
    else:
        with col4:
            st.markdown("### Forecast Visualization")
            fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax_forecast.plot(df_weekly.index, df_weekly, label='Historical Cases', color='black', marker='o')
            
            # Plot forecast
            ax_forecast.plot(df_future.index, df_future['Forecast'], label='Fourier Forecast', color='red')
            ax_forecast.fill_between(df_future.index, df_future['Lower'], df_future['Upper'], 
                                     color='red', alpha=0.2, label='95% CI')
            
            # Add STL forecast if desired
            # This would require implementing STL forecasting which is not in the original code
            
            # Format plot
            combined_index = df_weekly.index.union(df_future.index)
            ax_forecast.set_xticks(combined_index[::4])  # Show every 4th date for clarity
            date_labels = [d.strftime('%Y-%m-%d') for d in ax_forecast.get_xticks()]
            ax_forecast.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=10)
            
            ax_forecast.set_title('Influenza Forecast for Next 12 Weeks', fontsize=18)
            ax_forecast.set_xlabel('Date', fontsize=16)
            ax_forecast.set_ylabel("Forecast Cases", fontsize=14)
            ax_forecast.tick_params(axis='both', labelsize=14)
            ax_forecast.axvline(x=df_weekly.index[-1], linestyle='--', color='gray', 
                               label='Forecast Start')
            ax_forecast.legend()
            ax_forecast.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_forecast)
            
            st.info("SEIRV Model is disabled. Enable it in the sidebar if available.")
else:
    st.info("Please upload a CSV file with influenza surveillance data.")
    
    # Show sample format
    st.markdown("### Expected CSV Format")
    sample_data = pd.DataFrame({
        'Year': [2023, 2023, 2023, 2023],
        'Surveilaince Week': [40, 41, 42, 43],
        'A(H1N1)': [10, 15, 20, 25],
        'A(H3N2)': [15, 20, 25, 30],
        'A(Sub typed)': [5, 7, 10, 12],
        'Influenza B': [2, 3, 5, 8],
        'Percent Positive A': [0.5, 0.7, 1.0, 1.2],
        'Percent Positive B': [0.1, 0.2, 0.3, 0.4]
    })
    st.dataframe(sample_data)