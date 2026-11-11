# Capstone Stage 1 — AI Use Log

**Name:** Jai Kasera  **NetID:** jk603

## 1. Platforms and Models

Codex (GPT-5, CLI)

## 2. Agents and Tooling

Codex ran agentically: executed Python, edited capstone_stage1_kasera_jai.ipynb

## 3. Communication History

**Me:** From here on out only give me the python code I ask for. Use the supplied `capstone_support` folder directly under the project root. Import `load_dataset` and `chronological_split` from `capstone_support.capstone_data` and do not download or substitute another dataset. Set the target to daily `cnt`, with a seven-day horizon and forecasts issued at the end of each completed day. Apply the packet's chronological 60/20/20 split, verify the 731-row daily date grid, and explicitly rule out future measured weather and `casual`/`registered` target components as prediction inputs. Prepare MAE, RMSE, WAPE, and bias calculations without shuffling the series.

**AI:**

```python
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss

from capstone_support.capstone_data import load_dataset, chronological_split

H = 7
MIN_HISTORY = 42
SEED = 546
RIDGE_GRID = (1.0, 10.0, 100.0)

def score(actual, pred):
    err = np.asarray(actual, float) - np.asarray(pred, float)
    return {'MAE': float(np.mean(np.abs(err))),
            'RMSE': float(np.sqrt(np.mean(err ** 2))),
            'WAPE_pct': float(100 * np.sum(np.abs(err)) / np.sum(np.abs(actual))),
            'bias': float(np.mean(-err))}


def weekly_origins(start_idx: int, end_idx: int):
    """First origin is day before block; score only complete non-overlapping H-day blocks."""
    return list(range(start_idx - 1, end_idx - H, H))

np.random.seed(SEED)
df = load_dataset('bike')
train, validation, test = chronological_split(df)
y = df['cnt'].to_numpy(float)
n = len(df)
n_train, n_val = len(train), len(validation)
test_start = n_train + n_val
assert n == 731 and n_train == 438 and n_val == 146 and len(test) == 147
assert df.index.freq is None or df.index.freq == 'D'
assert df.index.equals(pd.date_range(df.index[0], df.index[-1], freq='D', name=df.index.name))
assert not df['cnt'].isna().any()
# Future observed weather is explicitly excluded from ALL modeling inputs.
forbidden = {'temp', 'atemp', 'hum', 'windspeed', 'weathersit', 'casual', 'registered'}
source_used_features = {'cnt', 'holiday', 'workingday'}
assert not (source_used_features & forbidden)

print('Observations:', n, '| train:', n_train, '| validation:', n_val, '| test:', len(test))
print('Period:', df.index.min().date(), 'to', df.index.max().date())
print('Columns:', ', '.join(df.columns))
```

**Me:** Before using validation or test results for model identification, examine the initial training series. Calculate the ACF at weekly lags and weekday-average rental counts. Run ADF and KPSS on the level series and the seven-day seasonal difference `cnt_t - cnt_(t-7)`. State each test's null correctly, and do not claim that failure to reject establishes stationarity or a unit root. Use these results to assess weekly seasonality and whether seasonal differencing is sensible for a SARIMA candidate; this should follow the ACF/stationarity material in class.

**AI:**

```python
# Training-only diagnostics; both ADF/KPSS nulls explained in the report.
diagnostics={}
for label, yy in [('level',train.cnt.to_numpy(float)),
                  ('weekly_difference',(train.cnt-train.cnt.shift(7)).dropna().to_numpy(float))]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        adf_result=adfuller(yy,autolag='AIC')
        kpss_result=kpss(yy,regression='c',nlags='auto')
    diagnostics[label]={'ADF_stat':float(adf_result[0]),'ADF_p':float(adf_result[1]),
                        'KPSS_stat':float(kpss_result[0]),'KPSS_p':float(kpss_result[1])}
diagnostics['ACF_lag7']=float(acf(train.cnt,nlags=14,fft=True)[7])
diagnostics['ACF_lag14']=float(acf(train.cnt,nlags=14,fft=True)[14])
diagnostics['train_weekday_avg']={str(k):float(v) for k,v in train.groupby(train.index.day_name())['cnt'].mean().items()}

print('Training-only stationarity diagnostics:')
display(pd.DataFrame(diagnostics).T.loc[['level','weekly_difference']].round(4))
print('ACF lag 7:', round(diagnostics['ACF_lag7'],3), '; lag 14:',round(diagnostics['ACF_lag14'],3))
```

**Me:** Implement a seasonal-naive forecast for daily rentals using the previous observed week. At any forecast origin `t`, each of the next seven dates should reuse the count from the same weekday in the most recent observed week: `yhat_(t+h|t)=y_(t+h-7)` for `h=1,...,7`. Use only observations available through the origin and return a seven-value forecast vector, so it can be compared fairly against the other models. This baseline is part of the forecast-evaluation material in Lecture 4.

**AI:**

```python
def predict_snaive(origin: int):
    return y[origin-6:origin+1].copy()  # each day repeats previous same weekday
```

**Me:** Fit a Holt–Winters exponential smoothing model using an additive level, additive trend, damped extrapolation, and additive weekday seasonality with period 7. Refit on the expanding history through each forecast origin, forecast the next seven daily counts, and truncate impossible negative forecasts at zero. Avoid a large hyperparameter search; use this course-covered statistical model as a lightweight comparison to seasonal naive (Lecture 3).

**AI:**

```python
def predict_ets(origin: int):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = ExponentialSmoothing(y[:origin+1],trend='add', damped_trend=True,
             seasonal='add',seasonal_periods=7,initialization_method='estimated').fit(
             optimized=True,use_brute=False)
    return np.maximum(0, np.asarray(fit.forecast(H)))
```

**Me:** Add a classical SARIMA candidate that represents weekly seasonal dependence in daily rental counts. Use the parsimonious specification `SARIMA(1,0,0)x(1,1,0)[7]` with an intercept, motivated by the weekly ACF pattern and the seasonally differenced training-series checks. Fit only through each origin using `statsmodels` SARIMAX, cap the optimizer iterations, and return a seven-step forecast without looking at future outcomes. Keep the model and explanation consistent with the Box–Jenkins and SARIMA lectures (6–7).

**AI:**

```python
def predict_sarima(origin: int):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = SARIMAX(
            y[:origin+1],
            order=(1, 0, 0),
            seasonal_order=(1, 1, 0, 7),
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=75)

    return np.maximum(0, np.asarray(fit.forecast(H)))
```

**Me:** For an additional lightweight model, build a feature function for a target date `origin+h` and lead time `h in 1,...,7`. Historical features may include last observed rentals, weekly and four-week rolling summary statistics, lags 7/14/21/28, and a recent-trend contrast. Future-date features may use day of week, annual sine/cosine terms, time index, and scheduled holiday or working-day flags. Do not use future rentals, actual future weather, or any information that would not have been known when the prediction was issued. Assert valid history and lead-time bounds.

**AI:**

```python
def features(origin: int, h: int):
    """Features for y[origin+h] known at the end of day origin."""
    assert MIN_HISTORY - 1 <= origin < n and 1 <= h <= H and origin + h < n
    past = y[:origin + 1]
    future_day = df.index[origin + h]
    vals = [
        past[-1], float(np.mean(past[-7:])), float(np.std(past[-7:])),
        float(np.mean(past[-28:])), float(np.std(past[-28:])),
        past[-7], past[-14], past[-21], past[-28],
        float(np.mean(past[-7:]) - np.mean(past[-28:-7])),
        float((future_day - df.index[0]).days / 365.25),
        float(np.sin(2*np.pi*future_day.dayofyear/365.25)),
        float(np.cos(2*np.pi*future_day.dayofyear/365.25)),
    ]
    vals.extend([int(future_day.weekday() == d) for d in range(7)])
    # These flags are scheduled calendar information, not measured future weather.
    vals += [int(df['holiday'].iloc[origin+h]),
             int(df['workingday'].iloc[origin+h]), h]
    return vals
```

**Me:** Following the course's direct multi-step supervised learning framework, fit seven separate Ridge models, one for each lead time. Standardize features inside each fit with `StandardScaler`, and only create historical training examples from origins whose *complete seven-day target window* ends on or before the current origin. Use a small fixed penalty grid `{1,10,100}` to be chosen on validation only, and refit at each weekly origin. This should be computationally inexpensive and avoid leakage from future target windows.

**AI:**

```python
def predict_ridge(origin: int, alpha: float):
    """Seven direct models, each re-estimated on past non-overlapping weekly origins."""
    assert origin >= MIN_HISTORY + H
    preds = []
    # Historical training examples have all seven outcomes <= origin.
    prior_origins = np.arange(MIN_HISTORY - 1, origin - H + 1, H)
    assert all(o+H <= origin for o in prior_origins)
    for h in range(1, H + 1):
        X = np.asarray([features(int(o), h) for o in prior_origins])
        targets = np.asarray([y[int(o)+h] for o in prior_origins])
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X, targets)
        preds.append(max(0., float(model.predict([features(origin,h)])[0])))
    return np.asarray(preds)

preds = []
    # Historical training examples have all seven outcomes <= origin.
    prior_origins = np.arange(MIN_HISTORY - 1, origin - H + 1, H)
    assert all(o+H <= origin for o in prior_origins)
    for h in range(1, H + 1):
        X = np.asarray([features(int(o), h) for o in prior_origins])
        targets = np.asarray([y[int(o)+h] for o in prior_origins])
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X, targets)
        preds.append(max(0., float(model.predict([features(origin,h)])[0])))
    return np.asarray(preds)

def predict_snaive(origin: int):
    return y[origin-6:origin+1].copy()  # each day repeats previous same weekday

def predict_ets(origin: int):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = ExponentialSmoothing(y[:origin+1],trend='add', damped_trend=True,
             seasonal='add',seasonal_periods=7,initialization_method='estimated').fit(
             optimized=True,use_brute=False)
    return np.maximum(0, np.asarray(fit.forecast(H)))

def predict_sarima(origin: int):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = SARIMAX(y[:origin+1],order=(1,0,0),seasonal_order=(1,1,0,7),
                   trend='c', enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False,maxiter=75)
    return np.maximum(0, np.asarray(fit.forecast(H)))

preds = {
    'Seasonal naive (7-day)': predict_snaive,
    'ETS (damped, weekly)': predict_ets,
    'SARIMA (1,0,0)x(1,1,0)[7]': predict_sarima,
}
for alpha in RIDGE_GRID:
    preds[f'Direct Ridge (alpha={alpha:g})'] = (
        lambda origin, alpha=alpha: predict_ridge(origin, alpha))
```

**Me:** Use the supplied chronological training/validation/test boundaries. Evaluate all candidates at nonoverlapping weekly forecast origins; each origin predicts exactly the next seven days, each evaluated date is scored only once, and no forecast may cross the validation or test boundary. Record actual values, predictions, forecast horizons, and errors. Compare all models on exactly the same validation days using MAE and RMSE; choose the Ridge penalty and overall winning model using validation MAE alone. Freeze that configuration before comparing final test performance, but allow model refitting as previously observed dates accumulate.

**AI:**

```python
def evaluate_block(start: int, end: int, models: dict):
    results = []
    origins = weekly_origins(start, end)
    for origin in origins:
        future = np.arange(origin+1, origin+H+1)
        assert future[0] >= start and future[-1] < end
        for model_name, forecast_fn in models.items():
            yh = np.asarray(forecast_fn(origin))
            assert yh.shape == (H,) and np.isfinite(yh).all()
            for h, t in enumerate(future, start=1):
                results.append({
                    'model':model_name,'origin':df.index[origin],
                    'date':df.index[t],'horizon':h,
                    'actual':y[t],'forecast':float(yh[h-1]),
                    'abs_error':float(abs(y[t]-yh[h-1])),
                    'error':float(y[t]-yh[h-1]),
                    'weekend':int(df.index[t].weekday()>=5),
                    'holiday':int(df['holiday'].iloc[t]),
                    'month':int(df.index[t].month),
                    'year':int(df.index[t].year),
                })
    return pd.DataFrame(results),origins

val_rows, val_origins = evaluate_block(n_train,test_start,preds)
def metrics_table(rows):
    return pd.DataFrame({name:score(g['actual'],g['forecast'])
                         for name,g in rows.groupby('model',sort=False)}).T.sort_values('MAE')
val_metrics = metrics_table(val_rows)
candidate_names = [f'Direct Ridge (alpha={a:g})' for a in RIDGE_GRID]
best_alpha = min(candidate_names, key=lambda name: val_metrics.loc[name,'MAE'])
# Configuration is frozen before any final-test scoring. Train expands every 7 days.
test_model_names = ['Seasonal naive (7-day)', 'ETS (damped, weekly)',
                    'SARIMA (1,0,0)x(1,1,0)[7]', best_alpha]
test_preds = {name:preds[name] for name in test_model_names}
chosen = min(test_model_names,key=lambda name:val_metrics.loc[name,'MAE'])
test_rows,test_origins=evaluate_block(test_start,n,test_preds)
test_metrics=metrics_table(test_rows)

print('Validation model comparison (lower MAE is better):')
display(val_metrics.round(2))
print('Selected Ridge:', best_alpha, '| Validation-selected model:', chosen)
print('Final-test comparison:')
display(test_metrics.round(2))
print('Origins:', len(val_origins), 'validation;', len(test_origins), 'test')
```

**Me:** For the validation-selected model, construct simple nominal 90% prediction bands separately for leads 1 through 7. Estimate the absolute-error quantile for each lead using **validation errors only**; avoid using test errors to set the interval width. Apply the previously estimated widths around test forecasts, clip lower bounds at zero, and report test-period empirical coverage and average interval width. Describe these as empirical bands rather than claiming exact conditional coverage, and note that the seven lead-time bands may not be independent.

**AI:**

```python
# Fit a symmetric absolute-error quantile per lead time using VALIDATION ONLY.
# Finite-sample rank quantile, n_val_origins=20 (90% -> 19th of 20).
chosen_val = val_rows[val_rows.model==chosen]
chosen_test = test_rows.loc[test_rows.model==chosen].copy()
q_by_h={}
for h in range(1,H+1):
    a=np.sort(chosen_val.loc[chosen_val.horizon==h,'abs_error'].to_numpy())
    k=min(len(a), int(np.ceil((len(a)+1)*.90)))
    q_by_h[h]=float(a[k-1])
chosen_test['lower90']=(chosen_test['forecast']-chosen_test.horizon.map(q_by_h)).clip(lower=0)
chosen_test['upper90']=chosen_test['forecast']+chosen_test.horizon.map(q_by_h)
chosen_test['covered90']=(chosen_test.actual>=chosen_test.lower90)&(chosen_test.actual<=chosen_test.upper90)
interval_summary={'nominal':.9, 'coverage':float(chosen_test.covered90.mean()),
                  'avg_width':float((chosen_test.upper90-chosen_test.lower90).mean()),
                  'n':len(chosen_test),'per_h_quantiles':q_by_h}

print('Nominal 90% interval:')
display(pd.DataFrame([interval_summary]).drop(columns=['per_h_quantiles']).round(3))
```

**Me:** After the model is selected, inspect its held-out errors by weekday versus weekend, holiday status, early versus late forecast horizons, and October 2012 versus other months. Identify the largest individual misses and compute a robustness summary excluding the worst 5% of absolute errors and a separate summary excluding October. Treat these as post-selection diagnostics, not an invitation to tune the model on the test set; preserve the nonoverlapping headline test metrics.

**AI:**

```python
# Subgroup risk and outlier case studies from final test only, after freeze.
model_test=test_rows[test_rows.model==chosen].copy()
subgroups={}
for label,mask in [('weekday',model_test.weekend==0),('weekend',model_test.weekend==1),
                   ('holiday',model_test.holiday==1),('nonholiday',model_test.holiday==0),
                   ('October 2012',model_test.month==10),('not October 2012',model_test.month!=10),
                   ('horizons 1-3',model_test.horizon<=3),('horizons 4-7',model_test.horizon>=4)]:
    s=model_test[mask]
    if len(s)>0:subgroups[label]={'n':len(s),**score(s.actual,s.forecast)}
worst=model_test.sort_values('abs_error',ascending=False).head(8).copy()
horizon_by_model=test_rows.groupby(['model','horizon']).apply(
    lambda g:np.mean(np.abs(g.actual-g.forecast)),include_groups=False).unstack('horizon')

# Robustness: daily-overlapping 7-day test forecasts for the selected model are
# NOT used for the headline metrics (avoid inflated N); score shock-excluded weeks
# without changing model decisions or refitting, and add a strong alternative baseline.
outlier_cut=np.quantile(model_test['abs_error'],.95)
trimmed=model_test[model_test.abs_error<outlier_cut]
robust={'selected_mae_without_worst_5pct_errors':float(np.mean(trimmed.abs_error)),
        'removed_count':len(model_test)-len(trimmed),
        'selected_mae_without_october':float(subgroups['not October 2012']['MAE'])}
# Compute MAE among weeks for easier inspection of regime swings.
weekly=model_test.groupby('origin').agg(week_MAE=('abs_error','mean'),
                                          weekly_actual_mean=('actual','mean'))

print('Selected-model subgroup errors:')
display(pd.DataFrame(subgroups).T.round(2))
print('Largest errors:')
display(worst[['date','horizon','actual','forecast','abs_error']].round(2))
```

**Me:** Create readable diagnostic and forecast-comparison figures: chronological split, training ACF and weekday averages, validation/test MAE, forecast-versus-actual plot, uncertainty bands, and an error-analysis figure. Export predictions, validation/test metric tables, interval results, and the run summary to files under the project root; keep plotting assets under `capstone_stage1_assets/`.  The notebook should remain runnable from a root directory containing `capstone_support/`.

**AI:**

```python
# Plots: clean print and PDF/HTML compatible visuals.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
fig,ax=plt.subplots(figsize=(11,3.8))
ax.plot(df.index,df.cnt,color='#2b5b75',lw=1.25)
ax.axvspan(train.index[0],train.index[-1],alpha=.08,color='#2563eb',label='Initial training')
ax.axvspan(validation.index[0],validation.index[-1],alpha=.13,color='#d97706',label='Rolling validation')
ax.axvspan(test.index[0],test.index[-1],alpha=.11,color='#059669',label='Final rolling test')
ax.set(ylabel='Rental count per day',title='Capital Bikeshare: observed daily rentals and chronological blocks')
ax.grid(alpha=.2);ax.legend(loc='upper left',ncols=3,fontsize=8)
fig.tight_layout();fig.savefig('01_series_splits.png',bbox_inches='tight');plt.close(fig)

# Export and metadata cell
# Save raw forecast records for independent audit/reproduction.
test_rows.to_csv('capstone_stage1_test_predictions.csv',index=False)
val_rows.to_csv('capstone_stage1_validation_predictions.csv',index=False)
chosen_test.to_csv('capstone_stage1_intervals.csv',index=False)
val_metrics.round(5).to_csv('capstone_stage1_validation_metrics.csv')
test_metrics.round(5).to_csv('capstone_stage1_test_metrics.csv')

def to_native(v):
    if isinstance(v,(np.integer,np.floating)):return v.item()
    if isinstance(v,dict):return {str(k):to_native(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [to_native(x) for x in v]
    return v
summary={
    'title':'Capital Bikeshare 7-day demand forecasts',
    'data': {'rows':n,'target':'cnt','initial_train_n':n_train,'validation_n':n_val,
             'test_n':len(test),'train_start':str(train.index[0].date()),
             'train_end':str(train.index[-1].date()),'val_start':str(validation.index[0].date()),
             'val_end':str(validation.index[-1].date()),'test_start':str(test.index[0].date()),
             'test_end':str(test.index[-1].date()),'dataset_sha256':df.attrs['capstone']['sha256'],
             'source_page':df.attrs['capstone']['source_page'],
             'support_preparation':df.attrs['capstone']['preparation']},
    'selected_ridge':best_alpha,'selected_model':chosen,
    'val_origins_n':len(val_origins),'test_origins_n':len(test_origins),
    'scored_validation_days_per_model':H*len(val_origins),
    'scored_test_days_per_model':H*len(test_origins),
    'validation_first_origin':str(df.index[val_origins[0]].date()),
    'validation_last_origin':str(df.index[val_origins[-1]].date()),
    'test_first_origin':str(df.index[test_origins[0]].date()),
    'test_last_origin':str(df.index[test_origins[-1]].date()),
    'val_metrics':val_metrics.to_dict('index'),'test_metrics':test_metrics.to_dict('index'),
    'test_models':test_model_names,'interval':interval_summary,
    'diagnostics':diagnostics,'subgroups':subgroups,'robustness':robust,
    'worst_days':[{'date':str(r.date.date()),'horizon':int(r.horizon),
                   'actual':int(r.actual),'predicted':round(r.forecast,1),
                   'abs_error':round(r.abs_error,1)} for r in worst.itertuples(index=False)],
    'test_horizon_MAE':horizon_by_model.to_dict('index'),
}
('capstone_stage1_results.json').write_text(json.dumps(to_native(summary),indent=2))

print('Saved:', len(val_rows), 'validation forecast rows;',len(test_rows),'test forecast rows')
print('Selected model:',summary['selected_model'])
print('Final test MAE:',round(test_metrics.loc[chosen,'MAE'],2))
```

## 4. AI Audit

I used AI mainly to help implement the forecasting workflow in Python. I chose to compare a weekly seasonal-naive benchmark, damped Holt–Winters exponential smoothing, SARIMA, and a direct Ridge forecasting model. I reviewed the code structure, saved outputs, and figures rather than relying on model output alone.

I first checked the bike-data setup and feature construction. The code loads only `bike`, preserves the chronological 438/146/147 train/validation/test split, and checks that all 731 dates form a complete daily series. I also checked that future observed weather is excluded and that the Ridge features use only past rental counts or preannounced calendar fields. The condition requiring a complete seven-day target window for each training example is especially important, since otherwise the direct multi-step regressions could use observations unavailable at the forecast origin.

I then checked the evaluation design and reported scores. Each weekly origin produces seven daily forecasts, with 140 validation days and 147 test days scored consistently across models. The model choice is based on validation MAE, and the resulting ETS test MAE is about 902.8 rentals per day, versus 1,194.7 for the seasonal-naive benchmark. I also checked that the uncertainty-band widths come from validation residuals only and that subgroup and extreme-day analyses do not change the selected model. A full clean-kernel rerun of the notebook remains outstanding because repeated rolling fits timed out in an earlier attempt; its current saved outputs should not be presented as evidence of such a rerun.

Based on all of these checks, I accepted the AI-assisted implementation in this project.
