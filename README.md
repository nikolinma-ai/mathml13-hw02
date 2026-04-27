# MATH&ML-13 HW-02 — Анализ временного ряда: ВВП Ганы

Решение задания «11.11. Задание» курса SkillFactory DSPR-2.0, модуль MATH&ML-13 «Анализ временных рядов II».

## Датасет
Ghana GDP (current US$), 1960–2021 — World Bank API, индикатор `NY.GDP.MKTP.CD`. 62 наблюдения, годовая частота.

## Что внутри
- `MATH-ML-13_HW-02_Ghana_GDP.ipynb` — итоговый Jupyter-ноутбук (35 ячеек, прогнан без ошибок, все графики и выводы)
- `build_notebook.py` — генератор ноутбука (`nbformat`)

## Структура решения (6 критериев, max 9 баллов)

1. **Загрузка и описательная статистика** — World Bank API, plot, describe
2. **Стационарность** — ADF на raw / diff(1) / log-diff(1) / log-diff(2) + `seasonal_decompose(period=5)`
3. **ARIMA grid search по AIC** — лучшая модель **ARIMA(1,1,1)**, AIC = -47.28
4. **Resampling в квартальную частоту** — `resample('QE')` + интерполяции `linear` / `cubic` / `quadratic` / `time`
5. **Волатильность** — rolling 5y std + **GARCH(1,1)**, прогноз variance на 5 шагов
6. **Сравнение моделей**
   - Hold-out (последние 5 лет): ARIMA RMSE=2.16 vs LinearRegression RMSE=31.48
   - Walk-forward CV (`TimeSeriesSplit`, n=5, test_size=3): mean RMSE ARIMA=7.57 vs LR=25.85

## Стек
`pandas`, `numpy`, `matplotlib`, `statsmodels` (ARIMA, ADF, seasonal_decompose, ACF/PACF), `arch` (GARCH), `scikit-learn` (LinearRegression, TimeSeriesSplit).

## Запуск
```bash
pip install pandas numpy matplotlib statsmodels arch scikit-learn jupyter
python build_notebook.py            # генерирует .ipynb
jupyter nbconvert --execute MATH-ML-13_HW-02_Ghana_GDP.ipynb --to notebook --inplace
```

Дата сабмита ментору: 2026-04-27.
