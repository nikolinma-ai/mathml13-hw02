# MATH&ML-13. HW-02. Анализ временного ряда: ВВП Ганы

> Блок 7. ML в бизнесе · Курс «Дата Сайентист Про» · SkillFactory
> Гайд по практическому заданию 11.11: проверка стационарности, ARIMA, ресэмплинг, GARCH, валидация моделей.

---

## Зачем это нужно

Это первое задание, где ты собираешь инструменты второго модуля по временным рядам в один пайплайн. Готовых ответов в LMS нет — задание принимает ментор. Поэтому решение оценивается не по «правильному числу», а по тому, **насколько грамотно ты его построил**.

Что отрабатывается на практике:

- умение скачать и подготовить ряд из открытого источника;
- проверка стационарности и приведение ряда к стационарности;
- декомпозиция и интерполяция;
- подбор модели ARIMA по информационному критерию;
- моделирование волатильности через GARCH;
- честная walk-forward валидация и сравнение с baseline;
- грамотная подача отчёта в Jupyter Notebook.

После этого задания у тебя в портфолио появляется законченный пример анализа реального экономического ряда — World Bank GDP. Такой кейс уверенно идёт на собеседование.

---

## Основные понятия

### Постановка задачи

В задании 11.11 модуля MATH&ML-13 нужно:

1. Загрузить ряд ВВП любой страны из World Bank (рекомендуется Ghana).
2. Выполнить разведочный анализ ряда.
3. Проверить стационарность, привести к стационарности.
4. Декомпозировать ряд.
5. Подобрать модель ARIMA, оценить её.
6. Поработать с переходом в другую частоту (ресэмплинг + интерполяция).
7. Построить модель волатильности GARCH.
8. Сравнить ARIMA с базовым подходом (LinearRegression).
9. Проверить честно через walk-forward (TimeSeriesSplit).
10. Прислать ноутбук ментору.

Максимальная оценка — **9 баллов** по шести критериям.

### Источник данных — World Bank API

Ghana GDP, индикатор `NY.GDP.MKTP.CD` («GDP, current US$»). Простой запрос:

```
https://api.worldbank.org/v2/country/GHA/indicator/NY.GDP.MKTP.CD?format=json&date=1960:2021
```

Вернёт ряд из 62 наблюдений (1960–2021), годовая частота. Это маленький ряд — в этом и сложность.

### Маленький ряд = осторожность

С 62 наблюдениями ничего сверхсложного не построишь. Поэтому:

- сложные нейросети не оправданы;
- большой p или q в ARIMA даст переобучение;
- в walk-forward держи окно теста коротким (3 точки) и сплиты немногочисленными (n=5).

### Логарифм для экспоненциального ряда

ВВП растёт мультипликативно: каждый год — процент к прошлому. Поэтому **логарифм** обязателен. Иначе:

- дисперсия в свежей части ряда в десятки раз больше, чем в старой;
- ADF и ARIMA на сырых значениях работают плохо;
- линейный baseline становится бесполезным.

После логарифма ряд становится приближённо линейным, а первая разность — стационарной.

### ARIMA на ВВП

Хороший выбор — ARIMA на **логарифмированном** ряде с d=1. Перебираешь сетку (0,1,1) / (1,1,0) / (1,1,1) / (2,1,1) / (1,1,2) / (2,1,2) и берёшь модель с минимальным AIC. На ВВП Ганы лучшая обычно — **ARIMA(1,1,1)**.

### Ресэмплинг и интерполяция

Годовой ряд можно перевести в квартальный через `resample('QE')` — но появятся пропуски. Их заполняют интерполяцией:

- **linear** — линейная между точками;
- **cubic** — кубический сплайн, гладкая кривая;
- **quadratic** — квадратичная, средне-гладкая;
- **time** — линейная с учётом интервалов времени.

Для гладких рядов вроде ВВП cubic и quadratic дают самые правдоподобные кривые, но могут давать «выбросы» на резких изломах. Linear — безопасный baseline.

### GARCH(1,1) для волатильности

После ARIMA на средние полезно посмотреть, как ведёт себя **разброс**. Берёшь log-returns (логарифмическая разность), кормишь её в `arch_model(...).fit()`. Получаешь две вещи:

- условную дисперсию по истории (видно периоды высокой и низкой волатильности);
- прогноз дисперсии на 5 шагов вперёд.

Для ВВП Ганы получаются хорошо различимые «волатильные» периоды (например, после кризисов 80-х и 2008 года).

### Walk-forward через TimeSeriesSplit

Ритуал:

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, test_size=3)
```

Это даёт пять сплитов с тестом по 3 года. На каждом сплите учишь ARIMA и LinearRegression, считаешь RMSE, потом усредняешь. Получаешь честную оценку.

Типичный результат: ARIMA выигрывает у LinearRegression в разы по RMSE — в нашем эксперименте 7.57 vs 25.85.

---

## Пошаговый разбор

### Шаг 1. Получи данные

```python
import pandas as pd
import requests

url = ('https://api.worldbank.org/v2/country/GHA/indicator/'
       'NY.GDP.MKTP.CD?format=json&date=1960:2021&per_page=200')
resp = requests.get(url, timeout=20).json()
rows = [(int(d['date']), d['value']) for d in resp[1] if d['value'] is not None]
df = pd.DataFrame(rows, columns=['year', 'gdp']).sort_values('year')
df.set_index(pd.to_datetime(df.year, format='%Y'), inplace=True)
gdp = df['gdp']
```

### Шаг 2. Визуализируй и опиши

```python
gdp.plot(title='Ghana GDP, 1960–2021')
gdp.describe()
```

На графике видно: до 2000 — почти плоско, после — резкий рост. Это намёк, что нужен лог.

### Шаг 3. Проверь стационарность

```python
from statsmodels.tsa.stattools import adfuller
import numpy as np

print('raw:', adfuller(gdp.dropna())[1])
print('diff(1):', adfuller(gdp.diff().dropna())[1])
print('log diff(1):', adfuller(np.log(gdp).diff().dropna())[1])
print('log diff(2):', adfuller(np.log(gdp).diff().diff().dropna())[1])
```

Обычно лучшее p-value у **log diff(1)** — берём d=1 на логарифме.

### Шаг 4. Декомпозируй

```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(np.log(gdp), model='additive', period=5)
decomp.plot()
```

Период 5 — экономический мини-цикл, годовой сезонности у годовых данных не бывает.

### Шаг 5. Подбери ARIMA по AIC

```python
from statsmodels.tsa.arima.model import ARIMA

orders = [(0,1,1),(1,1,0),(1,1,1),(2,1,1),(1,1,2),(2,1,2)]
results = []
log_gdp = np.log(gdp)
for o in orders:
    try:
        m = ARIMA(log_gdp, order=o).fit()
        results.append((o, m.aic))
    except Exception:
        pass
results.sort(key=lambda r: r[1])
print(results)
```

Лучшая обычно — **(1,1,1)**, AIC ≈ -47.28.

### Шаг 6. Прогноз и доверительный интервал

```python
fit = ARIMA(log_gdp, order=(1,1,1)).fit()
fc = fit.get_forecast(steps=5)
print(np.exp(fc.predicted_mean))
print(np.exp(fc.conf_int()))
```

Не забудь применить `np.exp` — мы прогнозировали лог, а нужны доллары.

### Шаг 7. Ресэмплинг в кварталы

```python
quarterly = gdp.resample('QE').mean()
methods = ['linear', 'cubic', 'quadratic', 'time']
for m in methods:
    plt.plot(quarterly.interpolate(method=m), label=m)
plt.legend()
```

Глазами выбираешь самый гладкий и без артефактов вариант.

### Шаг 8. GARCH(1,1)

```python
from arch import arch_model
returns = (np.log(gdp).diff() * 100).dropna()
am = arch_model(returns, vol='Garch', p=1, q=1)
res = am.fit(disp='off')
print(res.summary())
fc_var = res.forecast(horizon=5).variance.values[-1]
```

### Шаг 9. Сравнение моделей: hold-out

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train, test = log_gdp[:-5], log_gdp[-5:]
arima_pred = ARIMA(train, order=(1,1,1)).fit().forecast(steps=5)
years = np.arange(len(train)).reshape(-1, 1)
lr = LinearRegression().fit(years, train)
lr_pred = pd.Series(
    lr.predict(np.arange(len(train), len(log_gdp)).reshape(-1, 1)),
    index=test.index)

print('ARIMA RMSE:', mean_squared_error(np.exp(test), np.exp(arima_pred), squared=False))
print('LR    RMSE:', mean_squared_error(np.exp(test), np.exp(lr_pred), squared=False))
```

В нашем эксперименте: **ARIMA = 2.16 млрд $, LR = 31.48 млрд $**. ARIMA на порядок лучше.

### Шаг 10. Walk-forward

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=3)
arima_rmses, lr_rmses = [], []
for tr_idx, te_idx in tscv.split(log_gdp):
    tr, te = log_gdp.iloc[tr_idx], log_gdp.iloc[te_idx]
    a_pred = ARIMA(tr, order=(1,1,1)).fit().forecast(steps=len(te))
    yrs = np.arange(len(tr)).reshape(-1,1)
    yrs_te = np.arange(len(tr), len(tr)+len(te)).reshape(-1,1)
    l_pred = pd.Series(LinearRegression().fit(yrs, tr).predict(yrs_te), index=te.index)
    arima_rmses.append(mean_squared_error(np.exp(te), np.exp(a_pred), squared=False))
    lr_rmses.append(mean_squared_error(np.exp(te), np.exp(l_pred), squared=False))

print('mean ARIMA RMSE:', np.mean(arima_rmses))
print('mean LR    RMSE:', np.mean(lr_rmses))
```

Финальные числа в нашем эксперименте: **7.57 vs 25.85** — ARIMA побеждает.

### Шаг 11. Сводная таблица и выводы

В последней ячейке ноутбука сделай таблицу-резюме и текстовые выводы. Ментор любит, когда есть короткое «что я узнал» в конце.

### Шаг 12. Загрузи на проверку

В юните 11.11 жмёшь кнопку загрузки, прикрепляешь `.ipynb` (заранее **прогнанный** — все ячейки с выходами), пишешь в текстовое поле короткое описание решения, отправляешь.

---

## Примеры

### Пример 1. Минимальный ноутбук на 5 баллов

Если времени совсем нет:

1. Загрузка World Bank (1 балл).
2. ADF + diff (1 балл).
3. ARIMA с фиксированным (1,1,1) и прогноз (2 балла).
4. Сравнение с LinearRegression (1 балл).

Это ~50% от максимума, обычно проходит.

### Пример 2. Ноутбук на полный балл

К минимуму добавляешь:

1. Логарифмирование + сравнение ADF на сыром и на логе.
2. Декомпозицию.
3. Перебор по AIC + комментарий по выбору.
4. Ресэмплинг в QE с 4 видами интерполяции.
5. GARCH(1,1) с прогнозом variance.
6. Walk-forward через TimeSeriesSplit с 5 сплитами.
7. Финальную таблицу-резюме.

Это 9 баллов.

### Пример 3. Чужой кейс (не Ghana)

Можно взять любую страну. Изменится только часть с загрузкой:

```python
url = '...country/RUS/indicator/NY.GDP.MKTP.CD?...'
```

Код анализа остаётся тем же. ARIMA-параметры могут отличаться — пересчитай AIC.

---

## Итог

HW-02 — это сборка пайплайна: ADF → log+diff → seasonal_decompose → ARIMA grid → resample/interpolate → GARCH → walk-forward. Каждый шаг подкреплён кодом и графиком, в конце — таблица с числами и текстовый вывод.

Главное правило для тех, кого проверяет ментор: **показывай не только числа, но и логику принятия решений**. Не «я взял ARIMA(1,1,1)», а «я перебрал 6 порядков, AIC минимален у (1,1,1) — взял её».

---

## Частые ошибки

- **Ноутбук без выходов.** Прогрелся локально — сохрани с выводами через `jupyter nbconvert --execute`. Иначе ментор ничего не увидит.
- **Нет логарифма.** ADF на сырых ВВП обычно не проходит, ARIMA даёт мусор.
- **d=2 без проверки.** Если log-diff уже стационарен — d=1, не d=2.
- **k-fold вместо walk-forward.** Завалит ментор, и правильно.
- **Прогноз в логарифмах без `np.exp`.** В отчёт пойдут трёхзначные числа вместо миллиардов долларов.
- **GARCH на самом ВВП.** GARCH хочет log-returns, а не уровни.
- **Нет сравнения с baseline.** Один ARIMA без LinearRegression — нет точки отсчёта, ментор снимет балл.
- **Нет таблицы и выводов.** Голые ячейки без объяснений — очень неприятно ревьюить.
- **Слишком короткий test в walk-forward.** Если test_size=1 при n_splits=10 — шумит, метрика прыгает.

---

## Запомни

- World Bank индикатор: `NY.GDP.MKTP.CD`, формат JSON.
- ВВП — мультипликативный ряд → логарифм обязателен.
- Лучшая ARIMA на лог-ВВП Ганы: **(1,1,1)**, AIC ≈ -47.
- Resample method: `'QE'` (Quarter End) для квартального.
- TimeSeriesSplit: `n_splits=5`, `test_size=3` — оптимум для 62 точек.
- В прогноз → `np.exp(...)` обязательно.
- GARCH ест log-returns в процентах.
- Ноутбук — с прогнанными выводами.
- Подача: `services.skillfactory.ru`, save draft → upload file → save → submit.
- 9 баллов = 6 критериев × проработка по полной программе.

---

## Проверь себя

1. Какой индикатор World Bank даёт ВВП в текущих долларах?
2. Зачем перед ARIMA логарифмируют ряд ВВП?
3. Как выбирают (p, d, q) — на глаз по ACF/PACF или перебором по AIC?
4. Что произойдёт, если применить k-fold к ряду длиной 62?
5. Что моделирует GARCH — уровень ряда или его волатильность?
6. Что нужно сделать с прогнозом ARIMA, если он построен на логарифме?
7. Какие методы интерполяции работают для гладких экономических рядов?
8. Зачем нужен LinearRegression в этом задании?
9. Почему `period=5` в seasonal_decompose, а не 12?
10. Что должно быть в финальной ячейке ноутбука?

---

> **Финальный совет.** Ментор смотрит ноутбук минут пять. Сделай так, чтобы за эти пять минут он увидел: твой ход мысли, твои графики, твою таблицу-резюме, твой вывод. Не пиши лишнего, но и не оставляй ячейки без подписей.
