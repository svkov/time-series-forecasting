
# Stock Prices Forecasting

[![Snakemake](https://img.shields.io/badge/snakemake-≥5.6.0-brightgreen.svg?style=flat)](https://snakemake.readthedocs.io)

## Описание

Проект по предсказанию цены биржевых показателей. Данные взяты с Yahoo Finance (библиотека `yfinance`).

Полученные результаты используются для написания дипломной работы. 

Все вычисления легко воспроизводимы, а результат автоматически генерируется в LaTeX. 

## Структура проекта

- `assets` - папка с css- и js-кодом для web-интерфейса
- `data` - папка для хранения данных (скачиваются автоматически при помощи snakemake)
- `notebooks` - папка для хранения Jupyter-ноутбуков
- `reports` - папка, в которой хранится вся информация, необходимая для отчетов: результаты прогнозов, графики, латех
- `spbu_diploma` - папка для хранения стилей латеха и главного файла латеха
- `src` - python-пакет, в котором хранится вся кодовая база проекта  
- `workflow` - папка с snakemake пайплайном

## Установка
- `git clone https://github.com/svkov/BitcoinForecasting.git`
- Установить зависимости (скоро будет)

## Запуск

### Snakemake

- `snakemake -j8` - запустить генерацию pdf с дипломной работой
- После завершения можно посмотреть результат в `spbu_diploma/main_example.pdf`

### Приложения
- `python main.py`
- Идем на [http://127.0.0.1:8050][localhost]



[localhost]: http://127.0.0.1:8050