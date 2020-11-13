import yfinance as yf
import click


def download_ticker(ticker, period='max', interval='1d'):
    yf.Ticker(ticker).history(period=period, interval=interval)


@click.command()
@click.option('--output')
@click.option('--ticker')
def download(output, ticker):
    yf.Ticker(ticker).history(period='max').to_csv(output)


if __name__ == '__main__':
    download()  # noqa
