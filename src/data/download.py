import yfinance as yf
import click


def download_ticker(ticker, period='3y', interval='1d'):
    return yf.Ticker(ticker).history(period=period, interval=interval)


@click.command()
@click.option('--output')
@click.option('--ticker')
def download(output, ticker):
    download_ticker(ticker).to_csv(output)


if __name__ == '__main__':
    download()  # noqa
