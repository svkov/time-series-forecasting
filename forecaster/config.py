import os

password = os.environ['POSTGRES_PASSWORD']
host = os.environ['POSTGRES_HOST']
database = os.environ['POSTGRES_DB']
port = os.environ['POSTGRES_PORT']
user = os.environ['POSTGRES_USER']

data_loader_host = os.environ['DATA_LOADER_HOST']
data_loader_port = os.environ['DATA_LOADER_PORT']

DATABASE_CONNECTION_URI = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
# DATABASE_CONNECTION_URI = f'sqlite:///example.sqlite'
