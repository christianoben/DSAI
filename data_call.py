# Importing necessary libraries to use
import os
import pandas as pd
import sqlalchemy
from deep_translator import GoogleTranslator
import spacy
import pyreadr

# Setting the working directory
os.chdir("your path")

# Database connection parameters
db_host = "masked"  # PostgreSQL host IP
db_port = "masked"  # PostgreSQL port
db_name = "rapidpro"  # Database name
db_user = "masked"  # Username
db_password = "masked"  # Password

# Constructing the connection URL for PostgreSQL
connection_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = sqlalchemy.create_engine(connection_url)

# Querying the database and loading the data into pandas DataFrames
tables = pd.read_sql("SELECT schemaname, tablename FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema')", engine)

# Retrieving and processing message labels
msg_label = pd.read_sql("SELECT * FROM msgs_msg_labels", engine).drop(columns=['id']).rename(columns={"msg_id": "id"})

# Filtering and merging messages
msgs_msg = pd.read_sql("SELECT * FROM msgs_msg WHERE direction = 'I' AND length(text) > 10", engine).merge(msg_label, on="id", how="left").drop(columns=['id'])

# Retrieving labels
msgs_label = pd.read_sql("SELECT * FROM msgs_label", engine)[['id', 'name']]

# Filtering messages with labels and merging with label names
msgs_msg_filtered = msgs_msg[msgs_msg['label_id'].notna()].rename(columns={"label_id": "id"}).merge(msgs_label, on="id", how="left")

# Applying filtering based on specific ID values
filter_ids = [7, 5, 42, 10]
msgs_msg_filtered = msgs_msg_filtered[~msgs_msg_filtered['id'].isin(filter_ids)]
