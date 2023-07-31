import mltable
from mltable import MLTableHeaders, MLTableFileEncoding, DataType
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


# create paths to the data files
paths = [{"pattern": "wasbs://data@azuremlexampledata.blob.core.windows.net/feature-store-prp/observation_data/train/*.parquet"}]


# create a table from the parquet paths
tbl = mltable.from_parquet_files(paths)

""" 
# Apply Transformations

# table a random sample
tbl = tbl.take_random_sample(probability=0.001, seed=735)

# filter trips with a distance > 0
tbl = tbl.filter("col('tripDistance') > 0")

# Drop columns
tbl = tbl.drop_columns(["puLocationId", "doLocationId", "storeAndFwdFlag"])

# ensure survived column is treated as boolean
data_types = {
    "Survived": DataType.to_bool(
        true_values=["True", "true", "1"], false_values=["False", "false", "0"]
    )
}
tbl = tbl.convert_column_types(data_types)

# Create two new columns - year and month - where the values are taken from the path
tbl = tbl.extract_columns_from_partition_format("/puYear={year}/puMonth={month}")
"""

# print the first 5 records of the table as a check
print(tbl.show(5))

# save the data loading steps in an MLTable file
mltable_folder = "./fraud_train"
tbl.save(mltable_folder)

# Connect to the AzureML workspace
subscription_id = "e0d7a68e-191f-4f51-83ce-d93995cd5c09"
resource_group = "my_ml_tests"
workspace = "<AML_WORKSPACE_NAME>"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# Define the Data asset object
my_data = Data(
    path=mltable_folder,
    type=AssetTypes.MLTABLE,
    description="Fraud Training Sample",
    name="fraud_train_ds",
    version="1",
)

# Create the data asset in the workspace
ml_client.data.create_or_update(my_data)
