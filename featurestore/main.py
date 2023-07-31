#!/usr/bin/env python
# coding: utf-8

# In[1]:


print(f"spark version:{spark.version}")


# In[1]:


import os
import json
import pandas as pd
from datetime import datetime

from azure.ai.ml.entities import FeatureSetSpecification,RecurrenceTrigger

import sys

root_dir = "./Users/ezzatdemnati/e2e_mlops_process"
sys.path.insert(0,root_dir)

import featurestore.setup.featurestore_setuptools as fs_setup

create_featurestore=False #True
create_feature_sets=False #True
Register_feature_sets_and_entities=False #True
set_offline_store=False #True


# In[2]:


# Read config file:
with open(os.path.join(root_dir,"featurestore/config/feature_store_config.json"),'r') as f:
    fs_config = json.load(f)

# Init FeatureStore class
fs_class = fs_setup.FeatureStoreTools(subscription_id=fs_config["subscription_id"],
                resource_group_name=fs_config["resource_group_name"],
                location=fs_config["location"],
                featurestore_name=fs_config["name"],
                root_dir=root_dir,
                fs_config=fs_config,
                ml_client=None
            )


# In[6]:


# Create FeatureStore
if create_featurestore:        
    fs_class.create_feature_store(ml_client=None,verbose=0)



# In[4]:


# Get FeatureStore
feature_store = fs_class.get_feature_store(verbose=0)


# In[8]:


# Create Feature Sets
if create_feature_sets:
    # Create Feature Sets
    for fs_set in fs_config["feature_sets"]:
        print(fs_set)
        fs_class.create_feature_set(name=fs_set["name"],
            transformer_class_name=fs_set.get("transformer_class_name"), 
            data_source_path=fs_set["data_source_path"],
            timestamp_column_name=fs_set.get("timestamp_column_name"),
            source_delay=fs_set.get("source_delay"),
            index_columns=fs_set["index_columns"],
            source_lookback=fs_set.get("source_lookback"),
            temporal_join_lookback=fs_set.get("temporal_join_lookback"),
            infer_schema=fs_set["infer_schema"],
            export=True, # Export featuset specs locally
            verbose=1
            )

    


# In[9]:


# Register Feature Sets and Entities 
if Register_feature_sets_and_entities:
    # Register Entities 
    for fs_entity in fs_config["entities"]:
        fs_class.register_entity(name=fs_entity["name"],
            version=fs_entity["version"],
            index_columns=fs_entity["index_columns"],
            description=fs_entity["description"],
            tags=fs_entity["tags"]
        )
    
    # Register Feature Sets
    for fs_set in fs_config["feature_sets"]:
        spec_path = os.path.join(root_dir,f"featurestore/featuresets/{fs_set['name']}/spec")
        
        fs_class.register_feature_set(fs_set["name"], 
            version=fs_set["version"], 
            description=fs_set["description"], 
            entities=fs_set["entities"], 
            stage=fs_set["stage"], 
            specification=FeatureSetSpecification(path=spec_path), 
            tags=fs_set["tags"]
        )


# In[14]:


# Set Offline Materialization
if set_offline_store:
    storage_subscription_id = "e0d7a68e-191f-4f51-83ce-d93995cd5c09"
    storage_resource_group_name = "my_ml_tests"
    storage_account_name = "ezmylake"
    storage_file_system_name = "fsofflinetest"
    uai_name = "fstoreuaitest"

    gen2_container_arm_id = f"/subscriptions/{storage_subscription_id}/resourceGroups/{storage_resource_group_name}/providers/Microsoft.Storage/storageAccounts/{storage_account_name}/blobServices/default/containers/{storage_file_system_name}"

    # Set FeatureStore Offline materialization
    fs_class.set_feature_store_materialization(gen2_container_arm_id,storage_account_name,uai_name)

    # Set and run Feature Sets Offline materialization
    # create a schedule that runs the materialization job every 3 hours
    # Example: datetime(2023, 4, 15, 0, 4, 10, 0) = "2023-04-15 00:04:10"

    # Default offline config
    default_schedule = None
    default_spark_configuration = None
    if fs_config.get("offline_default"):
        if fs_config["offline_default"].get("schedule"):
            default_schedule = RecurrenceTrigger(
                interval=fs_config["offline_default"]["schedule"]["interval"], 
                frequency=fs_config["offline_default"]["schedule"]["frequency"], 
                start_time=datetime.fromisoformat(fs_config["offline_default"]["schedule"]["start_time"]) 
            )
            default_spark_configuration={
                        "spark.driver.cores": fs_config["offline_default"]["spark_configuration"]["spark.driver.cores"],
                        "spark.driver.memory": fs_config["offline_default"]["spark_configuration"]["spark.driver.memory"],
                        "spark.executor.cores": fs_config["offline_default"]["spark_configuration"]["spark.executor.cores"],
                        "spark.executor.memory": fs_config["offline_default"]["spark_configuration"]["spark.executor.memory"],
                        "spark.executor.instances": fs_config["offline_default"]["spark_configuration"]["spark.executor.instances"],
                    }
    


# In[12]:


#Set offline materialization for all feature sets (if applicable)
for fs_set in fs_config["feature_sets"]:
    # Check if offline materialization config
    if fs_set.get("offline") and fs_set.get("offline").get("active"):
        # Check if schedule config
        feature_set_schedule = None
        spark_configuration = None

        #Check if feature set config is available
        if fs_set["offline"].get("instance_type") and fs_set["offline"].get("spark_configuration"):
            if fs_set["offline"].get("schedule"):
                feature_set_schedule = RecurrenceTrigger(
                    interval=fs_set["offline"]["schedule"]["interval"], 
                    frequency=fs_set["offline"]["schedule"]["frequency"], 
                    start_time=datetime.fromisoformat(fs_set["offline"]["schedule"]["start_time"]) 
                )                
                spark_configuration={
                    "spark.driver.cores": fs_set["offline"]["spark_configuration"]["spark.driver.cores"],
                    "spark.driver.memory": fs_set["offline"]["spark_configuration"]["spark.driver.memory"],
                    "spark.executor.cores": fs_set["offline"]["spark_configuration"]["spark.executor.cores"],
                    "spark.executor.memory": fs_set["offline"]["spark_configuration"]["spark.executor.memory"],
                    "spark.executor.instances": fs_set["offline"]["spark_configuration"]["spark.executor.instances"],
                }
            elif default_spark_configuration:
                feature_set_schedule = default_schedule
                spark_configuration = default_spark_configuration
            
        # Set feature set offline materialization
        if spark_configuration and feature_set_schedule:
            fs_class.set_feature_set_materialization(name=fs_set["name"],
                version=fs_set["version"],
                instance_type=fs_set["offline"]["instance_type"],
                spark_configuration=spark_configuration,
                schedule=feature_set_schedule
        )


# ### Backfill data for the transactions feature set

# In[5]:


job_id= fs_class.run_backfill(featureset_name="transactions",
                      featureset_version="4",
                      start_datetime="2023-01-01 00:00:00",
                      end_datetime="2023-06-30 23:59:00"
                     )

# get the job URL, and stream the job logs
fs_client.jobs.stream(job_id)


# In[10]:


# look up the featureset by providing name and version
transactions_featureset = feature_store.feature_sets.get("transactions", "4")
df = transactions_featureset.to_spark_dataframe()

from pyspark.sql import functions

display(df.agg(functions.min('timestamp'),functions.max('timestamp')))


# ## Model Training

# In[5]:


# get the registered transactions feature set, version 1
transactions_featureset = feature_store.feature_sets.get("transactions", "4")
accounts_featureset = feature_store.feature_sets.get("accounts", "4")


# Notice that account feature set spec is in your local dev environment (this notebook): not registered with feature store yet
features = [
    accounts_featureset.get_feature("accountAge"),
    accounts_featureset.get_feature("numPaymentRejects1dPerUser"),
    transactions_featureset.get_feature("transaction_amount_7d_sum"),
    transactions_featureset.get_feature("transaction_amount_3d_sum"),
    transactions_featureset.get_feature("transaction_amount_7d_avg"),
]
features


# In[6]:


accounts_featureset = feature_store.feature_sets.get("accounts", "4")
# get access to the feature data
accounts_feature_df = accounts_featureset.to_spark_dataframe()
display(accounts_feature_df.head(5))


# In[18]:


transactions_featureset = feature_store.feature_sets.get("transactions", "4")

# get access to the feature data
transactions_featureset_df = transactions_featureset.to_spark_dataframe()
display(transactions_featureset_df.head(5))


# In[16]:






# In[52]:


# List available feature sets
all_featuresets = feature_store.feature_sets.list()
for fs in all_featuresets:
    print(fs)

# List of versions for transactions feature set
print("=====================All versions=============================")
all_transactions_featureset_versions = feature_store.feature_sets.list(
    name="transactions"
)
v=0
for fs in all_transactions_featureset_versions:
    if int(fs.name)>v:
        v = int(fs.name)
    print(f"version:{fs.name}")

# See properties of the transactions featureset including list of features
print(f"=====================Current version {v}=============================")
for f in feature_store.feature_sets.get(name="transactions", version=str(v)).features:
    print(f"Feature \n    name:{f.name} \n    Type:{f.type} \n    Description:{f.description}")

