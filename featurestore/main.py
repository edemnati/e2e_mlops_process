import os
import json
import pandas as pd
from datetime import datetime

from azure.ai.ml.entities import FeatureSetSpecification,RecurrenceTrigger

import featurestore.setup.featurestore_setuptools as fs_setup


def prepare_training_data():
    # look up the featureset by providing name and version
    transactions_featureset = featurestore.feature_sets.get("transactions", "1")
    # list its features
    transactions_featureset.features

    # Load observation data
    observation_data_path = "wasbs://data@azuremlexampledata.blob.core.windows.net/feature-store-prp/observation_data/train/*.parquet"
    observation_data_df = spark.read.parquet(observation_data_path)
    obs_data_timestamp_column = "timestamp"

    # get_offline_features
    from azureml.featurestore import get_offline_features
    selected_features = ["transaction_amount_7d_sum","transaction_amount_7d_avg","transaction_3d_count","transaction_amount_3d_avg"]
    features = [transactions_featureset.get_feature(f) for f in selected_features]

    # generate training dataframe by using feature data and observation data
    training_df = get_offline_features(
        features=selected_features,
        observation_data=observation_data_df,
        timestamp_column=obs_data_timestamp_column,
    )

    return training_df


def main(create_featurestore=True,
         create_feature_sets=True,
         Register_feature_sets_and_entities=True,
         set_offline_store=True
         ):
    # Read config file:
    fs_config = json.load("./featurestore/config/feature_store_config.json")
    fs_class = fs_setup.FeatureStoreTools(subscription_id=fs_config["subscription_id"],
                    resource_group_name=fs_config["resource_group_name"],
                    location=fs_config["location"],
                    featurestore_name=fs_config["name"],
                    root_dir=None,
                    ml_client=None
                )

    # Create FeatureStore
    if create_featurestore:        
        fs_class.create_feature_store(featurestore_location=fs_config["location"],
            ml_client=None,
            verbose=0
        )

    # Get FeatureStore
    feature_store = fs_class.get_feature_store(verbose=0)

    # Create Feature Sets
    if create_feature_sets:
        # Create Feature Sets
        for fs_set in fs_config["feature_sets"]:
            fs_class.create_feature_set(name=fs_set["name"],
                transformer_class_name=fs_set["transformer_class_name"], 
                data_source_path=fs_set["data_source_path"],
                timestamp_column_name=fs_set["timestamp_column_name"],
                source_delay=fs_set["source_delay"],
                index_columns=fs_set["index_columns"],
                source_lookback=fs_set["source_lookback"],
                temporal_join_lookback=fs_set["temporal_join_lookback"],
                infer_schema=fs_set["infer_schema"]
                )

            # Export Feature Sets locally
            spec_path = os.path.join(root_dir,f"./featurestore/specs/{fs_set['name']}")
            if not os.path.exists(spec_path):
                os.makedirs(spec_path)

            fs_class.export_feature_set_specs(name=fs_set["name"],
                spec_path=spec_path
            )
        
        # Register Feature Sets and Entities 
        if Register_feature_sets_and_entities:
            # Register Entities 
            for fs_entity in fs_config["feature_sets"]:
                fs_class.register_entity(name=fs_entity["name"],
                    version=fs_entity["version"],
                    index_columns=fs_entity["index_columns"],
                    description=fs_entity["description"],
                    tags=fs_entity["tags"])
            
            # Register Feature Sets
            for fs_set in fs_config["feature_sets"]:
                spec_path = os.path.join(root_dir,f"featurestore/featuresets/{fs_set['name']}/spec"),
                fs_class.register_feature_set(fs_set["name"], 
                    version=fs_set["version"], 
                    description=fs_set["description"], 
                    entities=fs_set["entities"], 
                    stage=fs_set["stage"], 
                    specification=FeatureSetSpecification(path=spec_path), 
                    tags=fs_set["tags"])

    # Set Offline Materialization
    if set_offline_store:
        storage_subscription_id = "e0d7a68e-191f-4f51-83ce-d93995cd5c09"
        storage_resource_group_name = "my_ml_tests"
        storage_account_name = "ezmylake"
        storage_file_system_name = "fsofflinetest"
        uai_name = "fstoreuai"

        gen2_container_arm_id = f"/subscriptions/{storage_subscription_id}/resourceGroups/{storage_resource_group_name}/providers/Microsoft.Storage/storageAccounts/{storage_account_name}/blobServices/default/containers/{storage_file_system_name}"

        # Set FeatureStore Offline materialization
        fs_class.set_feature_store_materialization(gen2_container_arm_id,uai_name)


        # Set and run Feature Sets Offline materialization
        # create a schedule that runs the materialization job every 3 hours
        #datetime(2023, 4, 15, 0, 4, 10, 0) = "2023-04-15 00:04:10"

        # Default offline config
        default_schedule = None
        default_spark_configuration = None
        if fs_config.get("offline_default"):
            if fs_config["offline_default"].get("schedule"):
                default_schedule = RecurrenceTrigger(
                    interval=fs_set["offline"]["schedule"]["interval"], 
                    frequency=fs_set["offline"]["schedule"]["frequency"], 
                    start_time=datetime.fromisoformat(fs_set["offline"]["schedule"]["start_time"]) 
                )
                default_spark_configuration={
                            "spark.driver.cores": fs_config["offline_default"]["spark_configuration"]["spark.driver.cores"],
                            "spark.driver.memory": fs_config["offline_default"]["spark_configuration"]["spark.driver.memory"],
                            "spark.executor.cores": fs_config["offline_default"]["spark_configuration"]["spark.executor.cores"],
                            "spark.executor.memory": fs_config["offline_default"]["spark_configuration"]["spark.executor.memory"],
                            "spark.executor.instances": fs_config["offline_default"]["spark_configuration"]["spark.executor.instances"],
                        }
        
        #Set offline materialization for all feature sets (if applicable)
        for fs_set in fs_config["feature_sets"]:
            # Check if offline materialization config
            if if fs_set.get("offline").get("active"):
                # Check if schedule config
                feature_set_schedule = None
                
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
                    else:
                        
                # Set feature set offline materialization
                fs_class.set_feature_set_materialization(name=fs_set["name"],
                    version=fs_set["version"],
                    instance_type=fs_set["offline"]["instance_type"],
                    spark_configuration=spark_configuration,
                    schedule=feature_set_schedule
                )










