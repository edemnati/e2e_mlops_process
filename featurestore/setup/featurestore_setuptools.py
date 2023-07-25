import os
import json
from datetime import datetime

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    FeatureStore,
    FeatureStoreEntity,
    FeatureSet,
)
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential


class FeatureStoreTools:
    def __init__(self,subscription_id,resource_group_name,location,featurestore_name,root_dir=None,ml_client=None,fs_config=None):        
        try:
            self.subscription_id = subscription_id
            self.resource_group_name = resource_group_name
            self.location = location
            self.featurestore_name = featurestore_name

            self.root_dir = os.path.join(root_dir,"featurestore")            
            print(f"root_dir:{root_dir}")
            print(f"self.root_dir:{self.root_dir}")
            
            # Read feature store config file
            if fs_config:
                feature_store_config = fs_config
            else:
                with open(os.path.join(self.root_dir,"config/feature_store_config.json"),"r") as f:
                    feature_store_config = json.load(f)
            
            if ml_client:
                self.ml_client = ml_client
            else:
                self.ml_client = MLClient(
                    AzureMLOnBehalfOfCredential(),
                    subscription_id=subscription_id,
                    resource_group_name=resource_group_name,
                )
            
        except Exception as e:
            print(f"ERROR occured while initiating FeatureStoreTools Class")
            print(f"Raies Exception:{e}")
            raise

    def create_feature_store(self,ml_client=None,verbose=0):
        if not ml_client:
            ml_client = self.ml_client

        fs = FeatureStore(name=self.featurestore_name, location=self.location)
        # wait for featurestore creation
        fs_poller = ml_client.feature_stores.begin_create(fs, update_dependent_resources=True)
        if verbose>0:
            print(fs_poller.result())

    def get_feature_store(self,verbose=0):
        # feature store client
        from azureml.featurestore import FeatureStoreClient
        from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
        featurestore = None
        try:
            featurestore = FeatureStoreClient(
                credential=AzureMLOnBehalfOfCredential(),
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group_name,
                name=self.featurestore_name,
            )
        except Exception as e:
            print(f"ERROR while getting feature store '{featurestore_name}'. Try to create the feature store if it doesn't exist.")
            print(f"Raies Exception:{e}")
            raise        
        
        finally:
            return featurestore

    def create_feature_set(self,
        name,
        transformer_class_name, 
        data_source_path,
        timestamp_column_name,
        source_delay,
        index_columns,
        source_lookback,
        temporal_join_lookback,
        infer_schema,
        export=True,
        verbose=0
        ):
        
        """
        Featureset specification is a self-contained definition of feature set that can be developed and tested locally.
        Parameters:
          - name:
          - transformer_class_name:
          - data_source_path:
          - timestamp_column_name:
          - source_delay:
          - index_columns:
          - source_lookback:
          - temporal_join_lookback:
          - infer_schema:
        
        """
        from azureml.featurestore import create_feature_set_spec, FeatureSetSpec
        from azureml.featurestore.contracts import (
            DateTimeOffset,
            FeatureSource,
            TransformationCode,
            Column,
            ColumnType,
            SourceType,
            TimestampColumn,
        )
        
        def get_datacolumntype(val):
            if val=="STRING":
                return ColumnType.STRING
            elif val=="INTEGER":
                return ColumnType.INTEGER
            elif val=="BOOLEAN":
                return ColumnType.BOOLEAN
            elif val=="DATETIME":
                return ColumnType.DATETIME
            elif val=="DOUBLE":
                return ColumnType.DOUBLE
            elif val=="FLOAT":
                return ColumnType.FLOAT
            elif val=="LONG":
                return ColumnType.LONG
            elif val=="BINARY":
                return ColumnType.BINARY
            else:
                return None

        try:
            transformation_code_path = os.path.join(self.root_dir,f"featuresets/{name}/transformation_code")
            transactions_featureset_code_path = os.path.join(self.root_dir,f"featuresets/{name}/transformation_code")
            
            if source_delay:
                source_delay_DTO = DateTimeOffset(days=source_delay["days"], 
                            hours=source_delay["hours"], 
                            minutes=source_delay["minutes"]
                        )   
            else:
                source_delay_DTO = DateTimeOffset(days="0",hours="0",minutes="0")
            
            if transformer_class_name:
                transformation_code = TransformationCode(
                    path=transactions_featureset_code_path,
                    transformer_class=transformer_class_name,
                )
            else:
                transformation_code = None
            
            if temporal_join_lookback:
                temporal_join_lookback_DTO = DateTimeOffset(
                        days=temporal_join_lookback["days"], 
                        hours=temporal_join_lookback["hours"], 
                        minutes=temporal_join_lookback["minutes"]
                    )  
            else:
                temporal_join_lookback_DTO = DateTimeOffset(days="365",hours="0",minutes="0")

            if source_lookback:
                source_lookback_DTO = DateTimeOffset(
                    days=source_lookback["days"], 
                    hours=source_lookback["hours"], 
                    minutes=source_lookback["minutes"]
                )
            else:
                source_lookback_DTO = DateTimeOffset(days="365",hours="0",minutes="0")

            if timestamp_column_name:
                timestamp_columns = TimestampColumn(name=timestamp_column_name)
            else:
                timestamp_columns = None
            
            transactions_featureset_spec = create_feature_set_spec(
                source=FeatureSource(
                    type=SourceType.parquet,
                    path=data_source_path,
                    timestamp_column=timestamp_columns,
                    source_delay=source_delay_DTO,
                ),
                transformation_code=transformation_code,
                index_columns=[Column(name=c["name"], type=get_datacolumntype(c["type"])) for c in index_columns],
                source_lookback=source_lookback_DTO,
                temporal_join_lookback=temporal_join_lookback_DTO,
                infer_schema=infer_schema,
            )
            
            # Export specs locally
            if export:
                self.export_feature_set_specs(name,transactions_featureset_spec)
            # display few records
            if verbose:
                # Generate a spark dataframe from the feature set specification
                transactions_fset_df = transactions_featureset_spec.to_spark_dataframe()
                display(transactions_fset_df.head(5))
        except Exception as e:
            print(f"ERROR occured while creating feature_set:{name}")
            print(f"Raies Exception:{e}")
            raise  
        return transactions_featureset_spec

    def export_feature_set_specs(self,name,transactions_featureset_spec,verbose=0):
        """
        Export as feature set spec
        In order to register the feature set spec with the feature store, it needs to be saved in a specific format. 
        the generated FeaturesetSpec willbe saved under: featurestore/featuresets/<name>/spec/FeaturesetSpec.yaml

        Spec contains these important elements:
          - source: reference to a storage. In this case a parquet file in a blob storage.
          - features: list of features and their datatypes. If you provide transformation code (see Day 2 section), 
                      the code has to return a dataframe that maps to the features and datatypes.
          - index_columns: the join keys required to access values from the feature set

        """
        # create a new folder to dump the feature set spec
        #fs_set_name = transactions
        transactions_featureset_spec_folder = os.path.join(self.root_dir,f"featuresets/{name}/spec")
        
        # check if the folder exists, create one if not
        if not os.path.exists(transactions_featureset_spec_folder):
            os.makedirs(transactions_featureset_spec_folder)        
        
        transactions_featureset_spec.dump(transactions_featureset_spec_folder)


    def register_entity(self,name,version,index_columns,description,tags,verbose=0):
        """
        DataColumnType: BINARY,BOOLEAN,DATETIME,DOUBLE,FLOAT,INTEGER,LONG,STRING

        name="Account",
        version="1",
        index_columns=[DataColumn(name="accountID", type=DataColumnType.STRING)],
        description="This entity represents user account index key accountID.",
        tags={"data_typ": "nonPII"},
        
        """
        from azure.ai.ml.entities import DataColumn, DataColumnType

        def get_datacolumntype(val):
            if val=="STRING":
                return DataColumnType.STRING
            elif val=="INTEGER":
                return DataColumnType.INTEGER
            elif val=="BOOLEAN":
                return DataColumnType.BOOLEAN
            elif val=="DATETIME":
                return DataColumnType.DATETIME
            elif val=="DOUBLE":
                return DataColumnType.DOUBLE
            elif val=="FLOAT":
                return DataColumnType.FLOAT
            elif val=="LONG":
                return DataColumnType.LONG
            elif val=="BINARY":
                return DataColumnType.BINARY
            else:
                return None

        entity_config = FeatureStoreEntity(
            name=name,
            version=version,
            index_columns=[DataColumn(name=c["name"], type=get_datacolumntype(c["type"])) for c in index_columns],
            description=description,
            tags=tags,
        )

        # feature store ml client
        fs_client = MLClient(
            AzureMLOnBehalfOfCredential(),
            self.subscription_id,
            self.resource_group_name,
            self.featurestore_name,
        )

        poller = fs_client.feature_store_entities.begin_create_or_update(entity_config)
        print(poller.result())

        return poller
        

    def register_feature_set(self,name, version, description, entities, stage, specification, tags,verbose=0):
        """
        Example
        name="transactions",
        version="1",
        description="7-day and 3-day rolling aggregation of transactions featureset",
        entities=["azureml:account:1"],
        stage="Development",
        specification=FeatureSetSpecification(path=transactions_featureset_spec_folder),
        tags={"data_type": "nonPII"},

        """
        transaction_fset_config = FeatureSet(
            name=name,
            version=version,
            description=description,
            entities=entities,
            stage=stage,
            specification=specification,
            tags=tags,
        )

        # feature store ml client
        fs_client = MLClient(
            AzureMLOnBehalfOfCredential(),
            self.subscription_id,
            self.resource_group_name,
            self.featurestore_name,
        )

        poller = fs_client.feature_sets.begin_create_or_update(transaction_fset_config)
        print(poller.result())

        return poller


    def set_feature_store_materialization(self,gen2_container_arm_id,storage_account_name,uai_name,verbose=0):
        """
        Enable offline store on the feature store by attaching offline materialization store and UAI
        """
        from azure.ai.ml.entities import (
            ManagedIdentityConfiguration,
            FeatureStore,
            MaterializationStore,
        )
        from azure.mgmt.msi import ManagedServiceIdentityClient
        from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
        from utilities.storage_tools import create_user_assigned_managed_identity,grant_rbac_permissions

        # Create or update User Managed Identity
        uai_principal_id, uai_client_id, uai_arm_id = create_user_assigned_managed_identity(
            AzureMLOnBehalfOfCredential(),
            uai_subscription_id=self.subscription_id,
            uai_resource_group_name=self.resource_group_name,
            uai_location=self.location,
            uai_name=uai_name,
        )

        # Get User Managed Identity
        msi_client = ManagedServiceIdentityClient(
            AzureMLOnBehalfOfCredential(), self.subscription_id
        )
        
        managed_identity = msi_client.user_assigned_identities.get(
            self.resource_group_name, uai_name
        )        
        
        uai_principal_id = managed_identity.principal_id
        uai_client_id = managed_identity.client_id
        uai_arm_id = managed_identity.id

        # Grant RBAC permissions
        # This utility function is created for ease of use in the docs tutorials. 
        # It uses standard azure API's. You can optionally inspect it `featurestore/setup/setup_storage_uai.py`
        grant_rbac_permissions(
            AzureMLOnBehalfOfCredential(),
            uai_principal_id,
            storage_subscription_id=self.subscription_id,
            storage_resource_group_name=self.resource_group_name,
            storage_account_name=storage_account_name,
            featurestore_subscription_id=self.subscription_id,
            featurestore_resource_group_name=self.resource_group_name,
            featurestore_name=self.featurestore_name,
        )

        # Configure Offline Store 
        offline_store = MaterializationStore(
            type="azure_data_lake_gen2",
            target=gen2_container_arm_id,
        )
        materialization_identity = ManagedIdentityConfiguration(
            client_id=uai_client_id, principal_id=uai_principal_id, resource_id=uai_arm_id
        )

        fs = FeatureStore(
            name=self.featurestore_name,
            offline_store=offline_store,
            materialization_identity=materialization_identity,
        )

        # feature store ml client
        fs_client = MLClient(
            AzureMLOnBehalfOfCredential(),
            self.subscription_id,
            self.resource_group_name,
            self.featurestore_name,
        )

        fs_poller = fs_client.feature_stores.begin_update(fs, update_dependent_resources=True)

        print(fs_poller.result())
        return fs_poller


    def set_feature_set_materialization(self,name,version,instance_type,spark_configuration,schedule=None,verbose=0):
        """

        Example
         name = "transactions"
        version = "1"
        instance_type="standard_e8s_v3"
        spark_configuration={
                "spark.driver.cores": 4,
                "spark.driver.memory": "36g",
                "spark.executor.cores": 4,
                "spark.executor.memory": "36g",
                "spark.executor.instances": 2,
            }
        schedule
        """
        from azure.ai.ml.entities import (
            MaterializationSettings,
            MaterializationComputeResource,
        )

        # feature store ml client
        fs_client = MLClient(
            AzureMLOnBehalfOfCredential(),
            self.subscription_id,
            self.resource_group_name,
            self.featurestore_name,
        )
            
        transactions_fset_config = fs_client._featuresets.get(name=name, version=version)

        transactions_fset_config.materialization_settings = MaterializationSettings(
            offline_enabled=True,
            resource=MaterializationComputeResource(instance_type=instance_type),
            spark_configuration=spark_configuration,
            schedule=schedule,
        )
        fs_poller = fs_client.feature_sets.begin_create_or_update(transactions_fset_config)
        print(fs_poller.result())
        return fs_poller
    
    def run_backfill(self,featureset_name,featureset_version,start_datetime,end_datetime):
        
        # feature store ml client
        fs_client = MLClient(
            AzureMLOnBehalfOfCredential(),
            self.subscription_id,
            self.resource_group_name,
            self.featurestore_name,
        )

        st = datetime.fromisoformat(start_datetime)
        ed = datetime.fromisoformat(end_datetime)

        poller = fs_client.feature_sets.begin_backfill(
            name=featureset_name,
            version=featureset_version,
            feature_window_start_time=st,
            feature_window_end_time=ed,
        )
        
        return poller





    
                                