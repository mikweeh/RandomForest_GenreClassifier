import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    # It takes the value of the project name from the config.yaml file in section
    # main and segment project_name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    # It takes the value of the experiment name from the config.yaml file in section
    # main and segment experiment_name
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute. By default all of them will be executed,
    # but in you specify in the command line the field main.execute_steps (for example,
    # mlflow run . -P main.execute_steps="preprocess") only those steps will be executed
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = list(config["main"]["execute_steps"])

    # First step - download
    if "download" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "download"),  # path to specific component folder
            "main",  # entry point "main" defined in the MLproject file of the folder above
            parameters={  # parameters defined in MLproject file of the folder above
                # input parameters
                "file_url": config["data"]["file_url"],  # Input defined in the config file
                # output parameters
                "artifact_name": "raw_data.parquet",  # Name of the artifact file
                "artifact_type": "raw_data",  # Type of the artifact
                "artifact_description": "Data as downloaded"  # Description of the artifact
            },
        )

    # Second step - preprocess
    if "preprocess" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),  # path to the component "preprocess" (the folder "preprocess")
            "main",  # entry point "main" defined in the MLproject file of the folder "preprocess"
            parameters={  # parameters defined in MLproject file of the folder "preprocess"
                # Input parameters
                # See that the input parameter of this run is the output of the previous one,
                # so the "input_artifact" here is the name of "artifact_name" of the previous run,
                # including the version
                "input_artifact": "raw_data.parquet:latest",  # Name of the artifact input including the version
                # Output parameters
                "artifact_name": "preprocessed_data.csv",  # Choosen name for the artifact file (the output)
                "artifact_type": "preprocessed_data",
                "artifact_description": "Data with preprocessing applied"
            },
        )

    # Third step - check_data
    # We do some statistical tests here. Some of them are deterministic tests and some of them are not.
    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),  # path to specific component folder
            "main",  # entry point "main" defined in the MLproject file of the folder above
            parameters={  # parameters defined in MLproject file of the folder above
                # Input parameters
                # In this case we have three inputs and no output. The first input is the data generated in the
                # previous run, including the version.
                "sample_artifact": "preprocessed_data.csv:latest",
                # The second data is a reference dataset. We are going to do some non-deterministic
                # tests, so we want another dataset to compare with. This is it, and we point to it
                # as indicated in the configuration file.
                "reference_artifact": config["data"]["reference_dataset"],
                # The third input is the kolmogorov-smirnov threshold taken form the configuration file
                "ks_alpha": config["data"]["ks_alpha"]
            },
        )

    # Fourth step - segregate
    # Here we divide the data between training and testing. The results are 2 artifacts, so pay attention
    # to what we do.
    if "segregate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),  # path to specific component folder
            "main",  # entry point "main" defined in the MLproject file of the folder above
            parameters={  # parameters defined in MLproject file of the folder above
                # Input parameters
                "input_artifact": "preprocessed_data.csv:latest",  # The output of the second run
                "test_size": config["data"]["test_size"],  # Test percentage as indicated in config
                "stratify": config["data"]["stratify"],  # Name of the column to stratify
                # Output parameters
                # We are going to generate two files here, one for training and the other for testing.
                # So the next output is just the root of the filenames, i.e. data_train.csv and data_test.csv
                # filename
                "artifact_root": "data",
                "artifact_type": "segregated_data"  # Artifact output type
            },
        )

    # Fifth step - random_forest
    # Here we have a lot of parameters to pass to the run, because the random forest algorithm
    # has a lot of configuration parameters. To avoid having to write so many thing here, we do a trick.
    # We create a new file on the run
    if "random_forest" in steps_to_execute:
        # Select the name and path of the new file
        model_config = os.path.abspath("random_forest_config.yml")
        # Create this file and write all the random_forest_pipeline segment of the config file
        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        # Run the run
        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),  # Path to specific component folder
            "main",  # Entry point "main" defined in the MLproject file of the folder above
            parameters={  # Parameters defined in MLproject file of the folder above
                # Input parameters
                "train_data": "data_train.csv:latest",  # Input artifact generated in the previous run
                "model_config": model_config,  # Input parameters file generated in this step
                "random_seed": config["main"]["random_seed"],  # Other parameters from the configuration file
                "val_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"],
                # Output parameters
                # Name for the output artifact, that is our inference pipeline
                "export_artifact": config["random_forest_pipeline"]["export_artifact"]
            },
        )

    # Last step - evaluate
    # Get the generated inference pipeline and test it against the test dataset
    if "evaluate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                # Input parameters
                # This is the latest versión of the export_artifact of the previous run
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                # The test dataset generated in step 4 (segregate) 
                "test_data": "data_test.csv:latest"
            },
        )


if __name__ == "__main__":
    go()
