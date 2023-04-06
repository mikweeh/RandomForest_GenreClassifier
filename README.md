# RandomForest_GenreClassifier
Genre classifier through Random Forest algorithm developed with MLflow.

To run this program just download the repository, open a terminal in the main folder and run:

`$ mlflow run .`

Do you want to do it easier? While being logged in Weights & Biases, open a terminal. Make sure you have mlflow installed. And type:

`$ mlflow run -v v1.0.1 https://github.com/mikweeh/RandomForest_GenreClassifier.git`

If you find a newer version, of course, put the correct number in the command above.

Remember that you can change the parameters of the config file from the terminal. For example, change the project name:

`$ mlflow run -v v1.0.1 https://github.com/mikweeh/RandomForest_GenreClassifier.git -P hydra_options="main.project_name=remote_execution"`
