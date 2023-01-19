#%%
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import mlflow
from meta_model import MetaModel
%load_ext autoreload
%autoreload 2
#%%
def main():

    spark = (
        SparkSession.builder.master("local[4]")
        .config("spark.driver.memory", "30g")
        .config("spark.sql.shuffle.partitions", 4)
        .config("spark.driver.maxResultSize", "5g")
        .config("spark.sql.execution.arrow.enabled", "true")
        .config("spark.executor.heartbeatInterval", "36000000s")
        .config("spark.network.timeout", "72000000s")
        .getOrCreate()
    )
    spark.sparkContext.setCheckpointDir("/checkpoint")
    df = spark.read.parquet("../example_dataset/feature/")
    df = df.filter(F.col("date") < "2014-01-01")
    df = df.withColumn("date", F.to_timestamp("date"))

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    def hyperparam_space_fn(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        }

    with mlflow.start_run() as run:
        model = MetaModel(
            run_id=run.info.run_id,
            group_col="cat_id",
            id_cols=["id"],
            date_col="date",
            date_frequency="days",
            n_cv_splits=3,
            max_forecast_horizon=7,
            model_horizon=7*8,
            target_col="sales",
            tracking_uri="http://127.0.0.1:5000",
            max_hyperparam_evals=5,
            metric="wmape",
            hyperparam_space_fn=hyperparam_space_fn,
        )
        mlflow.pyfunc.log_model(python_model=model, artifact_path="meta_model")
        model.train(df)

    model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/meta_model")
    model.predict(df).show()

if __name__ == "__main__":
    main()
# %%
