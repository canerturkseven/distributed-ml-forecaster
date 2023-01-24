#%%
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import mlflow
from mlforecastflow.meta_model import MetaModel
from mlforecastflow.data.loader import load_wallmart


spark = (
    SparkSession.builder.master("local[4]")
    .config("spark.driver.memory", "30g")
    .config("spark.sql.shuffle.partitions", 2)
    .config("spark.driver.maxResultSize", "5g")
    .config("spark.sql.execution.arrow.enabled", "true")
    .config("spark.executor.heartbeatInterval", "36000000s")
    .config("spark.network.timeout", "72000000s")
    .getOrCreate()
)

df_train, df_test = load_wallmart()

def hyperparam_space_fn(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
    }


with mlflow.start_run() as run:
    run_id = run.info.run_id
    model = MetaModel(
        run_id=run_id,
        group_col="cat_id",
        id_cols=["id"],
        date_col="date",
        date_frequency="days",
        n_cv_splits=1,
        max_forecast_horizon=7,
        model_horizon=7,
        target_col="sales",
        tracking_uri="./mlruns",
        max_hyperparam_evals=5,
        metric="wmape",
        hyperparam_space_fn=hyperparam_space_fn,
    )
mlflow.pyfunc.log_model(python_model=model, artifact_path="meta_model")
model.train(df_train)