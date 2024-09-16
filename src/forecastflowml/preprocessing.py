import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from typing import Dict, List, Optional, Union


def _history_length(df, id_col, date_col):
    w = Window.partitionBy(id_col).orderBy(date_col)
    df = df.withColumn("history_length", F.row_number().over(w))
    return df


def _lag_window_summarizer(df, id_col, target_col, date_col, features):
    w1 = Window.partitionBy(id_col).orderBy(date_col)
    for key, values in features.items():
        if key == "lag":
            for lag in values:
                df = df.withColumn(f"lag_{lag}", F.lag(target_col, lag).over(w1))
        else:
            for window, lag in values:
                w2 = w1.rowsBetween(-(lag + window - 1), -lag)
                df = df.withColumn(
                    f"window_{window}_lag_{lag}_{key}",
                    F.expr(f"{key}({target_col})").over(w2),
                )
    return df


def _count_consecutive_values(df, id_col, value_col, date_col, value, lags):
    w1 = (
        Window.partitionBy(id_col)
        .orderBy(date_col)
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    w2 = (
        Window.partitionBy(id_col, "value_group")
        .orderBy(date_col)
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    df = (
        df.withColumn("mask", F.when(F.col(value_col) == value, 1).otherwise(0))
        .withColumn("value_group", F.sum(1 - F.col("mask")).over(w1))
        .withColumn("count", F.sum("mask").over(w2))
    )

    w3 = Window.partitionBy(id_col).orderBy(date_col)
    for lag in lags:
        output_col = f"count_consecutive_value_lag_{lag}"
        df = df.withColumn(output_col, F.lag("count", lag).over(w3))
    df = df.drop("mask", "value_group", "count")
    return df


def _date_features(df, date_col, features):
    supported_features = [
        "day_of_week",
        "day_of_year",
        "day_of_month",
        "week_of_year",
        "week_of_month",
        "weekend",
        "month",
        "quarter",
        "year",
    ]

    not_supported = set(features) - set(supported_features)
    if len(not_supported) > 0:
        raise ValueError(f"{', '.join(not_supported)} feature(s) not supported.")

    for feature in features:
        if feature == "day_of_week":
            df = df.withColumn(
                feature, F.dayofweek(F.col(date_col)).cast("tinyint") - 1
            )
            df = df.withColumn(
                feature, F.when(F.col(feature) == 0, 7).otherwise(F.col(feature))
            )
        if feature == "day_of_year":
            df = df.withColumn(feature, F.dayofyear(F.col(date_col)).cast("smallint"))
        if feature == "day_of_month":
            df = df.withColumn(feature, F.dayofmonth(F.col(date_col)).cast("tinyint"))
        if feature == "week_of_year":
            df = df.withColumn(feature, F.weekofyear(F.col(date_col)).cast("tinyint"))
        if feature == "week_of_month":
            df = df.withColumn(
                feature, F.ceil(F.dayofmonth(F.col(date_col)) / 7).cast("tinyint")
            )
        if feature == "weekend":
            df = df.withColumn(
                feature,
                F.when(F.dayofweek(F.col(date_col)).isin([1, 7]), 1)
                .otherwise(0)
                .cast("tinyint"),
            )
        if feature == "month":
            df = df.withColumn("month", F.month(F.col(date_col)).cast("tinyint"))
        if feature == "quarter":
            df = df.withColumn("quarter", F.quarter(F.col(date_col)).cast("tinyint"))
        if feature == "year":
            df = df.withColumn("year", F.year(F.col(date_col)).cast("smallint"))
    return df


class FeatureExtractor:
    """Extract features from time series

    Parameters
    ----------
    id_col
        Id column name.
    date_col
        Date column name.
    target_col
        Target column name.
    lag_window_features
        Dictionary that contains different types of functions as keys and
        their corresponding lag-window arguments as values.
        The lag argument specifies how many units in the past the window should start,
        while the window specifies the size of the window to apply the function across.

        - For the lag function, only list of integers needs to be provided.
        - For all other functions, list of lists such that [[window, lag]] needs to be provided.

        ========= ====================================================================
        function  example
        ========= ====================================================================
        lag       {"lag": [1, 2, 3, 4]}
        mean      {"mean": [[window, lag] for lag in [1, 2, 3] for window in [7, 14]]}
        stddev    {"stddev": [window, lag] for lag in [1, 2, 3] for window in [7, 14]}
        ========= ====================================================================

        The logic of the code is represented visually using symbols:

        - o: denotes the time stamp for which the window is summarized to
        - x: represents other time stamps within the window being summarized.
        - -: is used to denote observations, past or future, that are not part of the window.

        ==== ====== ============================
        lag  window calculation
        ==== ====== ============================
        1    3      [- - - - - * * * o - - - -]
        2    3      [- - - - * * * - o - - - -]
        1    5      [- - - * * * * * o - - - -]
        ==== ====== ============================

        Keys needs to be a native pyspark functions.
    date_features
        Date features to extract: day_of_week, day_of_year, day_of_month,
        week_of_year, week_of_month, weekend, month, quarter, year.
    count_consecutive_values
        Counts consecutive apperance of spesific value. Needs to be a dictionary that
        contains value for counting, and lags for how many units in the past the counting should start,

        - Example: count_consecutive_values={"value": 0,  "lags": [7, 14, 21, 28]}
    history_length
        Whether to count number of time periods after the start of time series.
    """

    def __init__(
        self,
        id_col: str,
        date_col: str,
        target_col: str,
        max_forecast_horizon: int = 1,
        lags: int = 0,
        lag_window_features: Optional[Dict[str, List[Union[int, List[int]]]]] = None,
        date_features: List[str] = None,
        count_consecutive_values: Optional[
            Dict[str, List[Union[int, List[int]]]]
        ] = None,
        history_length: bool = False,
    ):
        self.id_col = id_col
        self.date_col = date_col
        self.target_col = target_col
        self.max_forecast_horizon = max_forecast_horizon
        self.lag_window_features = lag_window_features
        self.date_features = date_features
        self.count_consecutive_values = count_consecutive_values
        self.history_length = history_length

    def _check_input_type(self, df: pyspark.sql.DataFrame) -> None:
        if not isinstance(df, pyspark.sql.dataframe.DataFrame):
            raise NotImplementedError(
                "Input is expected to be pyspark.sql.dataframe.DataFrame"
            )
        else:
            pass

    def transform(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """Extract features

        Parameters
        ----------
        df
            DataFrame to extract features.

        Returns
        -------
            DataFrame with features added.
        """
        self._check_input_type(df)

        if self.lag_window_features is not None:
            df = _lag_window_summarizer(
                df,
                self.id_col,
                self.target_col,
                self.date_col,
                self.lag_window_features,
            )
        if self.count_consecutive_values is not None:
            df = _count_consecutive_values(
                df,
                self.id_col,
                self.target_col,
                self.date_col,
                self.count_consecutive_values["value"],
                self.count_consecutive_values["lags"],
            )
        if self.history_length:
            df = _history_length(df, self.id_col, self.date_col)
        if self.date_features is not None:
            df = _date_features(df, self.date_col, self.date_features)

        return df
