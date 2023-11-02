from tmlt.core.utils.arb import Arb

from pyspark.sql import SparkSession
from pyspark.sql import types as st
from pyspark.sql import functions as sf


spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame([(1, 2), (2, 3)], schema=["A", "B"])

newColumn = sf.udf(lambda: int(float(Arb.from_int(10))), st.IntegerType())
df = df.withColumn("X", newColumn())
assert df.agg(sf.sum("X").alias("SUM")).collect()[0]["SUM"] == 20
