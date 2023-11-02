import os, sys

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/usr/lib/spark'
sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python'))
sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python', 'lib', 'py4j-src.zip'))
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

from ctools.s3 import s3open
import numpy as np

from phsafe_safetab_reader.cef_validator_classes import CEF20_UNIT, CEF20_PER

import configparser

"""The interface for PHSafe readers."""

from abc import ABC
from typing import List
from pyspark.sql import DataFrame as SparkDataFrame

class CEFReader(ABC):
    """The interface for PHSafe readers."""

    def __init__(self, config_path: str, state_filter: List[str]):

        config = configparser.ConfigParser()

        if config_path.startswith("s3://"):
            with s3open(config_path) as s3_config_file:
                config.read_file(s3_config_file)
        else:
            config.read(config_path)

        paths = config['paths']

        """Initialize reader with config path and the state FIPS codes to read."""
        spark = SparkSession.builder.getOrCreate()

        per_validator = CEF20_PER
        unit_validator = CEF20_UNIT

        per_s3_dir = paths["per_dir"]
        unit_s3_dir = paths["unit_dir"]
        per_s3_dir_pr = paths["per_dir_pr"]
        unit_s3_dir_pr = paths["unit_dir_pr"]
        geo_s3_dir = paths["grfc_dir"]
        per_file_format = paths["per_file_format"]
        unit_file_format = paths["unit_file_format"]

        # Puerto Rico files may be in a different directory
        # create person dataframe for specified states/PR
        if len(state_filter):
            per_files = [per_s3_dir + per_file_format % (code,) for code in state_filter if code != "72"]
            if "72" in state_filter:
                per_files.append(per_s3_dir_pr + per_file_format % "72")
            per_files = ",".join(per_files)
        else:
            per_files = per_s3_dir
            per_files = per_files + "," + per_s3_dir_pr + per_file_format % "72"

        per_rdd = spark.sparkContext.textFile(per_files)
        per_rdd = per_rdd.map(lambda line: per_validator.parse_line(line))
        self.per_df = spark.createDataFrame(per_rdd).cache()

        # create unit dataframe for specified states/PR
        if len(state_filter):
            unit_files = [unit_s3_dir + unit_file_format % (code,) for code in state_filter if code != "72"]
            if "72" in state_filter:
                unit_files.append(unit_s3_dir_pr + unit_file_format % "72")
            unit_files = ",".join(unit_files)
        else:
            unit_files = unit_s3_dir
            unit_files = unit_files + "," + unit_s3_dir_pr + unit_file_format % "72"

        unit_rdd = spark.sparkContext.textFile(unit_files)
        unit_rdd = unit_rdd.map(lambda line: unit_validator.parse_line(line))
        self.unit_df = spark.createDataFrame(unit_rdd).cache()

        if len(state_filter):
            import boto3

            bucket = geo_s3_dir.split('/')[2]
            folder = '/'.join(geo_s3_dir.split('/')[3:])
            s3 = boto3.resource('s3')
            s3_bucket = s3.Bucket(bucket)
            geo_files = [f.key.split(folder)[1] for f in s3_bucket.objects.filter(Prefix=folder).all()]
            geo_files = list(filter(lambda file: file[11:13] in state_filter, geo_files))
            geo_files = ['s3://' + bucket + '/' + folder + file for file in geo_files]
        else:
            geo_files = geo_s3_dir

        self.geo_df = spark.read \
            .option("header", "true") \
            .option("delimiter", "|") \
            .csv(geo_files) \
            .withColumn("OIDTABBLK", F.col("OIDTABBLK").cast(LongType()))

        # for each unit, project RTPYE and join it to unit's geo data to make geo_df frame
        unit_id_df = self.unit_df.select("rtype", "mafid", "oidtb")
        self.geo_df = unit_id_df.join(self.geo_df, unit_id_df.oidtb == self.geo_df.OIDTABBLK, "inner")

        def capitalize_column_names(df):
            for col in df.columns:
                df = df.withColumnRenamed(col, col.upper())
            return df

        self.per_df = capitalize_column_names(self.per_df)
        self.unit_df = capitalize_column_names(self.unit_df)
        self.geo_df = capitalize_column_names(self.geo_df)

        # Strategy for making NPF column:
        # 1. .join: join unit and per frames on MAFID
        # 2. .where: filter out units without RTYPE=2 and with FINAL_POP<2
        # 3. .withColumn: make a binary column for each join-row indicating a qualifying relship type or not
        # 4. .groupBy: group join data by MAFID and count qualifying NPF relationships
        # 5. .where: filter out NPF=1 (there is only a householder and no other qualifying family, I think)

        npf_df = self.per_df.select(F.col("MAFID").alias("per_MAFID"), F.col("RELSHIP"))\
                           .join(self.unit_df, F.col("MAFID") == F.col("per_MAFID"), "inner")\
                           .where((F.col("FINAL_POP") >= 2) & (F.col("RTYPE") == 2))\
                           .withColumn("has_npf_relship", F.when(
                                              (F.col("RELSHIP") == '20') |
                                              (F.col("RELSHIP") == '21') |
                                              (F.col("RELSHIP") == '23') |
                                              (F.col("RELSHIP") == '25') |
                                              (F.col("RELSHIP") == '26') |
                                              (F.col("RELSHIP") == '27') |
                                              (F.col("RELSHIP") == '28') |
                                              (F.col("RELSHIP") == '29') |
                                              (F.col("RELSHIP") == '30') |
                                              (F.col("RELSHIP") == '31') |
                                              (F.col("RELSHIP") == '32') |
                                              (F.col("RELSHIP") == '33'), 1).otherwise(F.lit(0)))\
                           .groupBy("per_MAFID").agg(F.sum("has_npf_relship").alias("NPF"))\
                           .where(F.col("NPF") > 1)

        # left-join unit data and NPF counts, filling in zero when NPF is absent
        self.unit_df = self.unit_df.join(npf_df, F.col("MAFID") == F.col("per_MAFID"), "left").fillna({"NPF": 0})

        # cast FINAL_POP and HHSPAN to Longs and replace nulled non-numerics (i.e., whitespace) with 0
        self.unit_df = self.unit_df.withColumn("FINAL_POP", F.col("FINAL_POP").cast(LongType()))\
                                   .withColumn("HHSPAN", F.col("HHSPAN").cast(LongType()))\
                                   .fillna({"HHSPAN": 0})

        # make NPF nullable per PHSafe spec
        def set_df_columns_nullable(spark, df, column_list, nullable=True):
            for struct_field in df.schema:
                if struct_field.name in column_list:
                    struct_field.nullable = nullable
            df_mod = spark.createDataFrame(df.rdd, df.schema)
            return df_mod

        self.unit_df = set_df_columns_nullable(spark, self.unit_df, ["NPF", "HHSPAN"])

        # replace HHRACE whitespace with "00" per PHSafe spec
        self.unit_df = self.unit_df.withColumn("HHRACE", F.regexp_replace(F.col("HHRACE"), "^\s+$", "00"))

        # perform projection and ordering of columns per specification
        self.per_df = self.per_df.select("RTYPE", "MAFID", "QAGE", "CENHISP", "CENRACE", "RELSHIP")
        self.unit_df = self.unit_df.select("RTYPE", "MAFID", "FINAL_POP", "NPF", "HHSPAN", "HHRACE", "TEN", "HHT", "HHT2", "CPLT")
        self.geo_df = self.geo_df.select("RTYPE", "MAFID", "TABBLKST", "TABBLKCOU", "TABTRACTCE", "TABBLK", "TABBLKGRPCE", "REGIONCE", "DIVISIONCE", "PLACEFP", "AIANNHCE")
        print(per_files)
        print(unit_files)
        print(geo_files)
    def get_geo_df(self) -> SparkDataFrame:
        """Return a spark dataframe derived from the CEF Unit file and the GRFC.

        This should contain only the state codes specified by the state_filter.

        Returns:
            A spark dataframe containing all of the expected columns from `geo_df`_ in
                `Appendix A`_.
        """
        return self.geo_df

    def get_unit_df(self) -> SparkDataFrame:
        """Return a spark dataframe derived from the CEF unit file.

        This should contain only records with MAFIDs corresponding to the state codes
        specified by the state_filter.

        Returns:
            A spark dataframe containing all of the expected columns from `unit_df`_ in
                `Appendix A`_.
        """
        return self.unit_df

    def get_person_df(self) -> SparkDataFrame:
        """Return a spark dataframe derived from the CEF person file.

        This should contain only records with MAFIDs corresponding to the state codes
        specified by the state_filter.

        Returns:
            A spark dataframe containing all of the expected columns from `person_df`_
                in `Appendix A`_.
        """
        return self.per_df


if __name__ == "__main__":
    # Run using spark-submit
    #print("SMAPLES BELOW")
    #print(per_df.rdd.takeSample(False, 1))
    #print(unit_df.rdd.takeSample(False, 1))


    CEFReader("cef_config.ini", ["72", '01'])
