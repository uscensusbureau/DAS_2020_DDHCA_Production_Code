from typing import List
from pyspark.sql import SparkSession

def get_pop_count(popFiles: List[str]):
   """ Return total counts of population group details that read directly from the csv files. This method is called internally.
        Return:
           total counts of population group details
   """
   spark = SparkSession.builder.getOrCreate()
   pop_counts = []
   for f in popFiles:
      pop_df = spark.read \
                .option("header", "true") \
                .option("delimiter", "|") \
                .csv(f)
      pop_counts.append(pop_df.count())

   return pop_counts


def get_geo_pop_file_list(geo: bool, geo_s3_dir: str, pop_s3_dir: str, state_filter: List[str]):
    """ Return the list of geo or pop files. This method is called internally.
        Return:
           list of geo or population group details files from the inputs
    """

    import boto3
    if geo:
        bucket = geo_s3_dir.split('/')[2]
        folder = '/'.join(geo_s3_dir.split('/')[3:])
    else:
        bucket = pop_s3_dir.split('/')[2]
        folder = '/'.join(pop_s3_dir.split('/')[3:])

    s3 = boto3.resource('s3')
    s3_bucket = s3.Bucket(bucket)

    files_list = [f.key.split(folder)[1] for f in s3_bucket.objects.filter(Prefix=folder).all()]


    if geo:
        files_list = list(filter(lambda file: file[11:13] in state_filter, files_list))
    else:
        files_list = list(filter(lambda file: len(file) > 0, files_list))

    files_list = ['s3://' + bucket + '/' + folder + file for file in files_list]

    return files_list
