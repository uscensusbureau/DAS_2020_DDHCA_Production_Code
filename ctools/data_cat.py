#!/usr/bin/env python3
#
"""
data_cat.py
Combine two S3 files with metadata using Spark

"""

import sys
import os
import boto3
import time

THIS_DIR = os.path.dirname( os.path.abspath(__file__))
DAS_DECENNIAL_DIR = os.path.join( os.path.dirname(THIS_DIR), "das_decennial")
# assert os.path.exists(DAS_DECENNIAL_DIR), f"Cannot find {DAS_DECENNIAL_DIR}"

sys.path.append(DAS_DECENNIAL_DIR)

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/usr/lib/spark'
sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python'))
sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python', 'lib', 'py4j-src.zip'))

# Note: ctools is not available on the mappers unless pyfiles are shipped out
# So be careful about running functions in map!
#
sys.path.append( os.path.join( os.path.dirname(__file__), "..") )
from ctools import s3, clogging, cspark

if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/usr/lib/spark'

sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python'))
sys.path.append(os.path.join(os.environ['SPARK_HOME'], 'python', 'lib', 'py4j-src.zip'))

from pyspark.sql import SparkSession
from pyspark import RDD



def extract_comments(rdd: RDD) -> str:
    """Returns an RDD with the comments"""
    comments = rdd.filter(lambda line:line[0:1]=='#')
    return comments


def extract_schema(rdd: RDD) -> str:
    """Returns the schema as a string"""
    return rdd.filter(lambda line:line[0:1]!='#').first()


def extract_data(rdd: RDD) -> RDD:
    """Return all of the data, no comments or schema"""
    schema = extract_schema(rdd)
    return rdd.filter(lambda line:line[0:1]!='#' and line!=schema)


if __name__=="__main__":
    from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
    parser = ArgumentParser( formatter_class = ArgumentDefaultsHelpFormatter,
                             description="Combine two files" )
    parser.add_argument("file1", help="File1 in s3")
    parser.add_argument("file1prefix", nargs='?', help="Metdata prefix for file1")
    g = parser.add_argument_group('file2')
    g.add_argument("file2", nargs='?', help="File1 in s3")
    g.add_argument("file2prefix", nargs='?', help="Metdata prefix for file1")
    g.add_argument("outfile", nargs='?', help='Output file')
    parser.add_argument("--debug", action='store_true', help='print additional debugging info')
    parser.add_argument("--num_executors", type=int, default=8)
    parser.add_argument("--schema", action='store_true', help='just dump the schema of file1')
    parser.add_argument("--comments", action='store_true', help='just dump the comments of file1')
    parser.add_argument("--collect", action='store_true', help='collect the results to the head-end')
    clogging.add_argument(parser)
    args  = parser.parse_args()
    clogging.setup(args.loglevel,
                   syslog=True,
                   filename=args.logfilename,
                   log_format=clogging.LOG_FORMAT,
                   syslog_format=clogging.YEAR + " " + clogging.SYSLOG_FORMAT)

    if not s3.s3exists(args.file1):
        raise FileNotFoundError(args.file1)

    if args.file2 and not s3.s3exists(args.file2):
        raise FileNotFoundError(args.file2)


    spark = cspark.spark_session(logLevel='ERROR',
                                 num_executors=args.num_executors)
    # If we are running under py.test, spark_session can return None
    if 'PYTEST_CURRENT_TEST' in os.environ:
        if spark is None:
            exit(0)

    sc = spark.sparkContext

    rdd1 = sc.textFile(args.file1)
    if args.schema:
        schema = extract_schema(rdd1)
        print("Schema:",schema)

    if args.comments:
        comments = extract_comments(rdd1)
        print("comments:")
        print("\n".join(comments.collect()))

    rdd2 = sc.textFile(args.file2)

    print(f"input1: {args.file1}")
    print(f"input2: {args.file2}")

    # Make sure that both have the same schema
    schema1 = extract_schema(rdd1)
    schema2 = extract_schema(rdd2)
    if schema1 != schema2:
        print(f"schema1 != schema2")
        print(f"schema1: {schema1}", file=sys.stderr)
        print(f"schema2: {schema2}", file=sys.stderr)
        assert schema1==schema2

    print(f"schema: {schema1}")

    data1 = extract_data(rdd1)
    data2 = extract_data(rdd2)

    # Get the comments for both and prefix them with the
    # This assumes that the comments are short.
    comments0 = [f'# combined {time.asctime()}',
                 f'# input1: {args.file1}  prefix: {args.file1prefix}',
                 f'# input2: {args.file2}  prefix: {args.file2prefix}',
                 f'# records: {data1.count()+data2.count()} (combined)']
    comments1 = [f'# {args.file1prefix}{line[1:]}' for line in extract_comments(rdd1).collect()]
    comments2 = [f'# {args.file2prefix}{line[1:]}' for line in extract_comments(rdd2).collect()]
    header    = comments0 + ['#'] + comments1 + ['#'] + comments2 + [schema1]

    # Make this an rdd where each line is numbered as a (k,v)
    header_rdd = sc.parallelize( header )

    # Sort and write out the combined file
    # Note that order is preserved
    # https://stackoverflow.com/questions/29284095/which-operations-preserve-rdd-order
    # https://stackoverflow.com/questions/31820230/ordering-of-rows-in-javardds-after-union
    # Previously we had added a (k,v) and sorted on that, but it doesn't seem to be necessary
    combined = header_rdd.union( data1 ).union( data2 )

    # And write it out!
    if args.outfile:
        print(f"writing {args.outfile}")
        combined.saveAsTextFile(args.outfile)
        print("done")

    # Dump if requested (could be long!)
    if args.collect:
        for line in combined.collect():
            print(line)

    exit(0)
