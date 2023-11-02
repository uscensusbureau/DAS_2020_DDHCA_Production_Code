import pytest
import glob
import os
import time
import boto3
import sys
import subprocess

sys.path.append( os.path.join( os.path.dirname(__file__), ".."))
sys.path.append( os.path.join( os.path.dirname(__file__), "../../../das_decennial"))

from ctools import s3, data_cat


@pytest.fixture(scope="module")
def testroot() -> None:
    # Setup code

    pid = os.getpid()
    timet = int(time.time())
    root = f"{os.environ['DAS_S3ROOT']}/tmp/data_cat_{timet}-{pid}"
    # root = f"{os.path.dirname(__file__)}"

    TESTFILES = glob.glob(os.path.join( f"{os.path.dirname(__file__)}/data_cat_test_files/", "*.txt"))
    print(f"TESTFILES {TESTFILES}", file=sys.stderr)
    s3urls = set()
    for path in TESTFILES:
        s3url = os.path.join(root, os.path.basename(path))

        (bucket, key) = s3.get_bucket_key(s3url)
        s3.put_object( bucket, key, path)
        s3urls.add(s3url)
        print(f"creating {s3url}")

    yield root
    # Takedown code

    for s3name in s3urls:
        (bucket, key) = s3.get_bucket_key(s3url)
        s3.delete_object( bucket, key )


OK='OK'
FAIL='FAIL'

@pytest.mark.skip
def test_userid() -> None:
    if os.getenv('USER')!='hadoop':
        raise RuntimeError("data_cat must be run as the hadoop user")


def run_data_cat(testroot: str, file1: str, file2: str, result: str) -> None:
    data_cat = os.path.join( os.path.dirname(os.path.dirname(__file__)), 'data_cat.py')
    cp = subprocess.run([sys.executable, data_cat, os.path.join(testroot,file1), 'file1', os.path.join(testroot,file2), 'file2'])
    if result==OK:
        assert cp.returncode==0
    elif result==FAIL:
        assert cp.returncode!=0,"run should fail, but it passed"
    else:
        assert "result code passed to run_data_cat is invalid"


def test_file_not_exist(testroot: str) -> None:
    # These tests can run without spark
    run_data_cat(testroot,"data_cat_test_files/good-file1.txt","data_cat_test_files/no-such-file.txt",FAIL)
    run_data_cat(testroot,"data_cat_test_files/no-such-file.txt","data_cat_test_files/good-file2.txt",FAIL)


def test_nodata(testroot: str) -> None:
    run_data_cat(testroot,"data_cat_test_files/good-file1.txt","data_cat_test_files/bad-nodata.txt",FAIL)


@pytest.mark.skip
def test_happy_path(testroot: str) -> None:
    run_data_cat(testroot,"data_cat_test_files/good-file1.txt","data_cat_test_files/good-file2.txt",OK)
