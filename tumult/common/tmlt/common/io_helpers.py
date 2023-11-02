"""Helper functions for reading/writing from different sources."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import glob
import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

import boto3
import pandas as pd
from smart_open import open  # pylint: disable=redefined-builtin


def read_csv(*args, **kwargs):
    """Wrapper around pd.read_csv."""
    with open(args[0]) as f:
        return pd.read_csv(f, *args[1:], **kwargs)


def is_s3_path(path: str) -> bool:
    """Returns whether the path is for an s3 bucket.

    Args:
        path: path to test.
    """
    return re.match(r"s3a?://", path) is not None


def get_logger_stream(
    logger_name: Optional[str] = None,
) -> Tuple[logging.Logger, io.StringIO]:
    """Returns the logger and StringIO stream associated with it.

    Args:
        logger_name: Name of the logger.
    """
    logger = logging.getLogger()
    if logger_name:
        logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    io_stream = io.StringIO()
    handler = logging.StreamHandler(io_stream)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger, io_stream


def write_log_file(log_file: str, content: str) -> None:
    """Write log file to S3 storage or local folder.

    Args:
        log_file: The log file name with file path.
        content: Contents of the log file.
    """
    if is_s3_path(log_file):
        parsed_s3_path = urlparse(log_file)
        bucket_name = parsed_s3_path.netloc
        obj_key = parsed_s3_path.path.lstrip("/")
        s3 = boto3.client("s3")
        s3.put_object(Body=content, Bucket=bucket_name, Key=obj_key)
    else:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(content)


def to_csv_with_create_dir(df: pd.DataFrame, filename: str, **kwargs: Any):
    """Call pandas to_csv after creating any missing intermediate directories.

    Args:
        df: The pandas DataFrame to write.
        filename: The file name.
        **kwargs: Keyword arguments that will be passed to pd.DataFrame.to_csv.
    """
    if not is_s3_path(filename):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, **kwargs)


def multi_read_csv(path: str, **kwargs: Any) -> pd.DataFrame:
    """Concatenate all csv files in a directory using pandas read_csv.

    Skips files that are empty or do not end with ".csv".

    Args:
        path: The directory containing the files to read.
        kwargs: Keyword arguments that will be passed to pd.DataFrame.read_csv.
    """
    dfs = []
    for filename in glob.glob(f"{path}/*.csv"):
        if os.stat(filename).st_size > 0:
            df = read_csv(filename, **kwargs)
            dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)
