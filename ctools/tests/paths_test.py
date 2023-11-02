import pytest
import pathlib
import ctools.paths as paths
from ctools.env import census_getenv

def test_mkpath_relpath_none():
    test_path = "/mnt/users/jbond007/ctools"
    mkpath_result = paths.mkpath(test_path)
    print(f"test_mkpath_relpath_none(): {mkpath_result}")
    assert test_path == mkpath_result


def test_mkpath_relpath_value():
    test_path = "/mnt/users/jbond007/ctools"
    test_relpath = "/mnt/users/mallory/ctools"
    mkpath_result = paths.mkpath(test_path, test_relpath)
    print(f"test_mkpath_relpath_value(): {mkpath_result}")
    assert mkpath_result == "/mnt/users/mallory/ctools"


def test_substvars_matches():
    test_path = "${DAS_S3INPUTS}/mnt/users/jbond007/ctools"
    matched_variable = census_getenv("DAS_S3INPUTS")
    substvars_result = paths.substvars(test_path)
    print(f"test_substvars_matches(): {substvars_result}")
    assert substvars_result == f"{matched_variable}/mnt/users/jbond007/ctools"


def test_substvars_matches_no_brackets():
    test_path = "$DAS_S3INPUTS/mnt/users/jbond007/ctools"
    matched_variable = census_getenv("DAS_S3INPUTS")
    substvars_result = paths.substvars(test_path)
    print(f"test_substvars_matches_no_brackets(): {substvars_result}")
    assert substvars_result == f"{matched_variable}/mnt/users/jbond007/ctools"


def test_substvars_no_matches():
    test_path = "/mnt/users/jbond007/ctools"
    substvars_result = paths.substvars(test_path)
    print(f"test_substvars_no_matches(): {substvars_result}")
    assert substvars_result == test_path


def test_substvars_key_error():
    test_path = "${}/mnt/users/jbond007/ctools"
    with pytest.raises(KeyError) as e_info:
        paths.substvars(test_path)
        print(f"test_substvars_key_error(): {e_info}.")


def test_substvars_multi_key():
    test_path = "${DAS_S3INPUTS}/${DAS_DASHBOARD_URL}"
    first_matched_variable = census_getenv("DAS_S3INPUTS")
    second_matched_variable = census_getenv("DAS_DASHBOARD_URL")
    substvars_result = paths.substvars(test_path)
    print(f"test_substvars_multi_key(): {substvars_result}")
    assert substvars_result == f"{first_matched_variable}/{second_matched_variable}"
