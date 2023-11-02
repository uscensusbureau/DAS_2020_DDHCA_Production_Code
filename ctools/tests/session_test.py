#!/usr/bin/env python3
# Test session code

import os
import sys
import warnings
import pytest
import logging
import boto3

import ctools.session as session

def test_get_session__raises_exception():
    """Should raise exception when only one argument is passed"""
    expectText="Requires role_arn and role_sess_name be set if either is passed"
    with pytest.raises(Exception, match=expectText):
        session.get_session(role_arn='a')

    with pytest.raises(Exception, match=expectText):
        session.get_session(role_sess_name='a')

def test_get_session__default():
    """Should return default session
    {
        'Credentials': {
            'AccessKeyId': 'string',
            'SecretAccessKey': 'string',
            'SessionToken': 'string',
            'Expiration': datetime(2015, 1, 1)
        },
        'PackedPolicySize': 123,
        'SourceIdentity': 'string'
    }
    """
    temp_session = session.get_session()
    assert isinstance(temp_session, boto3.session.Session), "Returned session from get_session() must be of type boto3.session.Session"
    sts = boto3.client('sts')
    caller_id = sts.get_caller_identity()
    sts_sess = temp_session.client('sts')
    caller_id_sess = sts_sess.get_caller_identity()
    assert caller_id['Account'] == caller_id_sess['Account'], "Returned Account from get_session() must match caller Account"

@pytest.mark.skip(reason="assume role not configured")
def test_get_session__assume_role():
    """Should return default session with assumed role
    {
        'Credentials': {
            'AccessKeyId': 'string',
            'SecretAccessKey': 'string',
            'SessionToken': 'string',
            'Expiration': datetime(2015, 1, 1)
        },
        'AssumedRoleUser': {
            'AssumedRoleId': 'string',
            'Arn': 'string'
        },
        'PackedPolicySize': 123,
        'SourceIdentity': 'string'
    }
    """
    assumed_arn_role = ""
    temp_session = session.get_session(role_arn=assumed_arn_role, role_sess_name="sessionTest")
    assert isinstance(temp_session, boto3.session.Session), "Returned session from get_session() must be of type boto3.session.Session"
    iam = temp_session.client('iam')
    user_list = iam.list_users(MaxItems=1)
    assert len(user_list) > 0, "Response from session.client('aim') must return a list of users, no users found in response."
    assert len(user_list['AssumedRoleUser']['Arn']) > 0, "Response from session.client('aim') must return Arn of AsumedRoleUser, no Arn returned"
    assert user_list['AssumedRoleUser']['Arn'].find(assumed_arn_role) > -1, "Response Arn of AssumedRoleUser from session.client('aim') does not match passed Arn"
