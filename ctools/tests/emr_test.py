#!/usr/bin/env python3
# Test session code

import os
import sys
import warnings
import pytest
import logging

import ctools.session as session
import ctools.emr as emr

def test_add_cluster_info():
    sess = session.get_session()
    creds = sess.get_credentials()
    passed_creds = { 'access_key':creds.access_key, 'secret_key':creds.secret_key, 'token':creds.token }
    clusters = emr.list_clusters(session=sess)
    for cluster in list(clusters):
        if cluster['Status']['State']=='TERMINATED':
            clusters.remove(cluster)
    for cluster in list(clusters):
        resp = emr.add_cluster_info(cluster, creds=passed_creds)
        break
    assert type(resp) == dict, f"Expect response type dictionary, got {type(resp)}"
    assert "MasterInstanceTags" in resp, "MasterInstanceTags key must be in response from add_cluster_info()"
    desc_cluster = emr.describe_cluster(cluster['Id'],session=sess)
    dc_Key = desc_cluster['Tags'][0]['Key']
    dc_Val = desc_cluster['Tags'][0]['Value']
    assert resp['MasterInstanceTags'][dc_Key] == dc_Val, f"Response['MasterInstanceTags'][{dc_Key}] from add_cluster_info() should equal {dc_Val}, but actually equals {resp['MasterInstanceTags'][dc_Key]}"
    # Would need hard coded key to test further responses
    #   ['MasterInstanceTags'][tag['Key']] = tag['Value']

def test_complete_cluster_info():
    sess = session.get_session()
    resp = emr.complete_cluster_info(session=sess)
    assert type(resp) == list, f"Expect response from complete_cluster_info() is type list, got {type(resp)}"
    assert len(resp) > 0, "No clusters returned from complete_cluster_info()"
    assert len(resp[0]['Status']['State']) > 0, "Returned cluster from complete_cluster_info() must include a State"

def test_list_clusters():
    sess = session.get_session()
    resp = emr.list_clusters(session=sess)
    assert type(resp) == list, f"Expect response from list_clusters() is type list, got {type(resp)}"
    resp = emr.list_clusters(state=['WAITING'],session=sess)
    assert type(resp) == list, f"Expect response from list_clusters() is type list, got {type(resp)}"
    assert len(resp[0]['Id']) > 0, "Response from list_clusters() must include an Id"

def test_list_instances():
    sess = session.get_session()
    resp = emr.list_instances(session=sess)
    assert type(resp) == list, f"Expect response from list_instances() is type list, got {type(resp)}"
    assert len(resp[0]['Id']) > 0, "Response from list_instances() must include an Id"

def test_isMaster():
    resp = emr.isMaster()
    assert type(resp) == bool, f"Expect response from isMaster() is type boolean, got {type(resp)}"

def test_isSlave():
    resp = emr.isSlave()
    assert type(resp) == bool, f"Expect response from isSlave() is type boolean, got {type(resp)}"

@pytest.mark.skip(reason="ThrottlingException, use in isolation of first similar test")
def test_complete_cluster_info_terminated():
    # separated and moved in order to avoid timeout
    sess = session.get_session()
    resp = emr.complete_cluster_info(terminated=True,session=sess)
    assert type(resp) == list, f"Expect response from complete_cluster_info() is type list, got {type(resp)}"
    assert len(resp[0]['Id']) > 0, "Response from complete_cluster_info() must include an Id"
    assert resp[0]['Status']['State'] == 'TERMINATED', f"Response State from complete_cluster_info() should be 'TERMINATED', got {resp[0]['Status']['State']}"

def test_describe_cluster():
    sess = session.get_session()
    clusterId = emr.clusterId()
    resp = emr.describe_cluster(clusterId,session=sess)
    assert type(resp) == dict, f"Expect response from describe_cluster() is type dictionary, got {type(resp)}"
    assert len(resp['Id']) > 0, "Response from describe_cluster() must include an Id"
