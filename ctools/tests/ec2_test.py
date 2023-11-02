#!/usr/bin/env python3
# Test session code

import os
import sys
import warnings
import pytest
import logging

import ctools.session as session
import ctools.ec2 as ec2

def test_get_instance_tags():
    sess = session.get_session()
    resp = ec2.get_instance_tags(session=sess)
    assert type(resp) == dict, f"Expect response from get_instance_tags() is type dictionary, got {type(resp)}"
    assert len(resp) > 0, "Response from get_instance_tags() must include instance tags, response empty"
    import aws
    client = ec2.get_client(session=sess)
    desc_instance = client.describe_instances(InstanceIds=[aws.instanceId()])
    taglist = desc_instance['Reservations'][0]['Instances'][0]['Tags']
    comp = {d['Key']: d['Value'] for d in taglist}
    assert resp == comp, "Response from get_instance_tags() should contain tags matching the instanceId called"

def test_get_instance_tags_pass_instanceId():
    import aws
    id = aws.instanceId()
    sess = session.get_session()
    resp = ec2.get_instance_tags(instanceId=id, session=sess)
    assert type(resp) == dict, f"Expect response from get_instance_tags() is type dictionary, got {type(resp)}"
    assert len(resp) > 0, "Response from get_instance_tags() must include instance tags, response empty"
    client = ec2.get_client(session=sess)
    desc_instance = client.describe_instances(InstanceIds=[id])
    taglist = desc_instance['Reservations'][0]['Instances'][0]['Tags']
    comp = {d['Key']: d['Value'] for d in taglist}
    assert resp == comp, "Response from get_instance_tags() should contain tags matching the instanceId called"


