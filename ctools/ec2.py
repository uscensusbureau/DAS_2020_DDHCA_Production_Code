import json
import subprocess
import boto3
from typing import Dict, List

Ec2InstanceId='Ec2InstanceId'
Status='Status'

def get_client(session=None):
    return boto3.client('ec2') if not session else session.client('ec2')

def create_tag(*, resourceId: str, key_value_pairs: Dict, session=None) -> None:
    if not session:
        cmd = ['aws', 'ec2', 'create-tags', '--resources', resourceId]
        for (key, value) in key_value_pairs:
            cmd += ['--tags', f'Key={key},Value={value}']
        subprocess.check_call(cmd)
    else:
        client = get_client(session=session)
        allTags = []
        for(key, value) in key_value_pairs:
            allTags.append({"Key":f'{key}', "Value":f'{value}'})
        client.create_tag(Resources=[f'{resourceId}'],Tags=allTags)

def delete_tag(*, resourceId: str, tags: List, session=None) -> None:
    if not session:
        cmd = ['aws', 'ec2', 'delete-tags', '--resources', resourceId]
        for tag in tags:
            cmd += ['--tags', f'Key={tag}']
        subprocess.check_call(cmd)
    else:
        client = get_client(session=session)
        allTags = []
        for tag in tags:
            allTags.append({"Key":f'{tag}'}) 
        client.remove_tags(Resources=[resourceId],Tags=allTags)

def describe_tags(*, resourceId, session=None):
    if not session:
        cmd = ['aws', 'ec2', 'describe-tags', '--filters', f'Name=resource-id,Values={resourceId}', '--output', 'json', '--no-paginate']
        response = subprocess.check_output(cmd)
        return json.loads(response)['Tags']
    else:
        client = get_client(session=session)
        filters = [{'Name':'resource-id','Values':[resourceId]}]
        response = client.describe_tags(Filters=filters)
        return response['Tags']

def describe_instances(*, groupid: str = None, session=None):
    if not session:
        cmd = ['aws', 'ec2', 'describe-instances', '--output', 'json']
        if groupid:
            cmd += ['--filters', f'Name=instance.group-id,Values={groupid}']
        response = subprocess.check_output(cmd)
        return sum([reservation['Instances'] for reservation in json.loads(response)['Reservations']], [])
    else:
        client = get_client(session=session)
        if groupid:
            filters = [f'Name=instance.group-id,Values={groupid}']
            response = client.describe_instances(Filters=filters)
        else:
            response = client.describe_instances()
        return sum([reservation['Instances'] for reservation in response['Reservations']], [])

def get_instance_tags(instanceId: str = None, session=None) -> Dict:
    """Return a dictionary of all the tags for a given instance, default this instance."""
    import aws
    if instanceId is None:
        instanceId = aws.instanceId()
    with aws.Proxy(https=True, http=False) as p:
        client = get_client(session=session)
        response = client.describe_instances(InstanceIds=[instanceId])
        taglist = response['Reservations'][0]['Instances'][0]['Tags']
        return {d['Key']: d['Value'] for d in taglist}

if __name__=="__main__":
    print("Instance tags:\n", json.dumps(get_instance_tags(), indent=4))
