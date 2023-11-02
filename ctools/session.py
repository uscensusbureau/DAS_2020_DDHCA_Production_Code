import boto3
from boto3.session import Session

def get_session(role_arn:str=None, role_sess_name:str=None):
    """ returns boto3 session object

        session valid for a maximum of 1 hour, 
        caller responsible for obtaining a new session after the 1 hour timeout
        
        role_arn str, required
        role_sess_name str, required, unique role session name
    """
    if not role_arn and not role_sess_name:
        return Session()
    if not role_arn or not role_sess_name:
        raise Exception("Requires role_arn and role_sess_name be set if either is passed")

    sts = boto3.client("sts")
    response = sts.assume_role(
        RoleArn=f"{role_arn}",
        RoleSessionName=f"{role_sess_name}"
    )
    
    assumed_session = Session(
        aws_access_key_id=response['Credentials']['AccessKeyId'],
        aws_secret_access_key=response['Credentials']['SecretAccessKey'],
        aws_session_token=response['Credentials']['SessionToken'])
        
    return assumed_session