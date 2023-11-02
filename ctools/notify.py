"""notify.py: notify using SNS or some other approach"""

import sys
import os
import copy
from subprocess import run, Popen, PIPE
import boto3

def get_client(session=None):
    return boto3.client('sns') if not session else session.client('sns')

def notify(msg: str, session=None) -> None:
    if 'DAS_SNSTOPIC' in os.environ:
        old_env = copy.deepcopy(os.environ)
        if 'BCC_PROXY' in os.environ:
            os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = os.environ['BCC_PROXY']
            if not session:
                ret = Popen(['aws', 'sns', 'publish', '--topic-arn', os.environ['DAS_SNSTOPIC'], '--message', msg], stdout=PIPE).communicate()[0]
            else:
                client = get_client(session=session)
                ret = client.publish(
                    TopicArn=os.environ['DAS_SNSTOPIC'],
                    Message=msg
                )

        for var in ['HTTP_PROXY', 'HTTPS_PROXY']:
            if var not in old_env:
                del os.environ[var]
            else:
                os.environ[var] = old_env[var]


if __name__=="__main__":
    msg = " ".join(sys.argv[1:])
    notify(msg)
