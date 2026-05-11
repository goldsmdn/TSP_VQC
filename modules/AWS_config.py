#AWS config
import boto3
from braket.aws import AwsDevice, AwsSession, LocalSimulator

ANKAA_DEVICE_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
TARGET = 'local'                    # Options 'local', 'ankaa', 'ankaa_sim'

AWS_PROFILE = 'qcap'
AWS_REGION = 'eu-west-2'

# Create a boto3 session with region
BOTO_SESSION = boto3.Session(
    profile_name=AWS_PROFILE,
    region_name=AWS_REGION,
    )

AWS_SESSION = AwsSession(boto_session=BOTO_SESSION)

# modules/config.py
def get_aws_session():
    return AwsSession(
        boto_session=boto3.Session(
            profile_name=AWS_PROFILE,
            region_name=AWS_REGION,
        )
    )

def get_ankaa_device():
    session = get_aws_session()
    return AwsDevice(ANKAA_DEVICE_ARN, aws_session=session)


TARGETS = {'local': {'device': LocalSimulator()},
                   'ankaa': {'device': AwsDevice(ANKAA_DEVICE_ARN, aws_session = AWS_SESSION)},
                   'ankaa_em': {'device': AwsDevice(ANKAA_DEVICE_ARN, aws_session = AWS_SESSION).emulator()},
                   } 

