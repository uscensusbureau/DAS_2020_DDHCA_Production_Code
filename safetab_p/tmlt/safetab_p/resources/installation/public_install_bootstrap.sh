#! /bin/bash -xe
sudo yum install git -y
sudo python3 -m ensurepip --upgrade
sudo python3 -m pip install numpy==1.21.5 --target='/usr/local/lib/python3.7/dist-packages'
sudo python3 -m pip install boto3==1.26.138
sudo python3 -m pip install botocore==1.29.138
sudo python3 -m pip install colorama==0.4.6
sudo python3 -m pip install exceptiongroup==1.1.1
sudo python3 -m pip install importlib-metadata==5.2.0
sudo python3 -m pip install iniconfig==2.0.0
sudo python3 -m pip install jmespath==1.0.1
sudo python3 -m pip install mpmath==1.3.0
sudo python3 -m pip install nose==1.3.7
sudo python3 -m pip install packaging==23.1
sudo python3 -m pip install pandas==1.3.5
sudo python3 -m pip install parameterized==0.7.5
sudo python3 -m pip install pluggy==1.0.0
sudo python3 -m pip install py4j==0.10.9
sudo python3 -m pip install pyarrow==6.0.1
sudo python3 -m pip install pytest==7.3.1
sudo python3 -m pip install python-dateutil==2.8.2
sudo python3 -m pip install pytz==2023.3
sudo python3 -m pip install randomgen==1.23.1
sudo python3 -m pip install s3transfer==0.6.1
sudo python3 -m pip install scipy==1.6.0
sudo python3 -m pip install six==1.16.0
sudo python3 -m pip install smart-open==5.2.1
sudo python3 -m pip install sympy==1.9
sudo python3 -m pip install tomli==1.2.3
sudo python3 -m pip install typeguard==2.12.1
sudo python3 -m pip install typing-extensions==3.10.0.2
sudo python3 -m pip install urllib3==1.26.16
sudo python3 -m pip install zipp==3.15.0
sudo python3 -m pip install tmlt-core==0.6.0
