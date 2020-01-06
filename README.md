# ML_Project
[Spark, Python]

This application use Apache Spark and Keras library
It can be run on aws EC2 Cluster :

With Aws CLI on EMR:
Update Package -> sudo yum update
Using the python3 environment -> sudo sed -i -e '$a\export
PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh
Boto3 install -> python3 -m pip install --user boto3
Execute (ex) -> spark-submit --conf spark.dynamicAllocation.enabled=false --num-executors 5
script_fra-oc-p2.py s3://path_to_dir_whith_features/ leonberger vs
