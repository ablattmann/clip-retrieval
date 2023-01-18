from pyspark.sql import SparkSession
import os
import sys
from pyspark import SparkContext
from pyspark.sql.functions import rand
from pyspark.sql import SparkSession
import random
import math
import time
import boto3
import fire
import braceexpand


from clip_retrieval import clip_inference


def aws_ec2_s3_spark_session(master, num_cores=96, mem_gb=256):
    """Build a spark session on AWS EC2"""
    # .aws/sparkconfig should be the minimal profile
    print(num_cores)
    os.environ["AWS_CONFIG_FILE"]=os.path.expanduser('~') + "/.aws/sparkconfig"
    session = boto3.session.Session()
    sts_connection = session.client('sts')
    response = sts_connection.assume_role(RoleArn='arn:aws:iam::842865360552:role/s3_access_from_ec2', RoleSessionName='hi',DurationSeconds=12*3600)
    credentials = response['Credentials']
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    main_memory = str(int(mem_gb * 0.75)) + "g"
    memory_overhead = str(mem_gb - int(mem_gb * 0.25)) + "g"
    spark = (
        SparkSession.builder.config("spark.submit.deployMode", "client")
        # .config("spark.driver.cores", "96")
        # .config("spark.driver.memory", "500g")
        # .config("spark.driver.maxResultSize", "100g")
        # .config("spark.task.resource.gpu.amount", "1")
        # .config("spark.task.cpus",num_cores)
        # .config("spark.task.resource.cpu.amount", "12")
        # # .config("spark.worker.resource.gpu.adresses",["0","1","2","3","4","5","6","7"])
        # .config("spark.executor.resource.gpu.amount", "1")
        # .config("spark.executor.cores", num_cores)
        .config("spark.local.dir", "/fsx/Andreas/spark-tmp")
        # .config("spark.executor.memoryOverhead", memory_overhead)
        # .config("spark.driver.port", "4040")
        # .config("spark.driver.blockManager.port", "4041")
        # .config("spark.driver.host", f"{master}")
        # .config("spark.driver.bindAddress", f"{master}")
        .config("spark.executor.memory", main_memory)  # make sure to increase this if you're using more cores per executor
        .config("spark.executor.memoryOverhead", memory_overhead)
        .config("spark.task.maxFailures", "100")
        # com.amazonaws:aws-java-sdk-bundle:1.12.353,
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1,org.apache.spark:spark-hadoop-cloud_2.13:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.12.353")
        # # # change to the appropriate auth method, see https://hadoop.apache.org/docs/stable/hadoop-aws/tools/hadoop-aws/index.html
        .config("spark.hadoop.fs.s3a.access.key", credentials["AccessKeyId"])
        .config("spark.hadoop.fs.s3a.secret.key", credentials["SecretAccessKey"])
        .config("spark.hadoop.fs.s3a.session.token", credentials["SessionToken"])
        # # ton of options to try and make s3a run faster
        .config("spark.hadoop.fs.s3a.threads.max", "512")
        .config("spark.hadoop.fs.s3a.connection.maximum", "2048")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.sql.shuffle.partitions", "40000")
        .config("spark.hadoop.fs.s3a.directory.marker.retention", "keep")
        .config("spark.hadoop.fs.s3a.max.total.tasks", "512")
        .config("spark.hadoop.fs.s3a.multipart.threshold", "5M")
        .config("spark.hadoop.fs.s3a.multipart.size", "5M")
        .config("spark.hadoop.fs.s3a.fast.upload.active.blocks", "512")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "600000")
        .config("spark.hadoop.fs.s3a.readahead.range", "2M")
        .config("spark.hadoop.fs.s3a.socket.recv.buffer", "65536")
        .config("spark.hadoop.fs.s3a.socket.send.buffer", "65536")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.hadoop.fs.s3a.experimental.input.fadvise", "random")
        .config("spark.hadoop.fs.s3a.block.size", "2M")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.fast.buffer.size", "100M")
        .config("spark.hadoop.fs.s3a.fast.upload.buffer", "array")
        .config("spark.hadoop.fs.s3a.bucket.all.committer.magic.enabled", "true")
        .master(f'spark://{master}:7077')  # this should be set to the spark master url
        .appName("sbert-inference")
        .getOrCreate()
    )
    return spark

master_node = 'ip-26-0-143-73'
n_cores = 96
mem_gb = 512



spark = aws_ec2_s3_spark_session(master_node, num_cores=n_cores, mem_gb=mem_gb)


clip_inference(input_dataset="pipe:aws s3 cp --quiet s3://stability-aws/laion-a-native/part-0/{00000..18699}.tar - "
                             "::pipe:aws s3 cp --quiet s3://stability-aws/laion-a-native/part-1/{00000..18699}.tar - "
                             "::pipe:aws s3 cp --quiet s3://stability-aws/laion-a-native/part-2/{00000..18699}.tar - "
                             "::pipe:aws s3 cp --quiet s3://stability-aws/laion-a-native/part-3/{00000..18699}.tar - "
                             "::pipe:aws s3 cp --quiet s3://stability-aws/laion-a-native/part-4/{00000..18699}.tar -",
               output_folder="s3://stability-aws/laion-a-native/sbert_embeddings/",
               input_format="webdataset",
               enable_metadata=True,
               write_batch_size=1000000,
               num_prepro_workers=8,
               batch_size=512,
               cache_path=None,
               mapper_type='STRANS',
               distribution_strategy="pyspark")


