from clip_retrieval import clip_inference
import shutil
import os
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel


from pyspark import SparkConf, SparkContext


def create_spark_session():
    # this must be a path that is available on all worker nodes

    os.environ['PYSPARK_PYTHON'] = "/fsx/Andreas/projects/clip-retrieval/distributed/clip_retrieval.pex/__main__.py"
    spark = (
        SparkSession.builder
            .config("spark.submit.deployMode", "client") \
            .config("spark.executorEnv.PEX_ROOT", "./.pex")
            .config("spark.task.resource.gpu.amount", "1")
            .config("spark.executor.resource.gpu.amount", "8")
            # .config("spark.executor.cores", "16")
            # .config("spark.cores.max", "48") # you can reduce this number if you want to use only some cores ; if you're using yarn the option name is different, check spark doc
            .config("spark.driver.port", "7077")
            .config("spark.driver.blockManager.port", "7077")
            .config("spark.driver.host", "172.31.44.42")
            .config("spark.driver.bindAddress", "172.31.44.42")
            .config("spark.executor.memory", "16G")  # make sure to increase this if you're using more cores per executor
            .config("spark.executor.memoryOverhead", "8G")
            .config("spark.task.maxFailures", "100")
            .master("spark://ip-26-0-128-53:7077")  # this should point to your master node, if using the tunnelling version, keep this to localhost
            .appName("spark-stats")
            .getOrCreate()
    )
    return spark


spark = create_spark_session()

clip_inference(input_dataset="pipe:aws s3 cp --quiet s3://stability-aws/laion-a-native/part-0/{00000..18699}.tar -", output_folder="s3://stability-aws/laion-a-native/all-mpnet-base-v2-embeddings",
               input_format="webdataset", enable_metadata=True, write_batch_size=1000000, num_prepro_workers=8, batch_size=512, cache_path=None, distribution_strategy="pyspark", enable_image=False,
               mapper_type='STRANS',output_partition_count=50)