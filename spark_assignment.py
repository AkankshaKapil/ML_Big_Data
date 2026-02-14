


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, length, avg

spark = SparkSession.builder.appName("M25CSA033_Spark_Final").getOrCreate()

# 1. Load into books_df (Matches REQUIRED Schema)
raw_files = spark.sparkContext.wholeTextFiles("/user/hadoop/*.txt")
books_df = raw_files.toDF(["file_name", "text"])

# 2. Extract Metadata for Q10
title_regex = r"(?m)^Title:\s*(.*)"
date_regex = r"Release Date:\s*[A-Za-z]+,?\s*(\d{4})"
lang_regex = r"Language:\s*(\w+)"
enc_regex = r"Character set encoding:\s*([\w-]+)"

metadata_df = books_df.withColumn("title", regexp_extract(col("text"), title_regex, 1)) \
                      .withColumn("release_year", regexp_extract(col("text"), date_regex, 1)) \
                      .withColumn("language", regexp_extract(col("text"), lang_regex, 1)) \
                      .withColumn("encoding", regexp_extract(col("text"), enc_regex, 1))

# 3. Save Q10 Results to CSV
q10_output = metadata_df.select("file_name", "title", "release_year", "language", "encoding")
q10_output.coalesce(1).write.mode("overwrite").option("header", "true").csv("/user/hadoop/q10_metadata_output")

# Show it in terminal so you know it worked
print("--- Q10 RESULTS ---")
q10_output.show()

spark.stop()