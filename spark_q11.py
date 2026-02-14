


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, regexp_extract, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
import numpy as np

# Strict 1GB memory limits
spark = SparkSession.builder \
    .appName("M25CSA033_Q11_AllBooks") \
    .config("spark.driver.maxResultSize", "500m") \
    .config("spark.memory.fraction", "0.6") \
    .getOrCreate()

# THE TRICK: minPartitions=1000 forces strictly 1 book per task into RAM
raw_files = spark.sparkContext.wholeTextFiles("/user/hadoop/*.txt", minPartitions=1000)
books_df = raw_files.map(lambda x: (x[0].split("/")[-1], x[1])).toDF(["file_name", "text"])

# Standard Preprocessing
extract_pattern = r"(?s)\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*(.*?)\*\*\* END OF THE PROJECT GUTENBERG EBOOK"
clean_df = books_df.withColumn("text_clean", regexp_extract(col("text"), extract_pattern, 1))
clean_df = clean_df.withColumn("text_clean", when(col("text_clean") == "", col("text")).otherwise(col("text_clean")))

clean_df = clean_df.withColumn("text_clean", lower(col("text_clean"))) \
                   .withColumn("text_clean", regexp_replace(col("text_clean"), r'[^a-z\s]', ' '))

# NLP Pipeline
tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")

# THE SECOND TRICK: Limit the math vectors to exactly 1000 features
hashingTF = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=1000) 

# THE THIRD TRICK: minDocFreq=5 drops words that only appear in 1 or 2 books
idf = IDF(inputCol="tf", outputCol="tfidf", minDocFreq=5)

# Fit and Transform
processed_df = idf.fit(hashingTF.transform(remover.transform(tokenizer.transform(clean_df)))) \
                  .transform(hashingTF.transform(remover.transform(tokenizer.transform(clean_df))))

# ESSENTIAL: We MUST drop the raw text columns and cache ONLY the tiny math arrays
final_df = processed_df.select("file_name", "tfidf").cache()

# Extract target (10.txt)
target_row = final_df.filter(col("file_name") == "10.txt").collect()

if not target_row:
    print("Error: 10.txt not found in HDFS!")
    spark.stop()
    exit()

# Broadcast the target vector
bc_target_vec = spark.sparkContext.broadcast(target_row[0]['tfidf'].toArray())

def calculate_similarity(row_vec):
    v1 = bc_target_vec.value
    v2 = row_vec.toArray()
    dot = float(np.dot(v1, v2))
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm != 0 else 0.0

print("\n" + "="*50)
print("CALCULATING COSINE SIMILARITY ACROSS ALL 430+ BOOKS...")
print("="*50)

# Calculate similarity for the entire collection
similarity_rdd = final_df.filter(col("file_name") != "10.txt").rdd.map(lambda row: (
    row['file_name'], 
    calculate_similarity(row['tfidf'])
))

# Get top 5 across the entire dataset!
top_5 = similarity_rdd.takeOrdered(5, key=lambda x: -x[1])

print("\n" + "="*50)
print("TOP 5 SIMILAR BOOKS TO 10.TXT (King James Bible)")
print("="*50)
for name, score in top_5:
    print(f"Book: {name} | Score: {score:.4f}")

spark.stop()