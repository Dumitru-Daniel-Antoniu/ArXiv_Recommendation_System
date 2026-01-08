# =========================
# BERT / DistilBERT arXiv classification with Spark NLP
# Run with spark-submit
# =========================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split
from pyspark.ml import Pipeline

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import (
    Tokenizer,
    DistilBertEmbeddings,
    ClassifierDLApproach
)

# -------------------------------------------------
# 1. Spark Session (CLUSTER MODE)
# -------------------------------------------------
spark = SparkSession.builder \
    .appName("Arxiv_Transformer_SparkNLP") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -------------------------------------------------
# 2. Load arXiv data
# -------------------------------------------------
df = spark.read.json("arxiv-metadata-oai-snapshot.json")

# Keep only abstract + first category
df_ml = df.select("abstract", "categories") \
    .withColumn("label", split(col("categories"), " ").getItem(0)) \
    .select(
        col("abstract").alias("text"),
        "label"
    ) \
    .na.drop()

# -------------------------------------------------
# 3. Limit to TOP categories (CRITICAL)
# -------------------------------------------------
top_labels = df_ml.groupBy("label") \
    .count() \
    .orderBy(col("count").desc()) \
    .limit(5)

df_ml = df_ml.join(top_labels, "label") \
             .select("text", "label")

print("Training samples:", df_ml.count())

# -------------------------------------------------
# 4. Train / Test split
# -------------------------------------------------
train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

# -------------------------------------------------
# 5. Spark NLP Transformer Pipeline
# -------------------------------------------------
document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = DistilBertEmbeddings.pretrained(
    "distilbert_base_uncased", "en"
).setInputCols(["document", "token"]) \
 .setOutputCol("embeddings")

classifier = ClassifierDLApproach() \
    .setInputCols(["embeddings"]) \
    .setOutputCol("prediction") \
    .setLabelColumn("label") \
    .setBatchSize(8) \
    .setMaxEpochs(3) \
    .setEnableOutputLogs(True)

pipeline = Pipeline(stages=[
    document,
    tokenizer,
    embeddings,
    classifier
])

# -------------------------------------------------
# 6. Train model
# -------------------------------------------------
model = pipeline.fit(train)

# -------------------------------------------------
# 7. Predict
# -------------------------------------------------
predictions = model.transform(test)

predictions.select(
    "label",
    col("prediction.result")[0].alias("predicted")
).show(20, truncate=False)

# -------------------------------------------------
# 8. Save model
# -------------------------------------------------
model.write().overwrite().save("arxiv_distilbert_model")

print("Model saved to arxiv_distilbert_model")

spark.stop()
