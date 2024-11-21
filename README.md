# Hive Query Language (HQL)

## Creating a Database

```sql
hive> CREATE DATABASE log_data;

hive> CREATE DATABASE IF NOT EXISTS log_data;

hive> SHOW DATABASES;

hive> USE log_data;
```

## Creating Tables

```sql
hive> CREATE TABLE shakespeare (
    lineno STRING,
    linetext STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t';
```

## Data Types

| Type      | Description                                   | Example                   |
| --------- | --------------------------------------------- | ------------------------- |
| TINYINT   | 8-bit signed integer, from -128 to 127        | 127                       |
| SMALLINT  | 16-bit signed integer, from -32,768 to 32,767 | 32,767                    |
| INT       | 32-bit signed integer                         | 2,147,483,647             |
| BIGINT    | 64-bit signed integer                         | 9,223,372,036,854,775,807 |
| FLOAT     | 32-bit single-precision float                 | 1.99                      |
| DOUBLE    | 64-bit double-precision float                 | 3.14159265359             |
| BOOLEAN   | True/false                                    | true                      |
| STRING    | 2 GB max character string                     | hello world               |
| TIMESTAMP | Nanosecond precision                          | 1400561325                |

## Loading Data

```sql
hive> LOAD DATA INPATH 'statistics/log-data/apache.log' OVERWRITE INTO TABLE apache_log;
```

## Grouping

```sql
hive> SELECT
    month,
    count(1) AS count
FROM (SELECT split(time, '/')[1] AS month FROM apache_log) l
GROUP BY month
ORDER BY count DESC;

hive> CREATE TABLE
remote_hits_by_month
AS
SELECT
    month,
    count(1) AS count
FROM (
    SELECT split(time, '/')[1] AS month
    FROM apache_log
    WHERE host == 'remote'
) l
GROUP BY month
ORDER BY count DESC;
```

## Aggregation and Joins

```sql
hive> SELECT
    a.description,
    AVG(f.depart_delay)
FROM airlines a
JOIN flights f ON a.code = f.airline_code
GROUP BY a.description;
```

# HBase

## Creating a Table

```sql
hbase> create 'linkshare', 'link'

hbase> describe 'linkshare'
```

## Adding Column-Family

```sql
hbase> disable 'linkshare'

hbase> alter 'linkshare', 'statistics'

hbase> enable 'linkshare'
```

## Inserting Rows

```sql
hbase> put 'linkshare', 'org.hbase.www', 'link:title', 'Apache HBase'
```

## Incrementing a Column

```sql
hbase> incr 'linkshare', 'org.hbase.www', 'statistics:share', 1

hbase> get_counter 'linkshare', 'org.hbase.www', 'statistics:share', 'dummy'
COUNTER VALUE = 1
```

## Getting Row/Cell Values

```sql
hbase> get 'linkshare', 'org.hbase.www'

hbase> get 'linkshare', 'org.hbase.www', ['link:title', 'statistics:share']

hbase> get 'linkshare', 'org.hbase.www', {COLUMN => 'statistics:share',
VERSIONS => 2}

hbase> get 'linkshare', 'org.hbase.www', {TIMERANGE => [1399887705673,
1400133976734]}
```

## Searching Rows

```sql
hbase> scan 'linkshare'

hbase> scan 'linkshare', {COLUMNS => ['link:title'], STARTROW => 'org.hbase.www'}
```

## Filters

```sql
hbase> import org.apache.hadoop.hbase.util.Bytes
hbase> import org.apache.hadoop.hbase.filter.SingleColumnValueFilter
hbase> import org.apache.hadoop.hbase.filter.BinaryComparator
hbase> import org.apache.hadoop.hbase.filter.CompareFilter

hbase> likeFilter = SingleColumnValueFilter.new(Bytes.toBytes('statistics'),
    Bytes.toBytes('like'),
    CompareFilter::CompareOp.valueOf('GREATER_OR_EQUAL'),
    BinaryComparator.new(Bytes.toBytes(10)))

hbase> scan 'linkshare', { FILTER => likeFilter }
```

# Scoop

## Importing from MySQL to HDFS

```bash
/srv/sqoop$ sqoop import --connect jdbc:mysql://localhost:3306/energydata
    --username root --table average_price_by_state -m 1
```

## Importing from MySQL to Hive

```bash
/srv/sqoop$ sqoop import --connect jdbc:mysql://localhost:3306/energydata
    --username root --table average_price_by_state
    --hive-import --fields-terminated-by ','
    --lines-terminated-by '\n' --null-string 'null' -m 1
```

## Importing from MySQL to HBase

```bash
sqoop import --connect jdbc:mysql://localhost:3306/logdata
    --table weblogs --hbase-table weblogs --column-family traffic
    --hbase-row-key ipyear --hbase-create-table -m 1
```

# Pig Latin

## Relations and tuples

```sql
tweets = LOAD 'united_airlines_tweets.tsv' USING PigStorage('\t')
    AS (id_str:chararray, tweet_url:chararray, created_at:chararray,
    text:chararray, lang:chararray, retweet_count:int, favorite_count:int,
    screen_name:chararray);

dictionary = LOAD 'dictionary.tsv' USING PigStorage('\t') AS (word:chararray,
    score:int);
```

## Filtering

```sql
english_tweets = FILTER tweets BY lang == 'en';
```

## Projection

```sql
tokenized = FOREACH english_tweets GENERATE id_str,
    FLATTEN( TOKENIZE(text) ) AS word;
    clean_tokens = FOREACH tokenized GENERATE id_str,
    LOWER(REGEX_EXTRACT(word, '[#@]{0,1}(.*)', 1)) AS word;

grunt> ILLUSTRATE clean_tokens;
```

## Grouping and Joining

```sql
token_sentiment = JOIN clean_tokens BY word, dictionary BY word;

sentiment_group = GROUP token_sentiment BY id_str;

sentiment_score = FOREACH sentiment_group GENERATE group AS id,
    SUM(token_sentiment.score) AS final;

classified = FOREACH sentiment_score GENERATE id,
    ( (final >= 0)? 'POSITIVE' : 'NEGATIVE' )
    AS classification, final AS score;

final = ORDER classified BY score DESC;
```

## Storing and outputting data

```sql
STORE final INTO 'sentiment_analysis';

grunt> DUMP sentiment_analysis;
```

## Data Types

| Category | Type      | Description                   | Example     |
| -------- | --------- | ----------------------------- | ----------- |
| Numeric  | int       | 32-bit signed integer         | 12          |
|          | long      | 64-bit signed integer         | 34L         |
|          | float     | 32-bit floating-point number  | 2.18F       |
|          | double    | 64-bit floating-point number  | 3e-17       |
| Text     | chararray | String or array of characters | hello world |
| Binary   | bytearray | Blob or array of bytes        | N/A         |

## Relational Operators

| Category             | Operator         | Description                                                        |
| -------------------- | ---------------- | ------------------------------------------------------------------ |
| Loading and storing  | LOAD             | Loads data from the file system or other storage source            |
|                      | STORE            | Saves a relation to the file system or other storage               |
|                      | DUMP             | Prints a relation to the console                                   |
| Filtering and        | FILTER           | Selects tuples from a relation based on some condition             |
| projection           | DISTINCT         | Removes duplicate tuples in a relation                             |
|                      | FOREACH…GENERATE | Generates data transformations based on columns of data            |
|                      | MAPREDUCE        | Executes native MapReduce jobs inside a Pig script                 |
|                      | STREAM           | Sends data to an external script or program                        |
|                      | SAMPLE           | Selects a random sample of data based on the specified sample size |
| Grouping and joining | JOIN             | Joins two or more relations                                        |
|                      | COGROUP          | Groups the data from two or more relations                         |
|                      | GROUP            | Groups the data in a single relation                               |
|                      | CROSS            | Creates the cross-product of two or more relations                 |
| Sorting              | ORDER            | Sorts the relation by one or more fields                           |
|                      | LIMIT            | Limits the number of tuples returned from a relation               |
| Combining and        | UNION            | Computes the union of two or more relations                        |
| splitting            | SPLIT            | Partitions a relation into two or more relations                   |

## User-Defined Functions

```java
package com.statistics.pig;

import java.io.IOException;
import org.apache.pig.EvalFunc;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.Tuple;

public class Classify extends EvalFunc {
    @Override
    public String exec(Tuple input) throws IOException {
        if (args == null || args.size() == 0) {
            return false;
        }
        try {
            Object object = args.get(0);
            if (object == null) {
                return false;
            }
            int i = (Integer) object;
            if (i >= 0) {
                return new String(“POSITIVE”);
            } else {
                return new String(“NEGATIVE”);
            }
        } catch (ExecException e) {
            throw new IOException(e);
        }
    }
}
```

```sql
grunt> REGISTER statistics-pig.jar;

grunt> classified = FOREACH sentiment_score GENERATE id,
com.statistics.pig.Classify(final) AS classification, final AS score;
```

# Spark’s Higher-Level APIs

# Spark SQL

```python
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

parking = sqlContext.read.json('../data/sf_parking/sf_parking_clean.json')

aggr_by_type = sqlContext.sql("""
    SELECT primetype, secondtype,
        count(1) AS count,
        round(avg(regcap), 0) AS avg_spaces
    FROM parking
    GROUP BY primetype, secondtype
    HAVING trim(primetype) != ''
    ORDER BY count DESC""")
```

## DataFrame Operations

```python
from pyspark.sql import functions as F
aggr_by_type = parking.select("primetype", "secondtype", "regcap")
    .where("trim(primetype) != ''")
    .groupBy("primetype", "secondtype")
    .agg(
        F.count("*").alias("count"),
        F.round(F.avg("regcap"), 0).alias("avg_spaces")
    )
    .sort("count", ascending=False)
```

## Data wrangling DataFrames

```python
parking = parking.withColumnRenamed('regcap', 'regcap_old')
parking = parking.withColumn('regcap', parking['regcap_old'].cast('int'))
parking = parking.drop('regcap_old')
```

# Scalable Machine Learning with Spark

## Collaborative Filtering

`Let’s use MLlib’s ALS algorithm to generate recommendations or potential
matches for an online dating service. We’ll generate the recommendations for a
given user based on a dataset consisting of profile ratings from an existing dating
site.`

```py
import sys
import random
from math import sqrt
from operator import add
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

# Configure Spark
conf = SparkConf().setMaster("local") \
    .setAppName("Dating Recommender") \
    .set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)

# Parse ratings
def parse_rating(line, sep=','):
    """
    Parses a rating line
    Returns: tuple of (random integer, (user_id, profile_id, rating))
    """
    fields = line.strip().split(sep)
    user_id = int(fields[0])  # convert user_id to int
    profile_id = int(fields[1])  # convert profile_id to int
    rating = float(fields[2])  # convert rating to float
    return random.randint(1, 10), (user_id, profile_id, rating)

# Parse users
def parse_user(line, sep=','):
    """
    Parses a user line
    Returns: tuple of (user_id, gender)
    """
    fields = line.strip().split(sep)
    user_id = int(fields[0])  # convert user_id to int
    gender = fields[1]
    return user_id, gender

# Compute RMSE
def compute_rmse(model, data, n):
    """
    Compute Root Mean Squared Error (RMSE), or square root of the average value
    of (actual rating - predicted rating)^2
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictions_ratings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
        .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
        .values()
    return sqrt(predictions_ratings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

# Main script
if __name__ == "__main__":
    # Read user ID and gender preference
    matchseeker = int(sys.argv[1])
    gender_filter = sys.argv[2]

    # Create ratings RDD
    ratings = sc.textFile("/home/hadoop/hadoop-fundamentals/data/dating/ratings.dat") \
        .map(parse_rating)

    # Create users RDD
    users = dict(sc.textFile("/home/hadoop/hadoop-fundamentals/data/dating/gender.dat") \
        .map(parse_user).collect())

    # Create training (60%) and validation (40%) sets
    num_partitions = 4
    training = ratings.filter(lambda x: x[0] < 6) \
        .values() \
        .repartition(num_partitions) \
        .cache()
    validation = ratings.filter(lambda x: x[0] >= 6) \
        .values() \
        .repartition(num_partitions) \
        .cache()

    num_training = training.count()
    num_validation = validation.count()
    print(f"Training: {num_training} and validation: {num_validation}\n")

    # ALS training parameters
    rank = 8
    num_iterations = 8
    lambda_ = 0.1

    # Train ALS model
    model = ALS.train(training, rank, num_iterations, lambda_)
    print(f"The model was trained with rank = {rank}, lambda = {lambda_:.1f}, and {num_iterations} iterations.\n")

    # Evaluate RMSE
    validation_rmse = compute_rmse(model, validation, num_validation)
    print(f"The model was trained with rank={rank}, lambda={lambda_:.1f}, and {num_iterations} iterations.")
    print(f"Its RMSE on the validation set is {validation_rmse:.6f}.\n")

    # Generate recommendations
    partners = sc.parallelize([u[0] for u in filter(lambda u: u[1] == gender_filter, users.items())])
    predictions = model.predictAll(partners.map(lambda x: (matchseeker, x))) \
        .collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:10]

    # Print recommendations
    print(f"Eligible partners recommended for User ID: {matchseeker}")
    for i in range(len(recommendations)):
        print(f"{i + 1:2d}: {recommendations[i][1]}")

    # Clean up
    sc.stop()
```

## Classification

`In this example, we’ll build a simple spam classifier that we’ll train with email
data that we’ve categorized as spam and not spam (or ham). Our spam classifier
will utilize two MLlib algorithms, HashingTF, which we’ll use to extract the
feature vectors as term frequency vectors from the training text, and
LogisticRegressionWithSGD, which implements a logistic regression using
stochastic gradient descent`

```python
import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint

# Configure Spark
conf = SparkConf().setMaster("local") \
    .setAppName("Spam Classifier") \
    .set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)

# Read command-line arguments for spam and ham file paths
spam_file = sys.argv[1]
ham_file = sys.argv[2]

# Load the spam and ham data as RDDs
spam = sc.textFile(spam_file)
ham = sc.textFile(ham_file)

# Instantiate HashingTF to extract term frequency features (10,000 features)
tf = HashingTF(numFeatures=10000)

# Transform the spam and ham data into feature vectors
spam_features = spam.map(lambda email: tf.transform(email.split(" ")))
ham_features = ham.map(lambda email: tf.transform(email.split(" ")))

# Convert feature vectors into LabeledPoint (1 for spam, 0 for ham)
positive_examples = spam_features.map(lambda features: LabeledPoint(1, features))
negative_examples = ham_features.map(lambda features: LabeledPoint(0, features))

# Combine spam and ham data for training
training = positive_examples.union(negative_examples)

# Cache the training RDD as logistic regression is iterative
training.cache()

# Train a logistic regression model using stochastic gradient descent (SGD)
model = LogisticRegressionWithSGD.train(training)

# Create test data
positive_test = tf.transform("Guaranteed to Lose 20 lbs in 10 days Try FREE!".split(" "))
negative_test = tf.transform("Hi, Mom, I'm learning all about Hadoop and Spark!".split(" "))

# Predict for the test data
print(f"Prediction for positive test example: {model.predict(positive_test)}")
print(f"Prediction for negative test example: {model.predict(negative_test)}")

# Stop the SparkContext
sc.stop()
```

## Clustering

`In this example, we’ll apply the k-means clustering algorithm to determine
which areas in the United States. have been most hit by earthquakes so far this
year. This information can be found within the GitHub repo’s /data directory, as
earthquakes.csv.`

```python
import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans
from math import sqrt

# Configure Spark
conf = SparkConf().setMaster("local") \
    .setAppName("Earthquake Clustering") \
    .set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)

# Create training RDD from the earthquakes file
earthquakes_file = sys.argv[1]
training = sc.textFile(earthquakes_file).map(parse_vector)

# Set k-clusters (number of clusters)
k = int(sys.argv[2])

# Train the KMeans model based on the training data and k-clusters
model = KMeans.train(training, k)

# Output the cluster centers
print("Earthquake cluster centers: " + str(model.clusterCenters))

# Define a function to calculate the error (Within Set Sum of Squared Errors)
def error(point):
    center = model.centers[model.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

# Calculate the WSSSE (Within Set Sum of Squared Errors)
WSSSE = training.map(lambda point: error(point)).reduce(lambda x, y: x + y)

print("Within Set Sum of Squared Error = " + str(WSSSE))

# Stop the SparkContext
sc.stop()
```
