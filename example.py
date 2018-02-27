import tensorflow as tf
import shutil


# setup feature columns
domain = tf.feature_column.categorical_column_with_hash_bucket(
 "domain", 100000, dtype=tf.string)
hour = tf.feature_column.categorical_column_with_identity("hour", 24)
device_type = tf.feature_column.categorical_column_with_vocabulary_list(
 "device_type", vocabulary_list=["desktop", "mobile", "tablet"],
 default_value=0)
feature_columns = [domain, hour, device_type]


# actual model setup
ftrl = tf.train.FtrlOptimizer(
    learning_rate=0.1,
    learning_rate_power=-0.5,
    l1_regularization_strength=0.001,
    l2_regularization_strength=0.0
    )

model_dir = "."
estimator = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    optimizer=ftrl,
    model_dir=model_dir
    )


def input_fn(paths):
    """ model input function """

    names = ["domain", "hour", "device_type", "is_click"]
    record_defaults = [[""], [0], ["desktop"], [0]]

    def _parse_csv(rows_string_tensor):
        columns = tf.decode_csv(rows_string_tensor, record_defaults)
        features = dict(zip(names, columns[:-1]))
        labels = columns[-1]
        return features, labels

    def _input_fn():
        dataset = tf.data.TextLineDataset(paths)
        dataset = dataset.map(_parse_csv)
        dataset = dataset.batch(100)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return _input_fn


# Train the model.
paths = ["example.csv"]
estimator.train(input_fn=input_fn(paths), steps=None)


# Export our model
columns = [('hour', tf.int64),
           ('domain', tf.string),
           ('device_type', tf.string)]
feature_placeholders = {
 name: tf.placeholder(dtype, [1], name=name + "_placeholder")
 for name, dtype in columns
}
export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
    feature_placeholders)
path = estimator.export_savedmodel(model_dir, export_input_fn)

# rename export directory
shutil.move(path, "EXPORT")
