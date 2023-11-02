import pathlib

import numpy as np
try:
    import tensorflow as tf
except ImportError as err:
    raise ImportError('This require tensorflow.') from err


def load_dataset(dataset_dir: str) -> tf.data.Dataset:
    items_path = pathlib.Path(dataset_dir) / 'items/'
    def load_items():
        for item in items_path.iterdir():
            yield dict(np.load(item))
    def to_spec(x): return tf.TensorSpec(x.shape, x.dtype)
    tf_specs = tf.nest.map_structure(to_spec, next(load_items()))
    ds = tf.data.Dataset.from_generator(load_items, output_signature=tf_specs)
    return ds
