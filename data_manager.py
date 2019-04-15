import tensorflow as tf
import os
import numpy as np


class DataManager(object):
    def __init__(self, flags):
        self.flags = flags
        root = ''
        basepath = flags.dataset
        base = os.path.join(root, basepath)

        self.filenames = []
        self.num_files = 0
        for file in os.listdir(base):
            if file.endswith('tfrecord'):
                self.filenames.append(os.path.join(root, basepath, file))
                self.num_files += 1
        self.num_traj_per_file = 100

    def parser(self, record):
        """
        Instantiates the ops used to read and parse the data into tensors.
        """
        feature_map = {
            'init_x':
                tf.FixedLenFeature(
                    shape=[1],
                    dtype=tf.float32),
            'init_y':
                tf.FixedLenFeature(
                    shape=[1],
                    dtype=tf.float32),
            'init_hd':
                tf.FixedLenFeature(
                    shape=[1],
                    dtype=tf.float32),
            'ego_v':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'theta_x':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'theta_y':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'target_x':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'target_y':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
            'target_hd':
                tf.FixedLenFeature(
                    shape=[self.flags.sequence_length],
                    dtype=tf.float32),
        }
        example = tf.parse_single_example(record, feature_map)
        batch = [
            example['init_x'],
            example['init_y'],
            example['init_hd'],
            example['ego_v'],
            example['theta_x'],
            example['theta_y'],
            example['target_x'],
            example['target_y'],
            example['target_hd']
        ]
        return batch

    def get_batch(self):
        
        #choose from datasets
#         datasets = [
#             tf.data.TFRecordDataset(file).map(self.parser).repeat().batch(self.flags.batch_size)
#                     for file in self.filenames
#         ]
#         choice_idxs = tf.cast(np.repeat(np.arange(10), 1000), tf.int64)
#         choice_dataset = tf.data.Dataset.from_tensor_slices(choice_idxs).repeat()
#         dataset = tf.contrib.data.choose_from_datasets(datasets, choice_dataset)
        

#         # Define dataset
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self.parser)
        num_traj = self.num_files * self.num_traj_per_file
        dataset = dataset.shuffle(buffer_size=num_traj)
        num_epochs = self.flags.steps // self.flags.batch_size
        dataset = dataset.repeat()
        dataset = dataset.batch(self.flags.batch_size)

        iterator = dataset.make_one_shot_iterator()

        batch = iterator.get_next()

        return batch
