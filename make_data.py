from preprocessing import *
import multiprocessing
import tensorflow as tf


def get_record_raw(dataset):
    file = []
    for root, _, files in os.walk(dataset):
        for f in files:
            if '.hea' in f:
                file.append(os.path.join(root, f))
    return file


def get_record_preprocessed(mode):
    return sorted(os.listdir(PREPROCESSED_DATA_DIR + mode))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_tf_record(file_path, separate=None):
    file_name = file_path.split('/')[-1][:-4] if separate is None \
        else file_path.split('/')[-1][:-4] + '.' + str(int(separate))
    # print('Generating {}.tfrecord'.format(file_name))
    if os.path.exists(PREPROCESSED_DATA_DIR + file_name + '.tfrecord'):
        print('File ' + file_name + '.tfrecord is already.')
        return

    data, label = preprocess_data(file_path, separate)
    label = tf.keras.utils.to_categorical(label, 2).astype('int8')
    print('filename:', file_name, 'data_shape=', data.shape, 'label_shape=', label.shape)

    with tf.io.TFRecordWriter(PREPROCESSED_DATA_DIR + file_name + '.tfrecord') as writer:
        for index in range(data.shape[0]):
            feature = {'image': _float_feature(data[index].flatten().tolist()),
                       'label': _int64_feature(label[index].flatten().tolist())}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def generate_data(data_set, separate=None):
    print('Generating data...')
    if not os.path.exists(PREPROCESSED_DATA_DIR):
        os.mkdir(PREPROCESSED_DATA_DIR)
    processes1 = []
    for file_path in data_set:
        p1 = multiprocessing.Process(target=save_tf_record, args=(file_path, separate))
        processes1.append(p1)
        p1.start()
    for process in processes1:
        process.join()
    print('Finish generating data...')
    if separate:
        print('Generating data part 2...')
        processes2 = []
        for file_path in data_set:
            p2 = multiprocessing.Process(target=save_tf_record, args=(file_path, separate + 1))
            processes2.append(p2)
            p2.start()
        for process in processes2:
            process.join()
        print('Finish generating data part 2...')


def parse_batch(record_batch):
    # Create a description of the features
    feature_description = {
        'image': tf.io.FixedLenFeature([NEIGHBOUR_POINT, 1], tf.float32),
        'label': tf.io.FixedLenFeature([2], tf.int64),
    }
    example = tf.io.parse_example(record_batch, feature_description)
    example['label'] = tf.cast(example['label'], tf.int8)

    return example['image'], example['label']


def get_tf_records(data_set, batch_size, shuffle_buffer, prefetch_buffer, mode='train'):
    """mode = ['train', 'valid', 'test]"""
    # files = tf.data.Dataset.list_files(files)
    # ds = files.interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(100)
    #                       , num_parallel_calls=os.cpu_count())
    # ds = tf.data.TFRecordDataset(PREPROCESSED_TEST_DATA_RECORD_DIR + filename + '.tfrecord',
    #                              num_parallel_reads=os.cpu_count())
    if type(data_set) != list:
        data_set = [data_set]
    files = [PREPROCESSED_DATA_DIR + mode + "/{}".format(i) for i in data_set]

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=os.cpu_count())
    ds = ds.map(parse_batch, num_parallel_calls=os.cpu_count())
    if mode == 'train':
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch_buffer)

    return ds
