import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
from make_data import *
from util import *
import tqdm
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from ec57_test import ec57_eval
from multiprocessing import Pool

physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# class QRSModel(tf.keras.Model):
#     def __init__(self, input_shape=256, learning_rate=0.005, momentum=0.9):
#         super(QRSModel, self).__init__()
#         self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu', input_shape=(input_shape, 1), data_format="channels_last")
#         self.dropout1 = tf.keras.layers.Dropout(0.5)
#         self.maxpool1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')
#         self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='valid', activation='relu')
#         self.dropout2 = tf.keras.layers.Dropout(0.5)
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
#         self.dropout3 = tf.keras.layers.Dropout(0.5)
#         self.dense2 = tf.keras.layers.Dense(512, activation='relu')
#         self.dropout4 = tf.keras.layers.Dropout(0.5)
#         self.dense3 = tf.keras.layers.Dense(2, activation='softmax')

#         self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
#         self.loss_fn = tf.keras.losses.BinaryCrossentropy()
#         self.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.dropout1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.dropout2(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dropout3(x)
#         x = self.dense2(x)
#         x = self.dropout4(x)
#         return self.dense3(x)

# def get_qrs_model(input_shape=256, learning_rate=0.005, momentum=0.9):
#     return QRSModel(input_shape=input_shape, learning_rate=learning_rate, momentum=momentum)


# conv1d
def get_qrs_model(input_shape=NEIGHBOUR_POINT, learning_rate=0.005, momentum=0.9):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu',
                                         input_shape=(input_shape, 1), data_format="channels_last", ))
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'))
    cnn_model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='valid', activation='relu'))
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(1024, activation='relu'))
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.Dense(512, activation='relu'))
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # cnn_model.summary()
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=momentum)
    loss = tf.keras.losses.binary_crossentropy
    cnn_model.compile(optimizer, loss=loss, metrics=['accuracy'])

    return cnn_model

# conv2d
# def get_qrs_model(input_shape=(145, 1, 1), learning_rate=0.005, momentum=0.9):
#     cnn_model = tf.keras.models.Sequential()
#     cnn_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), padding='valid', activation='relu',
#                                          input_shape=input_shape))
#     cnn_model.add(tf.keras.layers.Dropout(0.5))
#     cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same'))
#     cnn_model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 1), padding='valid', activation='relu'))
#     cnn_model.add(tf.keras.layers.Dropout(0.5))
#     cnn_model.add(tf.keras.layers.Flatten())
#     cnn_model.add(tf.keras.layers.Dense(1024, activation='relu'))
#     cnn_model.add(tf.keras.layers.Dropout(0.5))
#     cnn_model.add(tf.keras.layers.Dense(512, activation='relu'))
#     cnn_model.add(tf.keras.layers.Dropout(0.5))
#     cnn_model.add(tf.keras.layers.Dense(2, activation='softmax'))

#     optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=momentum)
#     loss = tf.keras.losses.binary_crossentropy
#     cnn_model.compile(optimizer, loss=loss, metrics=['accuracy'])

#     return cnn_model

# class SliceLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(SliceLayer, self).__init__(**kwargs)

#     def call(self, inputs):
#         # Get the shape of the tensor from the inputs
#         shape = tf.shape(inputs)

#         # Define the start index and size for the slice
#         # This example slices the last element along the time dimension
#         start_index = [0, shape[1] - 1, 0]
#         size = [shape[0], shape[1] - 1, shape[2]]  # Correcting size to keep one time step

#         # Apply tf.slice to the inputs
#         sliced = tf.slice(inputs, start_index, size)

#         # Ensure the output has the shape (batch_size, 1, features)
#         return sliced
    
# def custom_padding(x, padding_size):
#     return tf.pad(x, [[0, 0], [padding_size, padding_size], [0, 0]])

# def get_qrs_model(input_shape=(145, 1), learning_rate=0.005, momentum=0.9):
#     # Define the input layer
#     inputs = tf.keras.Input(shape=input_shape)
#     x = tf.keras.layers.Lambda(custom_padding, arguments={'padding_size': 1})(inputs)

#     # First Conv1D layer
#     x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', data_format="channels_last")(x)
#     x = tf.keras.layers.Dropout(0.5)(x)

#     # MaxPooling1D layer
#     x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)

#     # Apply the custom slicing layer
#     x = SliceLayer()(x)

#     # Second Conv1D layer with kernel size 1
#     x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.5)(x)

#     # Flatten layer
#     x = tf.keras.layers.Flatten()(x)

#     # Dense and Dropout layers
#     x = tf.keras.layers.Dense(1024, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     x = tf.keras.layers.Dense(512, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.5)(x)

    
#     # Output layer
#     outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
#     # Create the model
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
# #     model.summary()
    
#     # Compile the model
#     optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
#     loss = tf.keras.losses.binary_crossentropy
#     model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
#     return model


def train_model(model, batch_size=128, epoch=1):
    if os.path.exists(SAVE_MODEL_DIR + TEST_TIME):
        print("Model %s was trained and are ready" % TEST_TIME)
        return

    shuffle_buffer = batch_size * 100
    prefetch_buffer = batch_size * 100
    # Calculate total sample in input dataset
    train_set = get_record_preprocessed('train')
    sample = 0
    for file in train_set:
        if file.split('.')[1] == '2':
            continue
        header = wfdb.rdheader(MITDB_DIR + file.split('.')[0])
        sample += header.sig_len - (NEIGHBOUR_POINT - 1) * 2

    train_data = get_tf_records(get_record_preprocessed('train'), batch_size, shuffle_buffer, prefetch_buffer)
    valid_data = get_tf_records(get_record_preprocessed('valid'), batch_size, shuffle_buffer, prefetch_buffer,
                                mode='valid')

    log_dir = TENSOR_BOARD_DIR + TEST_TIME
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    check_point_dir = CHECK_POINT_DIR + TEST_TIME + '/'
    if not os.path.exists(check_point_dir):
        os.mkdir(check_point_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_dir + "{epoch:02d}.weights.h5",
                                                    save_weights_only=True, verbose=0, save_freq='epoch')
    callback = [
        tensorboard,
        checkpoint
    ]
    with tf.device("/GPU:0"):
        model.fit(train_data, steps_per_epoch=int(sample / batch_size), epochs=epoch, verbose=1, 
                    # use_multiprocessing=True,
                    validation_data=valid_data,
                    callbacks=callback,
                    )
        model.save(SAVE_MODEL_DIR + TEST_TIME + '/model_run-0.h5')


def get_result(result_file_name, checkpoint=True, checkpoint_epoch=0, saved_model_name=None, batch_size=128):
    with open(RESULT_DIR + result_file_name + '.csv', 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(["Recording", "TP", "FN", 'FP', 'Se', 'P+'])
        total = np.zeros(5)
        shuffle_buffer = batch_size * 100
        prefetch_buffer = batch_size * 100
        if checkpoint:
            model = get_qrs_model()
            model.load_weights(CHECK_POINT_DIR + TEST_TIME + "/0{}.weights.h5".format(checkpoint_epoch))
            model.save(SAVE_MODEL_DIR + TEST_TIME + '/model_run-0.h5')
            print('Load model checkpoint')
        else:
            model = tf.keras.models.load_model(SAVE_MODEL_DIR + saved_model_name)

        num_file = len(get_record_preprocessed('test'))
        for file in get_record_preprocessed('test'):
            if file.split('.')[0] in ['104', '102', '107', '217']:
                continue
            test_data = get_tf_records(file, batch_size, shuffle_buffer, prefetch_buffer, mode='test')
            with tf.device("/GPU:0"):
                prediction = model.predict(test_data,
                                        #    use_multiprocessing=True,
                                           verbose=0)
            prediction = np.rint(prediction)
            # np.savetxt(TEMP_DIR + 'predict' + file + '.txt', prediction.astype('int'), fmt='%d')
            result = evaluate(file.split('.')[0], prediction, MITDB_DIR)
            print(file, result)
            total = total + result
            writer.writerow([file, result[0], result[1], result[2], result[3], result[4]])
        print(['total', total[0], total[1], total[2], total[3] / num_file, total[4] / num_file])
        writer.writerow(['total', total[0], total[1], total[2], total[3] / num_file, total[4] / num_file])
        ec57_eval(RESULT_DIR + 'ec57/', TEMP_DIR, 'atr', 'atr', 'pred', None)


def multi_predict(file_path, dataset, checkpoint=True, checkpoint_epoch=3, saved_model_name=None, batch_size=128):
    if checkpoint:
        model = get_qrs_model()
        model.load_weights(CHECK_POINT_DIR + TEST_TIME + "/0{}.weights.h5".format(checkpoint_epoch))
        print('Load model checkpoint')
    else:
        model = tf.keras.models.load_model(SAVE_MODEL_DIR + saved_model_name)
    test_data, _ = preprocess_data(file_path)
    with tf.device("/GPU:0"):
        prediction = model.predict(test_data, batch_size=batch_size, use_multiprocessing=True, verbose=0)
    prediction = np.rint(prediction)
    np.savetxt(TEMP_DIR + 'pred.txt', prediction, fmt='%d\t')
    evaluate(file_path.split('/')[-1][:-4], prediction, dataset, True)


def get_result_ec57():

    database = [
        MITDB_DIR,
        # AHADB_DIR,
        # ESCDB_DIR,
        # NSTDB_DIR
    ]

    for dataset in database:
        ann_dir = TEMP_DIR + dataset.split('/')[-2] + '/'
        if not os.path.isdir(ann_dir):
            os.makedirs(ann_dir)
        arg_list = []
        for file in get_record_raw(dataset):
            if file.split('/')[-1][:-4] in ['104', '102', '107', '217', 'bw', 'em', 'ma']:
                continue
            arg_list.append([file, dataset, True, 3])
            # test_data, _ = preprocess_data(file)
            # with tf.device("/GPU:0"):
            #     prediction = model.predict(test_data, batch_size=batch_size, use_multiprocessing=True, verbose=0)
            # prediction = np.rint(prediction)
            # evaluate(file.split('/')[-1][:-4], prediction, dataset, True)

        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap(multi_predict, arg_list)

        result_dir = RESULT_DIR + 'ec57/' + dataset.split('/')[-2] + '/'
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        ec57_eval(result_dir, ann_dir, 'atr', 'atr', 'pred', None)


if __name__ == '__main__':
    # generate_data(get_record_raw(MITDB_DIR), None)
    train_model(get_qrs_model(), epoch=1)
    get_result(TEST_TIME, checkpoint=True, checkpoint_epoch=1, saved_model_name='run-0')
    # get_result_ec57()
