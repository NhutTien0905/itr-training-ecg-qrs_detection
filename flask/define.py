TEST_TIME = 'run-0'
DATABASE = 'ltdb/'
# DATABASE = 'mitdb/'
PROJECT_DIR = '/home/tien/Documents/ITR/itr-training-ecg-qrs_detection/'

# SAVE_MODEL_DIR = PROJECT_DIR + 'model/' + DATABASE
SAVE_MODEL_DIR = '/home/tien/Documents/ITR/itr-training-ecg-qrs_detection/model/ltdb/model_run-1.h5'

TEMP_DIR = PROJECT_DIR + 'temp2/'
TEMP_DIR_IF = PROJECT_DIR + 'tmp_infer_tfserver/'

TENSOR_BOARD_DIR = PROJECT_DIR + 'log/' + DATABASE

RESULT_DIR = PROJECT_DIR + 'result2/' + DATABASE
RESULT_DIR_IF = PROJECT_DIR + 'r_infer_tfserver/' + DATABASE


LTDB_DIR = '/media/dev7/Data_1/Dataset/ltdb/'
# MITDB_DIR = '/home/tien/Documents/ITR/mit-bih-arrhythmia-database-1.0.0/'
MITDB_DIR = '/home/tien/Documents/ITR/mit-bih-arrhythmia-database-1.0.0/'
AHADB_DIR = '/media/dev7/Data_1/Dataset/ahadb/'
ESCDB_DIR = '/media/dev7/Data_1/Dataset/escdb/'
NSTDB_DIR = '/media/dev7/Data_1/Dataset/nstdb/'

PREPROCESSED_DATA_DIR = '/media/tien/7E90428C90424ABB2/ITR/preprocessed/' + DATABASE
# CHECK_POINT_DIR = '/home/tien/Documents/ITR/checkpoint/' + DATABASE
CHECK_POINT_DIR = '/home/tien/Documents/ITR/checkpoint-1/'

FREQUENCY_SAMPLING = 360
NEIGHBOUR_POINT = int(FREQUENCY_SAMPLING * 0.1) + int(FREQUENCY_SAMPLING * 0.3) + 1 #  ~145
NEIGHBOUR_PRE = int(FREQUENCY_SAMPLING * 0.1) # ~36
NEIGHBOUR_POST = int(FREQUENCY_SAMPLING * 0.3) + 1 # ~108
POSITIVE_RANGE = int(FREQUENCY_SAMPLING * 0.04) + 1 # ~15

DATA_TYPE = 'float32'
