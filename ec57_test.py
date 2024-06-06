from glob import glob
import numpy as np
import os
import csv
import subprocess
import wfdb as wf
from os.path import isdir
from define import *

def HR_calcu(fs_origin, path_db, ext_db):
    # Calculate HR value and write .mnn file
    NRRMAX = 10  # default
    RRTop = -1
    nRR = 6
    # SR = 360
    AuxAnns = ['+', 's', 'T', '=', '\'', '@']
    BeatAnns = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    HR_METHOD = 1  # defaul
    RRs = np.zeros(NRRMAX)
    MEAS = 0
    NUM = 0
    CHAN = 0
    SYMBOL = '='
    T = 1 / float(fs_origin)
    T_ms = 1000 / fs_origin

    file_names = glob(path_db + '/*.rd' + ext_db)

    for file in file_names:
        # print('file: ', os.path.basename(file))
        print(os.path.dirname(file) + '/' + os.path.splitext(os.path.basename(file))[0] + '.hr' + ext_db)
        csv_file = open(os.path.dirname(file) + '/' + os.path.splitext(os.path.basename(file))[0] + '.hr' + ext_db, 'w')
        # init = 0
        samples = 0
        oldsamples = 0
        nsamples = 0
        with open(file, 'r') as File:
            if RRTop == -1:
                for i in range(nRR):
                    RRs[i] = 0
                RRTop += 1
            csv_reader = csv.reader(File, delimiter=',')
            for line in File:
                # print(line)
                output = [s.strip() for s in line.split(' ') if s]
                if len(output) > 6:
                    time = output[0] + ']'
                    BeatSample = int(output[2])
                    BeatStatus = output[3]

                else:
                    try:
                        time = output[0]
                        BeatSample = int(output[1])
                        BeatStatus = output[2]
                    except Exception as e:  # pylint: disable=broad-except
                        print(e)
                        pass
                HR = 0
                has_Aux = 0
                is_BeatAnn = 0
                for symbol in range(len(AuxAnns)):
                    if BeatStatus == AuxAnns[symbol]:
                        has_Aux = 1
                        break
                if has_Aux == 1:
                    k = len(output) + 1
                else:
                    k = len(output)
                if (k != 6) and (k != 7) and (k != 8):
                    break
                for beat in range(len(BeatAnns)):
                    if BeatStatus == BeatAnns[beat]:
                        is_BeatAnn = 1
                        break
                if is_BeatAnn == 1:
                    samples = BeatSample
                    nsamples += 1

                    if (nsamples == 1):
                        oldsamples = BeatSample

                    if HR_METHOD == 0:
                        RR_sample = samples - oldsamples
                        RR = RR_sample * T
                        HR = 60 / RR
                        oldsamples = samples

                    if HR_METHOD == 1:
                        RR_sample = samples - oldsamples
                        oldsamples = samples
                        RRs[RRTop] = RR_sample
                        RRTop += 1
                        if (RRTop >= nRR):
                            RRTop = 0

                    if nsamples >= 7:
                        # init = 1
                        RRmax = 0
                        RRmin = 1000000
                        RRsum = 0
                        for i in range(nRR):
                            RRsum += RRs[i]
                            if (RRs[i] > RRmax):
                                RRmax = RRs[i]
                            if (RRs[i] < RRmin):
                                RRmin = RRs[i]
                        RR = (RRsum - RRmax - RRmin) * T / 4
                        HR = round((60 / RR), 2)
                        try:
                            csv_file.writelines('  ' + str(time) + '{:>9}'.format(
                                str(BeatSample)) + '     ' + SYMBOL + '    ' + str(MEAS) + '    ' + str(
                                NUM) + '    ' + str(CHAN) + ' \t' + str(HR) + '\n')
                        except Exception as e:  # pylint: disable=broad-except
                            print(e)
                        pass

        csv_file.close()


def rm_file(bashlink, dir, dir_report, del_res='0'):
    """

    :param bashlink:
    :param dir:
    :param dir_report:
    :return:
    """
    subprocess.call(bashlink + ' ' +
                    dir + '/ ' +            #$1
                    dir_report + '/ ' +     #$2
                    del_res,                #$3
                    shell=True)


def rann_script(bashlink, dir, ext_atr, ext_ai):
    subprocess.call(bashlink + ' ' +
                    dir + '/ ' +        # $1
                    ext_atr + ' ' +     # $2
                    ext_ai + ' ',       # $3
                    shell=True)


def wrann_script(bashlink, dir, ext_out, ext_in):
    """

    :param bashlink:
    :param dir:
    :param ext_out:
    :param ext_in:
    :return:
    """
    subprocess.call(bashlink + ' ' +
                    dir + '/ ' +        # $1
                    ext_out + ' ' +     # $2
                    ext_in,             # $3
                    shell=True)  # $5


def bxb_script(bashlink, dir_path, dir_report, ext_ref, ext_ai, fileout, fileout_stan):
    """

    :param bashlink: 
    :param dir_path: 
    :param dir_report:
    :param ext_ref: 
    :param ext_ai: 
    :param fileout: 
    :param fileout_stan: 
    :return: 
    """
    subprocess.call(bashlink + ' ' +
                    dir_path + '/ ' +       # $1
                    dir_report + '/ ' +     # $2
                    ext_ref + ' ' +         # $3
                    ext_ai + ' ' +          # $4
                    fileout + ' ' +         # $5
                    fileout_stan + ' ',     # $6
                    shell=True)


def epicmp_script(bashlink, dir_path, dir_report, fileout, ext_ref, ext_ai):
    """

    :param bashlink:
    :param dir_path:
    :param dir_report:
    :param fileout:
    :param ext_ref:
    :param ext_ai:
    :return:
    """
    subprocess.call(bashlink + ' ' +
                    dir_path + '/ ' +   # $1
                    dir_report + '/ ' + # $2
                    fileout + ' ' +     # $3
                    ext_ref + ' ' +     # $4
                    ext_ai + ' ',       # $5
                    shell=True)


def mxm_script(bashlink, dir_path, dir_report, ext_ref, ext_ai, fileout):
    """

    :param bashlink:
    :param dir_path:
    :param dir_report:
    :param ext_ref:
    :param ext_ai:
    :param fileout:
    :return:
    """

    subprocess.call(bashlink + ' ' +
                    dir_path + '/ ' +       # $1
                    dir_report + '/ ' +     # $2
                    fileout + ' ' +         # $3
                    ext_ref + ' ' +         # $4
                    ext_ai + ' ',           # $5
                    shell=True)


def del_result(dir_db, physionet_directory, output_ec57_directory):
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    rm_file(curr_dir + '/remove-script.sh',
            physionet_directory + dir_db,
            output_ec57_directory + dir_db,
            '1')


def ec57_eval(output_ec57_directory,
              annotation_dir,
              beat_ext_db,
              event_ext_db,
              beat_ext_ai,
              event_ext_ai,
              fs_origin=250):
    """
    :param output_ec57_directory:
    :param annotation_dir:
    :param beat_ext_db:
    :param event_ext_db:
    :param beat_ext_ai:
    :param event_ext_ai:
    :param half_ext:
    :param fs_origin:
    :return:
    """
    dataset = output_ec57_directory.split('/')[-2]
    if not isdir(output_ec57_directory):
        os.makedirs(output_ec57_directory)

    if beat_ext_ai is not None:
        bxb_script(PROJECT_DIR + 'script' + '/bxb-script.sh',
                   annotation_dir,
                   output_ec57_directory,
                   beat_ext_db,
                   beat_ext_ai,
                   dataset + '_qrs_report_line',
                   dataset + '_report_standard')
        # rann_script(PROJECT_DIR + 'script' + '/rdann-script.sh',
        #             annotation_dir,
        #             beat_ext_db,
        #             beat_ext_ai)
        #
        # HR_calcu(fs_origin,
        #          annotation_dir,
        #          beat_ext_db)
        #
        # HR_calcu(fs_origin,
        #          annotation_dir,
        #          beat_ext_ai)
        #
        # wrann_script(PROJECT_DIR + 'script' + '/wrann-script.sh',
        #              annotation_dir,
        #              'rr' + beat_ext_db,
        #              'hr' + beat_ext_db)
        #
        # wrann_script(PROJECT_DIR + 'script' + '/wrann-script.sh',
        #              annotation_dir,
        #              'rr' + beat_ext_ai,
        #              'hr' + beat_ext_ai)
        #
        # mxm_script(PROJECT_DIR + 'script' + '/mxm-script.sh',
        #            annotation_dir,
        #            output_ec57_directory,
        #            'rr' + beat_ext_db,
        #            'rr' + beat_ext_ai,
        #            dataset + '_HR_report')

    if event_ext_ai is not None:
        epicmp_script(PROJECT_DIR + 'script' + '/epicmp-script.sh',
                      annotation_dir,
                      output_ec57_directory,
                      dataset + '_AFib_report',
                      event_ext_db,
                      event_ext_ai)

