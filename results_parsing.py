import subprocess
from parameters import *
import pandas as pd


def convert_edf_to_ascii(session_identifier):
    path_to_edf = f'results/{session_identifier}/{session_identifier}.EDF'
    subprocess.call([path_to_edf2asc, path_to_edf])


def make_res_df(asc_file):
    # Make a pandas dataframe to store the results in.

    # Each row of our df contains information from 1 trial:
    res_df = pd.DataFrame(columns=['trigger_fired', 'eye_data_gaze', 'eye_data_img', 'end_trial'])

    # Store timestamps of each trigger:
    trigger_timestamp = 0
    # Store timestamps, image names and positions of each image loaded (1st presentation of the image):
    image_load_list = []
    # Store timestamps, gaze x and y positions, and events that show the image being updated (gaze-contingent):
    eye_data_list = []
    # Store the timestamp, reason for ending trial, and response time (if available):
    end_trial_list = []

    # Flag to start recording information for a given trial:
    recording_trial = False

    # Read the ascii file line-by-line:
    f = open(asc_file, 'r')
    for line in f:
        if recording_trial:
            # [Step 2a]: Start capturing all gaze information (here we check for image change events).
            if 'IMGLOAD CENTER' in line:
                # Record the image being shown:
                data_from_line = line.split()
                img_timestamp = data_from_line[1]  # The timestamp is always in index 1
                img_name = data_from_line[5]  # The img name (incl relative path) is in index 5
                start_pos = (int(data_from_line[6]), int(data_from_line[7]))  # Index starting x, y of where to show img
                end_pos = (int(data_from_line[8]), int(data_from_line[9]))  # Index ending x, y of where to show img
                image_load_list.append([img_timestamp, img_name, start_pos, end_pos])

            # [Step 2b]: Start capturing all gaze information (here we check for gaze data events).
            if not any(item.isalpha() for item in line.split()):
                # Record the gaze information:
                data_from_line = line.split()
                gaze_timestamp = data_from_line[0]  # The timestamp is always in index 0
                gaze_x = data_from_line[1]  # Gaze x is in index 1
                gaze_y = data_from_line[2]  # Gaze y is in index 2
                pupil_size = data_from_line[3]  # Pupil size is in index 3
                eye_data_list.append([gaze_timestamp, gaze_x, gaze_y, pupil_size])

            # Step [3]: Capture end of trial.
            if ('time_out' in line or 'key_pressed' in line or 'trial_skipped_by_user' in line or
                    'terminated_by_user' in line):
                data_from_line = line.split()
                end_timestamp = data_from_line[1]
                reason_end = data_from_line[2]
                end_trial_list.append([end_timestamp, reason_end, -1])
                # Stop recording data:
                recording_trial = False

            # When no RT is given, we've got all the information we need for this trial, so put it all in the df:
            if 'trial_skipped_by_user' in line or 'terminated_by_user' in line:
                res_df.loc[len(res_df.index)] = [trigger_timestamp, image_load_list,
                                                 eye_data_list, end_trial_list]
                # Initialise all lists back to empty to store next trial data:
                trigger_timestamp, image_load_list, eye_data_list, end_trial_list = 0, [], [], []

        # [Step 1]: Get the start of the experiment.
        if 'fix_trigger_fired' in line:
            recording_trial = True
            # The trigger was fired and the experiment can begin:
            trigger_timestamp = int(line.split()[1])  # The timestamp is always in index 1

        # Step [4]: Add RT.
        if 'TRIAL_VAR RT' in line:
            # Overwrite last element in list of lists with a value:
            end_trial_list[-1][-1] = line.split()[5]

            # We've got all the information we need for this trial, so put it all in the df:
            res_df.loc[len(res_df.index)] = [trigger_timestamp, image_load_list,
                                             eye_data_list, end_trial_list]
            # Initialise all lists back to empty to store next trial data:
            trigger_timestamp, image_load_list, eye_data_list, end_trial_list = 0, [], [], []

    print('a')

# convertEDFToAscii('test_2023_12_07_16_02')
make_res_df(asc_file = 'results/test_2023_12_09_12_54/test_2023_12_09_12_54.asc')


