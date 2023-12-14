import subprocess
from parameters import *
import pandas as pd


def convert_edf_to_ascii(session_identifier):
    path_to_edf = f'results/{session_identifier}/{session_identifier}.EDF'
    subprocess.call([path_to_edf2asc, path_to_edf])


def make_res_df(session_identifier):
    asc_file = f'results/{session_identifier}/{session_identifier}.asc'
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
                image_load_list.append([int(img_timestamp), img_name])

            # [Step 2b]: Start capturing all gaze information (here we check for gaze data events).
            if not any(item.isalpha() for item in line.split()):
                # Record the gaze information:
                data_from_line = line.split()
                gaze_timestamp = data_from_line[0]  # The timestamp is always in index 0
                gaze_x = data_from_line[1]  # Gaze x is in index 1
                gaze_y = data_from_line[2]  # Gaze y is in index 2
                pupil_size = data_from_line[3]  # Pupil size is in index 3

                if gaze_x == '.' and gaze_y == '.':
                    # This is a blink event. For these events only write the timestamp to file:
                    eye_data_list.append([int(gaze_timestamp)])
                else:
                    eye_data_list.append([int(gaze_timestamp), float(gaze_x), float(gaze_y), float(pupil_size)])

            # Step [3]: Capture end of trial.
            if ('time_out' in line or 'key_pressed' in line or 'trial_skipped_by_user' in line or
                    'terminated_by_user' in line):
                data_from_line = line.split()
                end_timestamp = data_from_line[1]
                reason_end = data_from_line[2]
                end_trial_list.extend([int(end_timestamp), reason_end, int(-1)])
                # Stop recording data:
                recording_trial = False

            # When no RT is given, we've got all the information we need for this trial, so put it all in the df:
            if 'trial_skipped_by_user' in line or 'terminated_by_user' in line:
                res_df.loc[len(res_df.index)] = [trigger_timestamp, eye_data_list,
                                                 image_load_list, end_trial_list]
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
            end_trial_list[-1] = int(line.split()[5])

            # We've got all the information we need for this trial, so put it all in the df:
            res_df.loc[len(res_df.index)] = [trigger_timestamp, eye_data_list,
                                             image_load_list, end_trial_list]
            # Initialise all lists back to empty to store next trial data:
            trigger_timestamp, image_load_list, eye_data_list, end_trial_list = 0, [], [], []

    # Save the DF to file
    res_df.to_pickle(f'results/{session_identifier}/{session_identifier}')


# For debugging:
test_session_identifier = 'test_2023_12_14_11_38'
# convert_edf_to_ascii(test_session_identifier)
make_res_df(test_session_identifier)
# To read back the pickled result:
# df = pd.read_pickle(f'results/{test_session_identifier}/{test_session_identifier}')


