from parameters import patch_size

VIDEO_SIZE = [1080, 1088]
CALIB_SHIFT = [0,0]
PATCH_SIZE = patch_size
INP_CHANNELS = 1
BINARY_SIMULATION = True
RECON_CHANNELS = 3
OUT_ACTIVATION = 'sigmoid'



# Input directories

ET_GAZE_DIR = 'sample/gaze_positions.csv'
ET_VIDEO_DIR = 'sample/world.mp4'

JAAP_ENC_DIR = "spvPlayer/models/exp4_B_S1_650_best_encoder.pth"
JAAP_MAP_DIR = "spvPlayer/models/phosphene_map_exp4.pt"

ASH_ENC_DIR = "spvPlayer/models/enc2_20230626_105329_35"

# Whether to use 'cpu' or 'cuda:0'
DEVICE_TYPE = 'cuda:0'
