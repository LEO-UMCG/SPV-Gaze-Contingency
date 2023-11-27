import torch
from spvPlayer.viseon_eye.model import E2E_Encoder_exp4, Simulator_exp4
from torchvision import transforms
# from matplotlib import pyplot as plt
from spvPlayer.config import *

DEVICE = torch.device(DEVICE_TYPE)

def prepJaapEncoder():

    encoder = E2E_Encoder_exp4(in_channels=INP_CHANNELS,
                                         binary_stimulation=BINARY_SIMULATION).to(DEVICE)
    # encoder.cuda()

    encoder.eval()
    encoder.load_state_dict(torch.load(JAAP_ENC_DIR, map_location=torch.device(DEVICE_TYPE)))
    return encoder

def prepSimulator():


    simulator = Simulator_exp4(device=DEVICE, pMap_from_file=JAAP_MAP_DIR)
    return simulator

# decoder = E2E_Decoder(out_channels=RECON_CHANNELS,
#                                         out_activation=OUT_ACTIVATION).to(DEVICE)

# pLocs = simulator.get_center_of_phosphenes()


# decoder.load_state_dict(torch.load("/media/ashdev/AshMem/E2E_exp4/exp4_B_S1_650_best_decoder.pth"))
# simulator.load_state_dict(torch.load("/media/ashdev/AshMem/E2E_exp4/phosphene_map_exp4.pt"))


# input_image = Image.open("dog.jpg")
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Grayscale()
])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) 


# simulator.eval()
# simulator.train(False)

def jaapPredict(img, encoder, simulator):
    # img = torch.from_numpy(img)
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) 
    with torch.no_grad():
        output = encoder(input_batch.to(DEVICE))
        spv = simulator(output)
        return spv.cpu().numpy()[0,0]