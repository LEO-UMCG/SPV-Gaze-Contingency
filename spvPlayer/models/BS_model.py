import torch
from torchvision import transforms
from spvPlayer.BS_SPV.models import *
import noise
import numpy as np
from spvPlayer.config import *

DEVICE = torch.device(DEVICE_TYPE)


def get_pMask_jaap(size=(PATCH_SIZE*2,PATCH_SIZE*2),phosphene_density=32,seed=1,
              jitter_amplitude=0., intensity_var=0.,
              dropout=False,perlin_noise_scale=.4):

    # Define resolution and phosphene_density
    [nx,ny] = size
    n_phosphenes = phosphene_density**2 # e.g. n_phosphenes = 32 x 32 = 1024
    pMask = torch.zeros(size)


    # Custom 'dropout_map'
    p_dropout = perlin_noise_map(shape=size,scale=perlin_noise_scale*size[0],seed=seed)
    np.random.seed(seed)

    for p in range(n_phosphenes):
        i, j = divmod(p, phosphene_density)
       
        jitter = np.round(np.multiply(np.array([nx,ny])//phosphene_density,
                                      jitter_amplitude * (np.random.rand(2)-.5))).astype(int)
        rx = (j*nx//phosphene_density) + nx//(2*phosphene_density) + jitter[0]
        ry = (i*ny//phosphene_density) + ny//(2*phosphene_density) + jitter[1]

        rx = np.clip(rx,0,nx-1)
        ry = np.clip(ry,0,ny-1)
        
        intensity = intensity_var*(np.random.rand()-0.5)+1.
        if dropout==True:
            pMask[rx,ry] = np.random.choice([0.,intensity], p=[p_dropout[rx,ry],1-p_dropout[rx,ry]])
        else:
            pMask[rx,ry] = intensity
            
    return pMask    

def perlin_noise_map(seed=0,shape=(PATCH_SIZE*2,PATCH_SIZE*2),scale=100,octaves=6,persistence=.5,lacunarity=2.):
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        base=seed)
    out = (out-out.min())/(out.max()-out.min())
    return out



def prepAshEncoder():

    encoder = Representer(arryaOut=False).to(DEVICE)
    if 'cuda' in DEVICE_TYPE:
        encoder.cuda()

    encoder.eval()
    encoder.load_state_dict(torch.load(ASH_ENC_DIR, map_location=torch.device(DEVICE_TYPE)))
    return encoder


def prepRegSimulator():


    pMask = get_pMask_jaap()
    simulator = E2E_PhospheneSimulator_jaap(pMask=pMask, device=DEVICE)
    simulator.to(DEVICE)
    if 'cuda' in DEVICE_TYPE:
        simulator.cuda()
    return simulator


gray2color = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
transes = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(size=PATCH_SIZE*2),
            gray2color
            ])

def ashPredict(img, encoder, simulator, toggle=False):
    input_tensor = transes(img)
    input_batch = input_tensor.unsqueeze(0) 
    with torch.no_grad():
        output = encoder(input_batch.to(DEVICE))
        if toggle: 
            res = output.clone()
            res[output==0] = 1
            res[output==1] = 0
            output = res
        spv = simulator(output)
        
        return spv.cpu().numpy()[0,0]