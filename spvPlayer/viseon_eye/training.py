## Import statements
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

#added
from csv import writer as wrt

# Local dependencies
import model
import utils
import local_datasets
# from simulator.simulator import GaussianSimulator
# from simulator.init import init_probabilistically
# from simulator.utils import load_params

# from loss_functions import CustomLoss, CustomLoss_jaap
from loss_functions import CustomLoss_jaap

# PyTorch
import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import gc


def initialize_components(cfg):
    """This function returns the required model, dataset and optimization components to initialize training.
    input: <cfg> training configuration (pandas series, or dataframe row)
    returns: dictionaries with the required model components: <models>, <datasets>,<optimization>, <train_settings>
    """

    # Random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Models
    models = dict()


    if cfg.binary_stimulation == True:
#         if cfg.j_exp_no == 0:
#             models['encoder'] = model.E2E_Encoder_Binary(in_channels=cfg.input_channels,
#                                         binary_stimulation=cfg.binary_stimulation).to(cfg.device)
            
# #            models['encoder'] = model.E2E_Encoder_Binarized(in_channels=cfg.input_channels,
# #                                        binary_stimulation=cfg.binary_stimulation).to(cfg.device)
        if cfg.j_exp_no == 3:
            models['encoder'] = model.E2E_Encoder_exp3(in_channels=cfg.input_channels,
                                                       binary_stimulation=cfg.binary_stimulation).to(cfg.device)

        elif cfg.j_exp_no == 4:
            models['encoder'] = model.E2E_Encoder_exp4(in_channels=cfg.input_channels,
                                                       binary_stimulation=cfg.binary_stimulation).to(cfg.device)

    # elif cfg.binary_stimulation == False:    
    #     models['encoder'] = model.E2E_Encoder_Continuous(in_channels=cfg.input_channels,
    #                                     binary_stimulation=cfg.binary_stimulation).to(cfg.device)
        
                    
    models['decoder'] = model.E2E_Decoder(out_channels=cfg.reconstruction_channels,
                                          out_activation=cfg.out_activation).to(cfg.device)

    if cfg.j_exp_no == 3:
        pMask = utils.get_pMask_jaap(jitter_amplitude=0, dropout=False, seed=cfg.seed) # phosphene mask with regular mapping
        models['simulator'] = model.E2E_PhospheneSimulator_jaap(pMask=pMask.to(cfg.device),
                                                                sigma=1.5,
                                                                intensity=10, device=cfg.device).to(cfg.device)
        pLocs = None

    elif cfg.j_exp_no == 4:
        pMask = utils.get_pMask_jaap(intensity_var=1, jitter_amplitude=0.5, dropout=False, seed=8)
        models['simulator'] = model.Simulator_exp4(device=cfg.device)
    
        pLocs = models['simulator'].get_center_of_phosphenes()
    
    # use_cuda = False if cfg.device=='cpu' else True
    # elif cfg.j_exp_no == 0:
        # if cfg.simulation_type == 'realistic':
        #     params = load_params('simulator/config/params.yaml')
        #     r, phi = init_probabilistically(params,n_phosphenes=1024) #1024
        #     print('r',r)
        #     print('phi', phi)
        #     models['simulator'] = model.E2E_RealisticPhospheneSimulator(cfg, params, r, phi).to(cfg.device)
        # # else:
        # if cfg.simulation_type == 'regular':
        #     pMask = utils.get_pMask(jitter_amplitude=0,dropout=False) # phosphene mask with regular mapping
        # elif cfg.simulation_type == 'personalized':
        #     pMask = utils.get_pMask(seed=1,jitter_amplitude=.5,dropout=True,perlin_noise_scale=.4) # pers. phosphene mask
        # models['simulator'] = model.E2E_PhospheneSimulator(pMask=pMask.to(cfg.device),
        #                                                 sigma=1.5,
        #                                                 intensity=15,
        #                                                 device=cfg.device).to(cfg.device)

    # Dataset
    dataset = dict()
    if cfg.dataset == 'characters':
        directory = './datasets/Characters/'
        trainset = local_datasets.Character_Dataset(device=cfg.device, directory=directory)
        valset = local_datasets.Character_Dataset(device=cfg.device, directory=directory, validation = True)
    elif cfg.dataset == 'ADE20K':
        # directory = './datasets/ADE20K/'
        #directory = '/content/drive/MyDrive/ADE20K/'
        directory = '/home/burkuc/data/ADE20K/'
        load_preprocessed = True if os.path.exists(directory+'/images/processed_train') and os.path.exists(directory+'/images/processed_val') else False
        # load_preprocessed = True
        circular_mask = True if cfg.j_exp_no==0 else False
        trainset = local_datasets.ADE_Dataset(device=cfg.device, directory=directory, load_preprocessed=load_preprocessed, circular_mask=circular_mask, normalize=False)
        valset = local_datasets.ADE_Dataset(device=cfg.device, directory=directory, validation=True, load_preprocessed=load_preprocessed, circular_mask=circular_mask, normalize=False)
    dataset['trainloader'] = DataLoader(trainset,batch_size=int(cfg.batch_size),shuffle=True)
    dataset['valloader'] = DataLoader(valset,batch_size=int(cfg.batch_size),shuffle=False)

    # Optimization
    optimization = dict()
    if cfg.optimizer == 'adam':
        optimization['encoder'] = torch.optim.Adam(models['encoder'].parameters(),lr=cfg.learning_rate)
        optimization['decoder'] = torch.optim.Adam(models['decoder'].parameters(),lr=cfg.learning_rate)
    elif cfg.optimizer == 'sgd':
        optimization['encoder'] = torch.optim.SGD(models['encoder'].parameters(),lr=cfg.learning_rate)
        optimization['decoder'] = torch.optim.SGD(models['decoder'].parameters(),lr=cfg.learning_rate)
    
    # if cfg.j_exp_no == 0:
    #     optimization['lossfunc'] = CustomLoss(recon_loss_type=cfg.reconstruction_loss,
    #                                                 recon_loss_param=cfg.reconstruction_loss_param,
    #                                                 stimu_loss_type=cfg.sparsity_loss,
    #                                                 phosrep_loss_type=cfg.representation_loss,
    #                                                 phosrep_loss_param=cfg.representation_loss_param,
    #                                                 kappa_repr=cfg.kappa_repr,
    #                                                 kappa_spars=cfg.kappa_spars,
    #                                                 device=cfg.device)
    # else:
    optimization['lossfunc'] = CustomLoss_jaap(recon_loss_type=cfg.reconstruction_loss,
                                                recon_loss_param=cfg.reconstruction_loss_param,
                                                stimu_loss_type=cfg.sparsity_loss,
                                                kappa=cfg.kappa_spars,
                                                phosphene_regularization= 'bce', ##None, # cfg.phosphene_regularization,
                                                reg_weight=1.0, #cfg.reg_weight,                                                
                                                device=cfg.device,
                                                phosphene_loss_param= 0.925, #cfg.phosphene_loss_param,
                                                plocs=pLocs)                                   
    
    # Additional train settings
    train_settings = dict()
    if not os.path.exists(cfg.savedir):
        os.makedirs(cfg.savedir)
    train_settings['model_name'] = cfg.model_name
    train_settings['savedir']=cfg.savedir
    train_settings['n_epochs'] = cfg.n_epochs
    train_settings['log_interval'] = cfg.log_interval
    train_settings['convergence_criterion'] = cfg.convergence_crit
    # train_settings['binned_stimulation'] = cfg.binned_stimulation
    # if cfg.binned_stimulation:
    #     models['intensities_array'] = torch.linspace(0,params['encoding']['max_stim'],params['encoding']['n_config']+1,device=cfg.device)
    return models, dataset, optimization, train_settings
    


def train(models, dataset, optimization, train_settings, tb_writer, cfg):
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    ## A. Unpack parameters
   
    # Models
    encoder   = models['encoder']
    decoder   = models['decoder']
    simulator = models['simulator']

    print('encoder', encoder)
#    print('decoder', decoder)
#    print('simulator', simulator)
      
    total_params = sum(param.numel() for param in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print('trainable params:', trainable_params)
    print('total params:' , total_params)

    


    # if train_settings['binned_stimulation']:
    #     intensities_array = models['intensities_array']

    # Dataset
    trainloader = dataset['trainloader']
    valloader   = dataset['valloader']
    print('trainloader length', len(trainloader))
    print('vallloader length', len(valloader))
    
    # Optimization
    encoder_optim = optimization['encoder']
    decoder_optim = optimization['decoder']
    loss_function = optimization['lossfunc']

    
    # Train settings
    model_name   = train_settings['model_name']
    savedir      = train_settings['savedir'] + '/'+ model_name
    n_epochs     = train_settings.get('n_epochs',2)
    log_interval = train_settings.get('log_interval',10)
    converg_crit = train_settings.get('convergence_criterion',50)
    print('log_interval',log_interval)
    
    ## B. Logging
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    #added
    # if not os.path.exists(savedir+'/'+model_name):
    #     os.makedirs(savedir+'/'+model_name)
    logger = utils.Logger(os.path.join(savedir, 'out_' + model_name + '.log'))
    csvpath = os.path.join(savedir,model_name+'_train_stats.csv')
    logstats = list(loss_function.stats.keys())
    with open(csvpath, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['epoch','i']+logstats)
    

    logger(cfg)
    logger(encoder)
    logger(simulator)
    logger(decoder)
    logger('trainable_params:'+ str(trainable_params))
    logger('total params:'+ str(total_params))  
    logger(loss_function)
  
    ## C. Training Loop
    n_not_improved = 0
    running_loss = 0.0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        count_samples=1
        count_val=1
        # count_backwards=1
        logger('Epoch %d' % (epoch+1))

#        for i, data in enumerate(tqdm(trainloader, desc='Training'), 0):
        for i, data in enumerate(trainloader):
            image,label = data
#            print('train image shape',image.shape, 'i', i)
            # print(np.unique(label.cpu()))
            # print('in training loop')
            # TRAINING
            encoder.train()
            decoder.train()
            encoder.zero_grad()
            decoder.zero_grad()

            # 1. Forward pass
            stimulation = encoder(image)
 #           print(f"stimulation: {stimulation.shape}")
            # proxy_stim = 30*torch.ones_like(stimulation)
            phosphenes  = simulator(stimulation)
            # print(f"phosphenes: {phosphenes.shape}")
            reconstruction = decoder(phosphenes)
            # print(f"reconstruction: {reconstruction.shape}")
            # print(f"passed sample through models {count_samples} times")
            
            # plt.imshow(image.cpu().detach().numpy()[0][0], cmap='gray')
            # plt.show()
            # plt.imshow(label.cpu().detach().numpy()[0][0], cmap='gray')
            # plt.show()
            # plt.imshow(phosphenes.cpu().detach().numpy()[0][0], cmap='gray')
            # plt.show()
            # plt.imshow(reconstruction.cpu().detach().numpy()[0][0], cmap='gray')
            # plt.show()
            
            
            
            count_samples+=1
            # 2. Calculate loss
            loss = loss_function(image=image,
                                 label=label,
                                 stimulation=stimulation, #encoder.out,
                                 phosphenes=phosphenes,
                                 reconstruction=reconstruction)
            # print("calculated loss")
            
            # 3. Backpropagation
            loss.backward()
            
            # print("backward step")
            encoder_optim.step()
            decoder_optim.step()
            # print("optimizer step")
            # print(f"optimizer step {count_backwards}")
            # count_backwards+=1
            del loss
            # running_loss += loss.item()

            # VALIDATION
            if i==len(trainloader) or i % log_interval == 0:
                # print("Running validation loop")
                # tb_writer.add_scalar('Loss/train', running_loss/log_interval, epoch * len(trainloader) + i)
                # running_loss = 0.0

                tb_writer.add_histogram(f'{model_name}/stimulation',stimulation,epoch * len(trainloader) + i)

                # utils.log_gradients_in_model(encoder, f'{model_name}/encoder', tb_writer, epoch * len(trainloader) + i)
                # utils.log_gradients_in_model(decoder, f'{model_name}/decoder', tb_writer, epoch * len(trainloader) + i)
                #changed
                filters_enc = utils.log_gradients_in_model(encoder, f'{model_name}/encoder', tb_writer, epoch * len(trainloader) + i)
                filters_dec = utils.log_gradients_in_model(decoder, f'{model_name}/decoder', tb_writer, epoch * len(trainloader) + i)
                
                fig_enc_filter = utils.full_fig2(filters_enc)
                plt.savefig(savedir+'/'+model_name+'_filters_enc_'+ str(epoch) + '_'+ str(i) +'.png')
                fig_dec_filter = utils.full_fig2(filters_dec)
                plt.savefig(savedir+'/'+model_name+'_filters_dec_'+ str(epoch) + '_'+ str(i)  +'.png')
                
                tb_writer.add_figure(f'{model_name}/encoder filters',fig_enc_filter,epoch * len(trainloader) + i)      
                tb_writer.add_figure(f'{model_name}/decoder filters',fig_dec_filter,epoch * len(trainloader) + i)  


                # print(stimulation)
                count_val+=1
                try:
                    sample_iter = np.random.randint(0,len(valloader))
#                    for j, data in enumerate(tqdm(valloader, leave=False, position=0, desc='Validation'), 0):
                    for j, data in enumerate(valloader):
                        image,label = data #next(iter(valloader))
                        # print(label)
 #                       print('validation image shape',image.shape)
                        encoder.eval()
                        decoder.eval()

                        with torch.no_grad():

                            # 1. Forward pass
                            stimulation = encoder(image)
                            # if train_settings['binned_stimulation']:
                            #     stimulation = utils.pred_to_intensities(intensities_array, stimulation)
                            phosphenes  = simulator(stimulation)
                            reconstruction = decoder(phosphenes)  

                            # plt.imshow(image.cpu().detach().numpy()[0][0], cmap='gray')
                            # plt.show()
                            # plt.imshow(label.cpu().detach().numpy()[0][0], cmap='gray')
                            # plt.show()
                            # plt.imshow(phosphenes.cpu().detach().numpy()[0][0], cmap='gray')
                            # plt.show()
                            # plt.imshow(reconstruction.cpu().detach().numpy()[0][0], cmap='gray')
                            # plt.show() 

                            if j==sample_iter: #save for plotting
                                sample_img = image
                                sample_phos = phosphenes
                                sample_recon = reconstruction
                                

                            # 2. Loss
                            _ = loss_function(image=image,
                                                label=label,
                                                stimulation=stimulation, #encoder.out,
                                                phosphenes=phosphenes,
                                                reconstruction=reconstruction,
                                                validation=True) 

                    reset_train = True if i==len(trainloader) else False #reset running losses if at end of loop, else keep going
                    
                    if cfg.j_exp_no == 0:
                        stats = loss_function.get_stats(reset_train,reset_val=True) #reset val loss always after validation loop completes
                    else:
                        stats = loss_function.get_stats()

                    tb_writer.add_scalars(f'{model_name}/Loss/validation', {key: stats[key][-1] for key in ['val_recon_loss','val_stimu_loss','val_phosrep_loss','val_total_loss']}, epoch * len(trainloader) + i)
                    tb_writer.add_scalars(f'{model_name}/Loss/training', {key: stats[key][-1] for key in ['tr_recon_loss','tr_stimu_loss','tr_phosrep_loss','tr_total_loss']}, epoch * len(trainloader) + i)

                    #added
                    # with open(csvpath, 'w') as csvfile:
                    with open(csvpath, 'a', newline='') as write_obj:
                        # csv_writer = csv.writer(csvfile, delimiter=',',
                        #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        csv_writer = wrt(write_obj)
                        csv_writer.writerow([epoch,i]+[stats[key][-1] for key in ['tr_recon_loss','val_recon_loss', 'tr_total_loss', 'val_total_loss','tr_stimu_loss','val_stimu_loss','tr_phosrep_loss','val_phosrep_loss']])
                        write_obj.close()

                    # sample = np.random.randint(0,sample_img.shape[0],5)
                    sample = np.random.choice(sample_img.shape[0],5, replace=False)
                    # print('sample', sample, sample_img.shape[0])
                    fig = utils.full_fig(sample_img[sample], sample_phos[sample], sample_recon[sample])
                            
                    #added

                    # fig_enc_filter = utils.full_fig2(filters_enc)
                    # plt.savefig(savedir+model_name+'_filters_enc_'+ str(epoch) + '_'+ str(i) +'.png')
                    # fig_dec_filter = utils.full_fig2(filters_dec)
                    # plt.savefig(savedir+model_name+'_filters_dec_'+ str(epoch) + '_'+ str(i)  +'.png')

                    fig_enc_features = utils.full_fig3(encoder, sample_img[sample])
                    plt.savefig(savedir+'/'+model_name+'_features_enc_'+ str(epoch) + '_'+ str(i) +'.png')
                    fig_dec_features = utils.full_fig3(decoder, sample_phos[sample])
                    plt.savefig(savedir+'/'+model_name+'_features_dec_'+ str(epoch) + '_'+ str(i)  +'.png')

                    # fig_enc = utils.full_fig2(filters_enc)
                    # for f, fig in enumerate(fig_enc):
                      # plt.savefig(savedir+model_name+'filterenc_'+ str(epoch) + '_'+ str(i) + '_filter_' + str(f) +'.png')
                    # fig_dec = utils.full_fig2(filters_dec)
                    # for f,fig in enumerate(fig_dec):
                    #   plt.savefig(savedir+model_name+'filterdec_'+ str(epoch) + '_'+ str(i) + '_filter_' + str(f)  +'.png')


                    # fig.show()
                    tb_writer.add_figure(f'{model_name}/predictions, phosphenes and reconstruction',fig,epoch * len(trainloader) + i)  
                    #added
                    # tb_writer.add_figure(f'{model_name}/encoder filters',fig_enc_filter,epoch * len(trainloader) + i)  
                    # tb_writer.add_figure(f'{model_name}/decoder filters',fig_dec_filter,epoch * len(trainloader) + i)  

                    tb_writer.add_figure(f'{model_name}/encoder features',fig_enc_features,epoch * len(trainloader) + i)  
                    tb_writer.add_figure(f'{model_name}/decoder features',fig_dec_features,epoch * len(trainloader) + i)  
                    
                    # np.save(savedir+model_name+'_enc_filters.npy',fig_enc)
                    # np.save(savedir+model_name+'_dec_filters.npy',fig_dec)
                    
                    # 5. Save model (if best)
                    if  np.argmin(stats['val_total_loss'])+1==len(stats['val_total_loss']):
                        savepath = os.path.join(savedir,model_name + '_best_encoder.pth' ) #'_e%d_encoder.pth' %(epoch))#,i))
                        logger('Saving best model ' + str(epoch)+ '_' + str(i) + ' to ' + savepath + '...')
                        torch.save(encoder.state_dict(), savepath)

                        savepath = os.path.join(savedir,model_name + '_best_decoder.pth' ) #'_e%d_decoder.pth' %(epoch))#,i))
                        logger('Saving best model ' + str(epoch)+ '_' + str(i) + ' to ' + savepath + '...')
                        torch.save(decoder.state_dict(), savepath)
                        
                        for tag,img in zip(['orig','phos','recon'],[sample_img[sample],sample_phos[sample],sample_recon[sample]]):
                            savepath = os.path.join(savedir,model_name + '_'+tag+'_img_'+str(epoch)+'.npy' )
                            img = img.detach().cpu().numpy()
                            with open(savepath, 'wb') as f:
                                np.save(f, img)
                                
                        n_not_improved = 0
                    else:
                        n_not_improved = n_not_improved + 1
                        logger('not improved for %5d iterations' % n_not_improved) 
                        savepath = os.path.join(savedir,model_name + '_last_encoder.pth' ) #'_e%d_encoder.pth' %(epoch))#,i))
                        logger('Saving last model ' + str(epoch)+ '_' + str(i) + ' to ' + savepath + '...')
                        torch.save(encoder.state_dict(), savepath)

                        savepath = os.path.join(savedir,model_name + '_last_decoder.pth' ) #'_e%d_decoder.pth' %(epoch))#,i))
                        logger('Saving last model ' + str(epoch)+ '_' +  str(i) + ' to ' + savepath + '...')
                        torch.save(decoder.state_dict(), savepath)
                        if n_not_improved>converg_crit:
                            print('conv cri')
                            break

                    # 5. Prepare for next iteration
                    encoder.train()
                    decoder.train()            

                except StopIteration:
                    pass
        if n_not_improved>converg_crit:
            break
        # added
        gc.collect()
    logger('Finished Training')
    tb_writer.close()
    return {'encoder': encoder, 'decoder':decoder}, loss_function.stats



if __name__ == '__main__':
    import pandas as pd
    
    args = utils.get_args()
    cfg = pd.Series(vars(args))
    print(cfg)
    models, dataset, optimization, train_settings = initialize_components(cfg)
    writer = SummaryWriter()
    writer.add_text("Config", cfg.to_string())
    writer.add_text("Model", 'old commit, new thresholding added')
    train(models, dataset, optimization, train_settings, writer,cfg)

