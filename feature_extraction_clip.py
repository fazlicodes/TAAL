import os
import glob
import argparse
import torch
import clip
from tqdm import tqdm
import numpy as np
import datasets as ds
# import utils as ut
from models import EmbModel
from torchvision.models import resnet18, resnet50
from sklearn.preprocessing import StandardScaler, normalize

def dino_resnet(pretrained=False):
    return torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

def main(args):

    print('Dataset: {}\n Extracting features with pretrained {} ({} backbone).'.format(args['dataset'],args['model_type'],args['model_subtype']))
    if args['use_pseudo']:
        print('.. from data pseudolabeled with {} and a {} confidence threshold (i.e. keep the top {} most confident images per predicted category)'.format(args['pseudolabel_model'],args['pseudo_conf'], args['pseudolabel_model']))
    print('\n')
    args['data_dir'] = os.path.join(args['root_data_dir'],args['dataset'], '')
    
    if args['use_pseudo']==True:
        args['metadata'] = os.path.join(args['data_dir'], args['dataset']+'_meta_{}_pseudo_{}.csv'.format(args['pseudolabel_model'],args['pseudo_conf']))
    else:
        args['metadata'] = os.path.join(args['data_dir'], args['dataset']+'_meta.csv')
   
    #####  Model loading #####
    if args['model_type'] == "clip":
        model, _ = clip.load(args['model_subtype'], device=args["device"], jit=False)
        model.eval()
        args['model_subtype'] ='{}_{}'.format(args['model_type'],args['model_subtype'])
    elif args['model_type'] == "ssl":
        raise NotImplementedError
        # if args['model_path']!='na':
        #     checkpoint = torch.load(args['model_path'])
        # else:
        #     # list_of_models = glob.glob('{}*.pt'.format(args['model_dir'])) # list all models in the dir (*.pt)
        #     # most_recent_model_path = max(list_of_models, key=os.path.getctime)
        #     # checkpoint = torch.load(most_recent_model_path)
        #     checkpoint = torch.load(args['model_dir'])
        breakpoint()
        model_args = checkpoint['args']
        model_args['im_res'] = args['im_res']
        args['pos_type'] = model_args['pos_type']
        args['model_subtype'] ='{}_{}_{}'.format(model_args['train_loss'],args['model_subtype'],model_args['pos_type'])
        base_encoder = eval(model_args['backbone'])
        model = EmbModel(base_encoder, model_args).to(args['device'])

        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(msg)
    elif args['model_type'] =="imagenet_transfer":
        base_encoder = eval('resnet50') #eval(args['backbone'])
        model = base_encoder(pretrained=True).to(args['device'])
        model.fc = torch.nn.Identity()
        model.eval()
        args['model_subtype'] ='{}_{}'.format('imagenet',args['model_subtype'])
    
    elif args['model_type'] =="dino":
        model = torch.hub.load('facebookresearch/dino:main', f'dino_{args["model_subtype"]}').to(args['device'])
        if args['model_subtype'] == 'resnet50':
            model.fc = torch.nn.Identity()
        elif 'vit' in args['model_subtype'] :
            model.head = torch.nn.Identity()
        model.eval()
        args['model_subtype'] ='{}_{}'.format('dino',args['model_subtype'])

    _,train_set_lin ,val_set_lin, test_set_lin , _, _ = ds.get_dataset(args)
    loaders={}
    loaders['train'] = torch.utils.data.DataLoader(train_set_lin, batch_size=args['batch_size'], 
                            num_workers=args['workers'], shuffle=False)
    loaders['val'] =   torch.utils.data.DataLoader(val_set_lin, batch_size=args['batch_size'], 
                            num_workers=args['workers'], shuffle=False)            
    loaders['test'] =   torch.utils.data.DataLoader(test_set_lin, batch_size=args['batch_size'], 
                            num_workers=args['workers'], shuffle=False)     

    ##### Feature extraction #####
    # split_set = ['train','val','test'] 
    split_set = ['train','test']   
    for split in split_set:
        dataiter = iter(loaders[split])
        model.eval()
        feature_list = []
        label_list = []
        with torch.no_grad():
            for train_step in tqdm(range(1, len(dataiter) + 1)):
                batch = next(dataiter)
                data = batch["im"].cuda()
            
                if args['model_type'] == "clip":
                    feature = model.visual(data.half())
                elif args['model_type']=="ssl":
                    op = model(data, only_feats=True)
                    feature = op['feat']
                elif args['model_type'] =="imagenet_transfer":
                    feature = model(data)
                elif args['model_type'] =="dino":
                    feature = model(data)

                feature = feature.cpu()
                for idx in range(len(data)):
                    feature_list.append(feature[idx].tolist())
                label_list.extend(batch["target"].cpu().tolist())
        os.makedirs(args['feature_path'], exist_ok=True)
        pseudo=''
        if args['use_pseudo']:
            pseudo='_{}_pseudolabels_{}'.format(args['pseudolabel_model'],args['pseudo_conf'])
        feature_dir = args['model_subtype'].replace('/','')
        os.makedirs(os.path.join(args['feature_path'], '{}_feat/'.format(feature_dir,)),exist_ok=True)
        save_filename =  "{}_feat/{}_{}_features{}".format(feature_dir,
                                                           feature_dir,
                                                           split,pseudo)
        
        
        np.savez(
            os.path.join(args['feature_path'], save_filename),
            feature_list=feature_list,
            label_list=label_list,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_data_dir",type=str,default='data/')
    parser.add_argument('--dataset', choices=['ucf101','food101','sun397','stanford_cars','fgvc-aircraft','caltech-101',
                                            'eurosat','resisc45','aid','patternnet','ucm','whurs19','mlrsnet','optimal31',
                                            'oxford_pets','oxford_flowers','dtd','imagenet','cifar10','cifar100',
                                              'kenya','cct20','serengeti','icct','fmow','oct'], type=str)
    parser.add_argument("--model_type", type=str, choices=["clip", "ssl","imagenet_transfer", "dino"], help="type of pretraining")
    parser.add_argument("--return_single_image",action="store_true",default=True)
    parser.add_argument("--model_subtype",type=str, choices=["ViT-B/32", "ViT-B/16","ViT-L/14", "RN50", "resnet50", "vitb16"],default="vitb16", help="exact type of pretraining backbone")
    parser.add_argument("--model_dir",default='',type=str, help="directory to load latest model available if not specific path is given through model_path")
    parser.add_argument("--model_path",default='na', type=str, help="path to torch model, if saved locally, just make sure its consistent with where ssl_pretraining.py saves the file to")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="which split")
    parser.add_argument("--feature_path",type=str,default='')
    parser.add_argument("--batch_size", type=int,default=512, help="dataloader batch size")
    parser.add_argument("--im_res", type=int,default=224, help="processed image resolution")
    parser.add_argument("--use_pseudo", action='store_true',default=True, help="use pseudolabels as extracted labels")
    parser.add_argument("--pseudo_conf",type=str,default='')
    parser.add_argument("--pseudolabel_model",type=str,choices=['georsclip','clip_GeoRSCLIP_ViTB32','clip_RN50','clip_ViT-L_14','ViT-B_32'], default='ViT-B_32')
    parser.add_argument("--workers", type=int,default=4)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],default="cuda",help="which device")
    parser.add_argument('--seed', default=2001, type=int)
    parser.add_argument('--half_precision', action='store_true',default=False, help="use half precision")
    
    args = vars(parser.parse_args())
    #ip_args = vars(parser.parse_args())

    main(args)