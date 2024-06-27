import ast
import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir =   '{}{}/'.format(args['root_data_dir'],args['dataset'])
    metadir = data_dir+'{}_meta.csv'.format(args['dataset'])
    meta = pd.read_csv(metadir,index_col=0)
    meta['category_id'] = meta.category_id.astype(int)
    split_list = ['train']
    sampled = meta.loc[meta.img_set.isin(split_list)]

    ### Added for ZS classifier
    label_df = meta[['category_id', 'label']]
    label_df = label_df.drop_duplicates(subset=['label'])
    cls2id = dict(zip(label_df.label, label_df.category_id))

    label_mapper = meta.drop_duplicates(subset=['label'])['label'].reset_index()
    label_mapper.drop(columns=['index'],inplace=True)
    label_mapper['clip_label'] = label_mapper.label.apply(lambda x: x.replace('_',' '))
    if args['dataset'] in ['aid','eurosat','patternnet','resisc45','ucm','whurs19','mlrsnet','optimal31']:
        text_pr = "a centered satellite image of "
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: '{}{}.'.format(text_pr,x))
    elif args['dataset']=="imagenet":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of a {}.'.format(x))
    else:
        print('No dataset found')

    clip_to_target = dict(label_mapper[['clip_label','label']].values)
    ## label to pretrained model id mapping
    grouped_label_to_clip_ids = pd.DataFrame(clip_to_target.values()).reset_index().rename(columns={'index':'clip_id',0:'dataset_label'}).groupby('dataset_label').clip_id.apply(list)
    N = sampled.shape[0]

    datasets_names = ['eurosat','ucm','aid','patternnet','resisc45','whurs19','mlrsnet','optimal31', 'caltech-101','oxford_pets','oxford_flowers', 'imagenet']
    map_datasets_name = ['EuroSAT', 'UCM', 'AID', 'PatternNet', 'RESISC45', 'WHURS19', 'MLRSNet', 'Optimal31','Caltech101','OxfordPets','OxfordFlowers','ImageNet']
    dataset = map_datasets_name[datasets_names.index(args['dataset'])]

    ### Load CLIP model
    model, preprocess = clip.load(args['model_subtype'], device=device, jit=False)
    dataset = map_datasets_name[datasets_names.index(args['dataset'])]
    classifier_weights = torch.load(f'/l/users/sanoojan.baliah/Felix/RS_zero_shot/embeddings/lafter_{dataset}_ZSembeddings.pt').squeeze()
    classifier_weights = classifier_weights[list(cls2id.values())].to(device)

    image_links = list(sampled.img_path)
    targets = list(sampled.label)
    target_indices = sampled['category_id'].values
    clip_labels = list(clip_to_target.keys())

    pred_df = []
    correct_list = []

    # Determine batch_size based on available GPU memory
    batch_size = 6144  # Adjust based on your GPU

    for i in range(0, N, batch_size):
        end_index = min(i + batch_size, N)
        print('{}% done'.format(i / N * 100))

        # Pre-load and preprocess images (outside the main loop, if needed)
        preprocessed_images = [
            preprocess(Image.open(os.path.join(data_dir, img_path))).unsqueeze(0).to(device)
            for img_path in image_links[i:end_index]
        ]
        batch_images = torch.cat(preprocessed_images, dim=0)

        with torch.no_grad():
            logits_per_image, _ = model.forward_with_text_desc(batch_images, classifier_weights)
            probs = logits_per_image.softmax(dim=-1)

        probs_cpu = probs.cpu().numpy()

        for b_index in range(len(probs_cpu)):
            prob = probs_cpu[b_index]
            target = targets[i + b_index]
            predicted_label = np.argmax(prob)
            pred_correct = target_indices[i + b_index] == predicted_label
            correct_list.append(pred_correct)

            pred_dict = {
                # Add additional fields as needed, e.g., 'img_path', 'prob1', 'target', etc.
                'correct': pred_correct,
                'target': target
            }

            pred_df.append(pred_dict)

    # Compute the mean accuracy
    acc_mean = 100 * np.mean(correct_list)
    print(f'Accuracy mean: {acc_mean:.2f}%')

    # Create DataFrame outside of the loop
    pred_df = pd.DataFrame(pred_df)

    clip_model = 'clip_'+args['model_subtype'].replace('/','').replace('-','_')
    save_line = "{},{}, {} Shot, Test acc stat: {:.2f} ()\n".format(args['dataset'],clip_model, num_shot, acc_mean, '')
    print(save_line, flush=True)

    pred_df['img_path_trimmed'] = pred_df['img_path'].apply(lambda x: x.replace(data_dir,''))
    label_to_category = dict(meta[['label','category_id']].drop_duplicates().values)

    predicted_labels = set(pred_df.pred1)
    sub_df = pred_df.copy()
    sub_df = sub_df.rename(columns={'target':'label'})
    images_left =sub_df.shape[0]
    unique_pred = sub_df.pred1.nunique()

    pseudo_df = pd.DataFrame()
    for pred_label in predicted_labels:
        sub_label_df = pred_df.loc[(pred_df.pred1==pred_label) & (pred_df.prob1>=args['confidence_lower_bound'])]
        sub_label_df = sub_label_df.sort_values('prob1',ascending=False).iloc[0:args['imgs_per_label']]
        pseudo_df = pd.concat((pseudo_df,sub_label_df))

    pseudo_full = pseudo_df.rename(columns={'target':'label'}).copy()
    print(f'Accuracy of {args["imgs_per_label"]} pseudo labels chosen for adapter {(pseudo_full["correct"].sum()) / (len(pseudo_full))}')
    pseudo_full.drop_duplicates(subset='img_path',inplace=True)

    ## Check whether every label has representation in the pseudolabel space
    list_of_classes_without_pseudolabel = set(pred_df.target)-predicted_labels

    if len(list_of_classes_without_pseudolabel)>0:
        print(args['dataset']+' NEEDED EXTRA GUESSES') ## if no 1st choice pseudolabels for a certain category, keep some of the 2nd guesses
        rare_df = pd.DataFrame()
        for rare_label in list_of_classes_without_pseudolabel:
            temp_df = pred_df.copy()
            indices = temp_df.rest_of_pred.apply(lambda x: x.index(rare_label) if rare_label in x else -1).index.to_list()
            order_of_pred = temp_df.rest_of_pred.apply(lambda x: x.index(rare_label) if rare_label in x else -1).values
            for idx in indices:
                if order_of_pred[idx]!=-1:
                    temp_df.loc[idx,'rare_pred'] = temp_df.loc[idx].rest_of_pred[order_of_pred[idx]]
                    temp_df.loc[idx,'rare_pred_prob'] = temp_df.loc[idx].rest_of_pred_probs[order_of_pred[idx]]

            if len(rare_df)>0:
                temp_df = temp_df.loc[~temp_df.img_path.isin(set(rare_df.img_path))]
            rare_df = pd.concat((rare_df,temp_df.dropna(subset=['rare_pred']).sort_values('rare_pred_prob',ascending=False).head(1)))
            rare_df['pred1'] = rare_df['rare_pred']

        pseudo_full = pseudo_full.loc[~pseudo_full.img_path.isin(set(rare_df.img_path))]

        pseudo_full = pd.concat((pseudo_full,rare_df))

    meta_train_replace = meta.loc[meta.img_path.isin(set(pseudo_full.img_path_trimmed))]
    pseudo_full.sort_values('img_path_trimmed',inplace=True)
    pseudo_full['pseudolabel'] = pseudo_full['pred1']

    meta_train_replace.sort_values('img_path',inplace=True)
    ### Replace label and the corresponding category id with the predicted ones(pseudolabels)
    meta_train_replace['label'] = pseudo_full['pseudolabel'].values
    meta_train_replace['category_id'] = meta_train_replace['label'].apply(lambda x: label_to_category[x])

    ### keep val/test as they are for evaluation purposes
    meta_test = meta.loc[~meta.img_set.isin(split_list)].copy()
    ### save file updated with pseudolabels
    meta_new = pd.concat((meta_train_replace,meta_test))
    # reset index
    meta_new.reset_index(inplace=True)
    meta_new.drop(columns=['index'],inplace=True)

    meta_new.to_csv('{}/{}_meta_fast_{}_pseudo_clip_{}shot.csv'.format(data_dir,args['dataset'],clip_model,args['imgs_per_label']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir",type=str,default='data/') 
    parser.add_argument('--dataset', choices=['resisc45','aid','patternnet', 'whurs19','ucm','optimal31', 'mlrsnet',
                                            'oxford_pets','oxford_flowers','dtd','imagenet'], type=str)
    parser.add_argument("--model_subtype",type=str, choices=["ViT-B/32", "ViT-B/16","ViT-L/14", "RN50"],default="RN50", help="exact type of clip pretraining backbone")
    parser.add_argument("--confidence_lower_bound", type=float,help='minimum confidence required for a pseudolabel to be kept',default=0.0)
    parser.add_argument("--imgs_per_label", type=int,help='the amount of pseudolabels to keep for each of the predicted labels (ranked based on their clip confidence, higher first)',default=16)
    # turn the args into a dictionary
    args = vars(parser.parse_args())
    main(args)