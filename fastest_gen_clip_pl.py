import ast
import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, image_paths, targets, preprocess, data_dir):
        self.image_paths = image_paths
        self.targets = targets
        self.preprocess = preprocess
        self.data_dir = data_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(image_path)
        image = self.preprocess(image)
        target = self.targets[idx]
        return image, target

def select_top_k_similarity_per_class(outputs, img_paths, K=1, image_features=None, is_softmax=True):
    # print(outputs.shape)
    if is_softmax:
        outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0] # 获得每行的类别标签

    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]

    predict_label_dict = {}
    predict_conf_dict = {}
    from tqdm import tqdm
    for id in tqdm(list(set(ids.tolist()))): # 标签去重
        index = np.where(ids==id)
        conf_class = output_m_max[index] # 置信度
        output_class = output_ori[index]
        img_paths_class = img_paths[index] # 每个类别的路径

        if image_features is not None:
            img_features = image_features[index]
            if K >= 0:
                for img_path, img_feature, conf, logit in zip(img_paths_class[:K], img_features[:K], conf_class[:K], output_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
            else:
                for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
        else:
            if K >= 0:
                for img_path, conf in zip(img_paths_class[:K], conf_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
            else:
                for img_path, conf in zip(img_paths_class, conf_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = '{}{}/'.format(args['root_data_dir'], args['dataset'])
    metadir = data_dir + '{}_meta.csv'.format(args['dataset'])
    meta = pd.read_csv(metadir, index_col=0)
    meta['category_id'] = meta.category_id.astype(int)
    split_list = ['train']
    sampled = meta.loc[meta.img_set.isin(split_list)]

    ### Added for ZS classifier
    label_df = meta[['category_id', 'label']]
    label_df = label_df.drop_duplicates(subset=['label'])
    cls2id = dict(zip(label_df.label, label_df.category_id))

    label_mapper = meta.drop_duplicates(subset=['label'])['label'].reset_index()
    label_mapper.drop(columns=['index'], inplace=True)
    label_mapper['clip_label'] = label_mapper.label.apply(lambda x: x.replace('_', ' '))
    if args['dataset'] in ['aid', 'eurosat', 'patternnet', 'resisc45', 'ucm', 'whurs19', 'mlrsnet', 'optimal31']:
        text_pr = "a centered satellite image of "
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: '{}{}.'.format(text_pr, x))
    elif args['dataset'] == "imagenet":
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: 'a photo of a {}.'.format(x))
    else:
        print('No dataset found')

    clip_to_target = dict(label_mapper[['clip_label', 'label']].values)
    grouped_label_to_clip_ids = pd.DataFrame(clip_to_target.values()).reset_index().rename(columns={'index': 'clip_id', 0: 'dataset_label'}).groupby('dataset_label').clip_id.apply(list)
    N = sampled.shape[0]

    datasets_names = ['eurosat', 'ucm', 'aid', 'patternnet', 'resisc45', 'whurs19', 'mlrsnet', 'optimal31', 'caltech-101', 'oxford_pets', 'oxford_flowers', 'imagenet']
    map_datasets_name = ['EuroSAT', 'UCM', 'AID', 'PatternNet', 'RESISC45', 'WHURS19', 'MLRSNet', 'Optimal31', 'Caltech101', 'OxfordPets', 'OxfordFlowers', 'ImageNet']
    dataset = map_datasets_name[datasets_names.index(args['dataset'])]

    ### Load CLIP model
    model, preprocess = clip.load(args['model_subtype'], device=device, jit=False)
    classifier_weights = torch.load(f'/l/users/sanoojan.baliah/Felix/RS_zero_shot/embeddings/lafter_{dataset}_ZSembeddings.pt').squeeze()
    classifier_weights = classifier_weights[list(cls2id.values())].to(device)

    image_links = list(sampled.img_path)
    targets = list(sampled.label)
    clip_labels = list(clip_to_target.keys())

    pred_df = pd.DataFrame()
    correct_list = []
    batch_size = 64  # Adjusted for better performance

    dataset = CustomDataset(image_links, targets, preprocess, data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    for i, (batch_images, batch_targets) in enumerate(dataloader):
        # print(f'{i * batch_size / N * 100:.2f}% done')
        batch_images = batch_images.to(device)
        batch_targets = np.array(batch_targets)

        with torch.no_grad():
            logits_per_image, _ = model.forward_with_text_desc(batch_images, classifier_weights)
            probs = logits_per_image.softmax(dim=-1)

        for b_index in range(batch_images.size(0)):
            prob = probs[b_index].cpu()
            label_probs = grouped_label_to_clip_ids.apply(lambda x: torch.index_select(prob, 0, torch.tensor(x)).sum().item())
            sorted_probs = label_probs.sort_values(ascending=False)

            if len(clip_labels) > 2:
                prob1, prob2, prob3 = sorted_probs[0:3].values
                pred1, pred2, pred3 = sorted_probs[0:3].index
            else:
                prob1, prob2 = sorted_probs[0:2].values
                pred1, pred2 = sorted_probs[0:2].index
                prob3 = np.nan
                pred3 = np.nan

            rest_of_the_predictions = sorted_probs[1:].index
            rest_of_the_prediction_probs = sorted_probs[1:].values

            pred_correct = batch_targets[b_index] == pred1
            correct_list.append(pred_correct)

            df_temp = pd.DataFrame({
                'dataset': args['dataset'],
                'img_path': image_links[i * batch_size + b_index],
                'pred1': [pred1],
                'pred2': [pred2],
                'pred3': [pred3],
                'prob1': [prob1],
                'prob2': [prob2],
                'prob3': [prob3],
                'rest_of_pred': [list(rest_of_the_predictions)],
                'rest_of_pred_probs': [list(rest_of_the_prediction_probs)],
                'correct': pred_correct,
                'target': [batch_targets[b_index]]
            })

            pred_df = pd.concat([pred_df, df_temp], ignore_index=True)

    acc_mean = 100 * np.mean(correct_list)
    print(f'Accuracy mean: {acc_mean:.2f}%')

    clip_model = 'clip_' + args['model_subtype'].replace('/', '').replace('-', '_')
    save_line = "{},{}, {} Shot, Test acc stat: {:.2f} ()\n".format(args['dataset'], clip_model, 0, acc_mean, '')
    print(save_line, flush=True)

    pred_df['img_path_trimmed'] = pred_df['img_path'].apply(lambda x: x.replace(data_dir, ''))
    label_to_category = dict(meta[['label', 'category_id']].drop_duplicates().values)

    predicted_labels = set(pred_df.pred1)
    sub_df = pred_df.copy()
    sub_df = sub_df.rename(columns={'target': 'label'})

    sub_df.to_csv('{}/{}_meta_faster_sub_df_{}_clip_{}shot.csv'.format(data_dir, args['dataset'], clip_model, 0))

    pseudo_df = pd.DataFrame()
    for pred_label in predicted_labels:
        sub_label_df = pred_df.loc[(pred_df.pred1 == pred_label) & (pred_df.prob1 >= args['confidence_lower_bound'])]
        sub_label_df = sub_label_df.sort_values('prob1', ascending=False).iloc[0:args['imgs_per_label']]
        pseudo_df = pd.concat((pseudo_df, sub_label_df))

    pseudo_full = pseudo_df.rename(columns={'target': 'label'}).copy()
    print(f'Accuracy of {args["imgs_per_label"]} pseudo labels chosen for adapter {(pseudo_full["correct"].sum()) / (len(pseudo_full))}')
    pseudo_full.drop_duplicates(subset='img_path', inplace=True)

    list_of_classes_without_pseudolabel = set(pred_df.target) - predicted_labels

    if len(list_of_classes_without_pseudolabel) > 0:
        print(args['dataset'] + ' NEEDED EXTRA GUESSES')
        rare_df = pd.DataFrame()
        for rare_label in list_of_classes_without_pseudolabel:
            temp_df = pred_df.copy()
            indices = temp_df.rest_of_pred.apply(lambda x: x.index(rare_label) if rare_label in x else -1).index.to_list()
            order_of_pred = temp_df.rest_of_pred.apply(lambda x: x.index(rare_label) if rare_label in x else -1).values
            for idx in indices:
                if order_of_pred[idx] != -1:
                    temp_df.loc[idx, 'rare_pred'] = temp_df.loc[idx].rest_of_pred[order_of_pred[idx]]
                    temp_df.loc[idx, 'rare_pred_prob'] = temp_df.loc[idx].rest_of_pred_probs[order_of_pred[idx]]

            if len(rare_df) > 0:
                temp_df = temp_df.loc[~temp_df.img_path.isin(set(rare_df.img_path))]
            rare_df = pd.concat((rare_df, temp_df.dropna(subset=['rare_pred']).sort_values('rare_pred_prob', ascending=False).head(1)))
            rare_df['pred1'] = rare_df['rare_pred']

        pseudo_full = pseudo_full.loc[~pseudo_full.img_path.isin(set(rare_df.img_path))]

        pseudo_full = pd.concat((pseudo_full, rare_df))

    meta_train_replace = meta.loc[meta.img_path.isin(set(pseudo_full.img_path_trimmed))]
    pseudo_full.sort_values('img_path_trimmed', inplace=True)
    pseudo_full['pseudolabel'] = pseudo_full['pred1']

    meta_train_replace.sort_values('img_path', inplace=True)
    meta_train_replace['label'] = pseudo_full['pseudolabel'].values
    meta_train_replace['category_id'] = meta_train_replace['label'].apply(lambda x: label_to_category[x])

    meta_test = meta.loc[~meta.img_set.isin(split_list)].copy()

    meta_new = pd.concat((meta_train_replace, meta_test))
    meta_new.reset_index(inplace=True)
    meta_new.drop(columns=['index'], inplace=True)

    meta_new.to_csv('{}/{}_meta_fast_{}_pseudo_clip_{}shot.csv'.format(data_dir, args['dataset'], clip_model, args['imgs_per_label']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, default='data/')
    parser.add_argument('--dataset', choices=['resisc45', 'aid', 'patternnet', 'whurs19', 'ucm', 'optimal31', 'mlrsnet',
                                              'oxford_pets', 'oxford_flowers', 'dtd', 'imagenet'], type=str)
    parser.add_argument("--model_subtype", type=str, choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"],
                        default="RN50", help="exact type of clip pretraining backbone")
    parser.add_argument("--confidence_lower_bound", type=float,
                        help='minimum confidence required for a pseudolabel to be kept', default=0.0)
    parser.add_argument("--imgs_per_label", type=int,
                        help='the amount of pseudolabels to keep for each of the predicted labels '
                             '(ranked based on their clip confidence, higher first)', default=16)
    args = vars(parser.parse_args())

    main(args)
