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
from torchvision import transforms
from torch.nn import DataParallel
import warnings
warnings.filterwarnings("ignore")
import time


class CustomClip(torch.nn.Module):
    def __init__(self, model, classifier_weights):
        super(CustomClip, self).__init__()
        self.model = model
        self.classifier_weights = torch.nn.Parameter(classifier_weights, requires_grad=False)

    def forward(self, x):
        x = self.model.encode_image(x)
        x = x / x.norm(dim=-1, keepdim=True)
        text_desc_emb = self.classifier_weights.half()

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * x @ text_desc_emb.t()
        
        return logits, None

class CustomDataset(Dataset):
    def __init__(self, image_paths, targets, preprocess, data_dir, label_to_idx):
        self.image_paths = image_paths
        self.targets = targets
        self.preprocess = preprocess
        self.data_dir = data_dir
        self.label_to_idx = label_to_idx
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Example normalization
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(image_path)
        image = self.preprocess(image)
        target = self.label_to_idx[self.targets[idx]]
        return image, target

def main(args):
    start_time = time.time()
    # device = "cuda:5" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda_devices']  # Set visible GPUs
    data_dir = os.path.join(args['root_data_dir'], args['dataset'])
    metadir = os.path.join(data_dir, '{}_meta.csv'.format(args['dataset']))
    meta = pd.read_csv(metadir, index_col=0)
    meta['category_id'] = meta.category_id.astype(int)
    
    split_list = ['train']
    sampled = meta.loc[meta.img_set.isin(split_list)]

    # Extract category_id and label mappings
    label_df = meta[['category_id', 'label']].drop_duplicates(subset=['label'])
    cls2id = dict(zip(label_df.label, label_df.category_id))

    # Prepare label mappings for CLIP
    label_mapper = meta.drop_duplicates(subset=['label'])[['label']]
    label_mapper['clip_label'] = label_mapper.label.apply(lambda x: x.replace('_', ' '))
    
    dataset_phrases = {
        'aid': 'a centered satellite image of ',
        'eurosat': 'a centered satellite image of ',
        'patternnet': 'a centered satellite image of ',
        'resisc45': 'a centered satellite image of ',
        'ucm': 'a centered satellite image of ',
        'whurs19': 'a centered satellite image of ',
        'mlrsnet': 'a centered satellite image of ',
        'optimal31': 'a centered satellite image of ',
        'imagenet': 'a photo of a {}.'
    }

    if args['dataset'] in dataset_phrases:
        prefix = dataset_phrases[args['dataset']]
        label_mapper['clip_label'] = label_mapper['clip_label'].apply(lambda x: f'{prefix}{x}.' if '{}' in prefix else f'{prefix}{x}')
    else:
        print('No dataset found')
        return

    clip_to_target = dict(label_mapper[['clip_label', 'label']].values)
    grouped_label_to_clip_ids = pd.DataFrame(clip_to_target.values()).reset_index().rename(
        columns={'index': 'clip_id', 0: 'dataset_label'}).groupby('dataset_label').clip_id.apply(list)
    
    N = sampled.shape[0]

    datasets_names = ['eurosat', 'ucm', 'aid', 'patternnet', 'resisc45', 'whurs19', 'mlrsnet', 'optimal31',
                      'caltech-101', 'oxford_pets', 'oxford_flowers', 'imagenet']
    map_datasets_name = ['EuroSAT', 'UCM', 'AID', 'PatternNet', 'RESISC45', 'WHURS19', 'MLRSNet', 'Optimal31',
                         'Caltech101', 'OxfordPets', 'OxfordFlowers', 'ImageNet']
    dataset_name = args['dataset']
    dataset_idx = datasets_names.index(dataset_name)
    dataset = map_datasets_name[dataset_idx]

    # Load CLIP model and classifier weights
    model, preprocess = clip.load(args['model_subtype'], device=device, jit=False)
    classifier_weights = torch.load(f'embeddings/lafter_{dataset}_ZSembeddings.pt').squeeze()
    classifier_weights = classifier_weights[list(cls2id.values())].to(device)

    image_links = list(sampled.img_path)
    targets = list(sampled.label)

    # Create label_to_idx mapping
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(targets)))}

    batch_size = 27000  # Adjusted for better performance

    dataset = CustomDataset(image_links, targets, preprocess, data_dir, label_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    model2 = CustomClip(model, classifier_weights)

    # Make the model data parallel if multiple GPUs are available
    # if torch.cuda.device_count() > 1:
    #     available_gpus = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
    #     model = torch.nn.DataParallel(model2, device_ids=available_gpus)
    #     print('Using DataParallel')
    #     print('Number of GPUs: ', torch.cuda.device_count())
    #     print('CUDA Devices ', os.environ["CUDA_VISIBLE_DEVICES"])
    model = model.to(device)

    model.eval()
    all_probs = []
    all_targets = []

    # Process batches using DataLoader
    for i, (batch_images, batch_targets) in enumerate(dataloader):
        print(f'{i * batch_size / N * 100:.2f}% done')
        batch_images = batch_images.to(device)
        batch_targets = batch_targets.to(device)

        with torch.no_grad():
            logits_per_image, _ = model2(batch_images)
            probs = logits_per_image.softmax(dim=-1)

        all_probs.append(probs.cpu())
        all_targets.append(batch_targets.cpu())
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    # Convert grouped_label_to_clip_ids to a list of tensors
    grouped_label_to_clip_ids_tensors = [torch.tensor(ids) for ids in grouped_label_to_clip_ids]

    # Vectorized calculation of predictions and correctness
    label_probs = torch.stack([
        torch.index_select(all_probs, 1, ids).sum(dim=1) for ids in grouped_label_to_clip_ids_tensors
    ], dim=1)
    sorted_probs, sorted_indices = label_probs.sort(dim=1, descending=True)
    pred1 = sorted_indices[:, 0]
    prob1 = sorted_probs[:, 0]

    pred_correct = all_targets == pred1
    correct_list = pred_correct.tolist()

    results = [
        {
            'img_path': image_links[i],
            'pred1': pred1[i].item(),
            'prob1': prob1[i].item(),
            'correct': pred_correct[i].item(),
            'target': all_targets[i].item()
        } for i in range(len(all_probs))
    ]

    pred_df = pd.DataFrame(results)
    acc_mean = 100 * np.mean(correct_list)
    print(f'Accuracy mean: {acc_mean:.2f}%')

    clip_model = args['model_subtype'].replace('/', '_')
    save_line = "{},{}, 0 Shot, Test acc stat: {:.2f} ()\n".format(args['dataset'], clip_model, acc_mean)
    print(save_line, flush=True)

    pred_df['img_path_trimmed'] = pred_df['img_path'].apply(lambda x: x.replace(data_dir, ''))
    label_to_category = dict(meta[['label', 'category_id']].drop_duplicates().values)
    predicted_labels = set(pred_df.pred1)

    pseudo_df = pd.DataFrame()
    for pred_label in predicted_labels:
        sub_label_df = pred_df.loc[(pred_df.pred1 == pred_label) & (pred_df.prob1 >= args['confidence_lower_bound'])]
        sub_label_df = sub_label_df.sort_values('prob1', ascending=False).iloc[:args['imgs_per_label']]
        pseudo_df = pd.concat((pseudo_df, sub_label_df))

    pseudo_full = pseudo_df.rename(columns={'target': 'label'}).copy()
    print(f'Accuracy of {args["imgs_per_label"]} pseudo labels chosen for adapter {(pseudo_full["correct"].sum()) / len(pseudo_full)}')
    pseudo_full.drop_duplicates(subset='img_path', inplace=True)

    list_of_classes_without_pseudolabel = len(set(pred_df.target)) - len(predicted_labels)

    if list_of_classes_without_pseudolabel > 0:
        print(args['dataset'] + ' NEEDED EXTRA GUESSES')
        raise NotImplementedError

    meta_train_replace = meta.loc[meta.img_path.isin(set(pseudo_full.img_path_trimmed))]
    pseudo_full.sort_values('img_path_trimmed', inplace=True)
    pseudo_full['pseudolabel'] = pseudo_full['pred1']

    meta_train_replace.sort_values('img_path', inplace=True)

    meta_train_replace['label'] = pseudo_full['pseudolabel'].apply(lambda x: list(label_to_category.keys())[list(label_to_category.values()).index(x)])
    meta_train_replace['category_id'] = pseudo_full['pseudolabel']

    meta_test = meta.loc[~meta.img_set.isin(split_list)].copy()

    meta_new = pd.concat((meta_train_replace, meta_test)).reset_index(drop=True)
    meta_new.to_csv('{}/{}_meta_{}_pseudo_{}shot.csv'.format(data_dir, args['dataset'], clip_model, args['imgs_per_label']), index=False)

    end_time = time.time()
    print(f'Time taken: {end_time - start_time:.2f} seconds')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_data_dir", type=str, default='data/')
    parser.add_argument('--dataset', default='eurosat', choices=['resisc45', 'aid', 'patternnet', 'whurs19', 'ucm', 'optimal31', 'mlrsnet', 'eurosat',
                                              'oxford_pets', 'oxford_flowers', 'dtd', 'imagenet'], type=str)
    parser.add_argument("--model_subtype", type=str, choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"],
                        default="ViT-B/32", help="exact type of clip pretraining backbone")
    parser.add_argument("--confidence_lower_bound", type=float,
                        help='minimum confidence required for a pseudolabel to be kept', default=0.0)
    parser.add_argument("--imgs_per_label", type=int,
                        help='the amount of pseudolabels to keep for each of the predicted labels '
                             '(ranked based on their clip confidence, higher first)', default=16)
    # parser.add_argument("--cuda_devices", type=str, default='5', help='comma separated list of CUDA device ids to use')
    args = vars(parser.parse_args())

    main(args)
