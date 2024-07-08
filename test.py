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

        for img_path, conf in zip(img_paths_class[:K], conf_class):
            if '/data/' in img_path:
                img_path = './data/' + img_path.split('/data/')[1]
            predict_label_dict[img_path] = id
            predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict