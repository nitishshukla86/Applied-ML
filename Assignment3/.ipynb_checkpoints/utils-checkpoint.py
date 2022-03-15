import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
from copy import deepcopy
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

NUM_WORKERS=16

def plotPCA2D(features,feature_class,name=None):
  x=pd.DataFrame( features.cpu().detach().numpy())
  y=pd.DataFrame(feature_class.cpu().detach().numpy().astype(np.int8))
  pca = PCA(n_components=2) 
  X = pd.DataFrame(StandardScaler().fit_transform(x))
  principalComponents = pca.fit_transform(X)
  principal_df = pd.DataFrame(data = principalComponents
              , columns = ['PC1', 'PC2'])

  centers=[principalComponents[feature_class.cpu().numpy()==i].mean(axis=0).tolist() for i in range(10)]
  plt.scatter(principalComponents[:, 0], principalComponents[:, 1], s= 5, c=y, cmap='Spectral')
  plt.gca().set_aspect('equal', 'datalim')
  plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
  plt.scatter([centers[i][0] for i in range(10)],[centers[i][1] for i in range(10)],marker='*',s=500,c='b')
  plt.title(f'PCA ; feature dim: {X.shape[1]} ;{name}', fontsize=24);
  plt.xlabel('PC1')
  plt.ylabel('PC2')
    
    


@torch.no_grad()
def prepare_data_features(model, dataset):
    # Prepare model
    network = deepcopy(model.vit)
    network.classifier = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats['logits'].detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return data.TensorDataset(feats, labels),feats,labels

def get_smaller_dataset(original_dataset, num_imgs_per_label):
    new_dataset = data.TensorDataset(
        *[t.unflatten(0, (10, -1))[:,:num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors]
    )
    return new_dataset