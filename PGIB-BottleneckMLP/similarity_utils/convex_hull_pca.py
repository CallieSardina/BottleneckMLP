# import numpy as np
# from scipy.spatial import ConvexHull
# import torch
# from similarity_metrics import NSALoss
# from similarity_metrics import LNSA_loss
# import numpy as np
# from GNSA_adapted import NSALoss_adapted
# from sklearn.decomposition import PCA

import numpy as np
from scipy.spatial import ConvexHull
import torch
from sklearn.decomposition import PCA

device = 'cpu'

def compute_volume_pca2d(tensor):
    if tensor.shape[0] < 4:
        return None
    try:
        reduced = PCA(n_components=3).fit_transform(tensor)
        hull = ConvexHull(reduced)
        return hull.volume
    except Exception:
        return None

for epoch in range(300):
    emb_dict = torch.load(f'./embeddings/embeddings_epoch_{epoch}.pt')
    category1_indices = torch.load(f"./indices_per_batch/category1_indices_epoch_{epoch}.pt")
    category2_indices = torch.load(f"./indices_per_batch/category2_indices_epoch_{epoch}.pt")
    category3_indices = torch.load(f"./indices_per_batch/category3_indices_epoch_{epoch}.pt")

    spaces = ['node_embs', 'layer_0', 'layer_1', 'layer_2', 'last_layer']

    max_len = category1_indices.shape[0]
    indices = torch.randperm(category2_indices.size(0))[:max_len]
    subset_cat2 = category2_indices[indices]
    indices = torch.randperm(category3_indices.size(0))[:max_len]
    subset_cat3 = category3_indices[indices]
    categories = {
        "cat1": category1_indices,
        "cat2": subset_cat2,
        "cat3": subset_cat3
    }


    for cat_name, indices in categories.items():
        volumes = []
        for layer in spaces:
            emb = emb_dict[layer][indices].detach().cpu().numpy()
            vol = compute_volume_pca2d(emb)
            volumes.append(vol)

        with open(f'./similarity_logs/{cat_name}_conv_hull_volume_pca.txt', 'a') as f:
            print(f"Epoch {epoch}, Vol: {', '.join(str(v) for v in volumes)}", file=f)

# device = 'cpu'

# for epoch in range(300):
#     emb_dict = torch.load(f'./embeddings/embeddings_epoch_{epoch}.pt')

#     space1 = emb_dict['node_embs'].detach().to(device)
#     space2 = emb_dict['layer_0'].detach().to(device)
#     space3 = emb_dict['layer_1'].detach().to(device)
#     space4 = emb_dict['layer_2'].detach().to(device)
#     space5 = emb_dict['last_layer'].detach().to(device)

#     category1_indices = torch.load(f"./indices_per_batch/category1_indices_epoch_{epoch}.pt")
#     category2_indices = torch.load(f"./indices_per_batch/category2_indices_epoch_{epoch}.pt")
#     category3_indices = torch.load(f"./indices_per_batch/category3_indices_epoch_{epoch}.pt")

#     space1_cat1 = space1[category1_indices]
#     space2_cat1 = space2[category1_indices]
#     space3_cat1 = space3[category1_indices]
#     space4_cat1 = space4[category1_indices]
#     space5_cat1 = space5[category1_indices]

#     space1_cat2 = space1[category2_indices]
#     space2_cat2 = space2[category2_indices]
#     space3_cat2 = space3[category2_indices]
#     space4_cat2 = space4[category2_indices]
#     space5_cat2 = space5[category2_indices]

#     space1_cat3 = space1[category3_indices]
#     space2_cat3 = space2[category3_indices]
#     space3_cat3 = space3[category3_indices]
#     space4_cat3 = space4[category3_indices]
#     space5_cat3 = space5[category3_indices]

#     space1_cat1 = PCA(n_components=2).fit_transform(space1_cat1)
#     space2_cat1 = PCA(n_components=2).fit_transform(space2_cat1)
#     space3_cat1 = PCA(n_components=2).fit_transform(space3_cat1)
#     space4_cat1 = PCA(n_components=2).fit_transform(space4_cat1)
#     space5_cat1 = PCA(n_components=2).fit_transform(space5_cat1)

#     space1_cat1 = PCA(n_components=2).fit_transform(space1_cat2)
#     space2_cat2 = PCA(n_components=2).fit_transform(space2_cat2)
#     space3_cat2 = PCA(n_components=2).fit_transform(space3_cat2)
#     space4_cat2 = PCA(n_components=2).fit_transform(space4_cat2)
#     space5_cat2 = PCA(n_components=2).fit_transform(space5_cat2)

#     space1_cat1 = PCA(n_components=2).fit_transform(space1_cat3)
#     space2_cat3 = PCA(n_components=2).fit_transform(space2_cat3)
#     space3_cat3 = PCA(n_components=2).fit_transform(space3_cat3)
#     space4_cat3 = PCA(n_components=2).fit_transform(space4_cat3)
#     space5_cat3 = PCA(n_components=2).fit_transform(space5_cat3)


#     if len(space1_cat1) <= 1:
#         print(f"No active nodes in epoch {epoch}.")
#         with open(f'./similarity_logs/cat1_train_tmp.txt', 'a') as f:
#             print(f"Epoch {epoch}, Vol: {None}, {None}, {None}, {None}, {None}", file=f)
#     else: 
#         hull_11 = ConvexHull(space1_cat1)
#         hull_12 = ConvexHull(space2_cat1)
#         hull_13 = ConvexHull(space3_cat1)
#         hull_14 = ConvexHull(space5_cat1)
#         hull_15 = ConvexHull(space4_cat1)

#         volume_11 = hull_11.volume
#         volume_12 = hull_12.volume
#         volume_13 = hull_13.volume 
#         volume_14 = hull_14.volume 
#         volume_15 = hull_15.volume    

#         with open(f'./similarity_logs/cat1_conv_hull_volume.txt', 'a') as f:
#             print(f"Epoch {epoch}, Vol: {volume_11}, {volume_12}, {volume_13}, {volume_14}, {volume_15}", file=f)

#     hull_21 = ConvexHull(space1_cat2)
#     hull_22 = ConvexHull(space2_cat2)
#     hull_23 = ConvexHull(space3_cat2)
#     hull_24 = ConvexHull(space4_cat2)
#     hull_25 = ConvexHull(space5_cat2)

#     volume_21 = hull_21.volume
#     volume_22 = hull_22.volume
#     volume_23 = hull_23.volume  
#     volume_24 = hull_24.volume
#     volume_25 = hull_25.volume  

#     with open(f'./similarity_logs/cat2_conv_hull_volume.txt', 'a') as f:
#         print(f"Epoch {epoch}, Vol: {volume_21}, {volume_22}, {volume_23}, {volume_24}, {volume_25}", file=f)

#     hull_31 = ConvexHull(space1_cat3)
#     hull_32 = ConvexHull(space2_cat3)
#     hull_33 = ConvexHull(space3_cat3)
#     hull_34 = ConvexHull(space4_cat3)
#     hull_35 = ConvexHull(space5_cat3)

#     volume_31 = hull_31.volume
#     volume_32 = hull_32.volume
#     volume_33 = hull_33.volume  
#     volume_34 = hull_34.volume
#     volume_35 = hull_35.volume  

#     with open(f'./similarity_logs/cat3_conv_hull_volume.txt', 'a') as f:
#         print(f"Epoch {epoch}, Vol: {volume_31}, {volume_32}, {volume_33}, {volume_34}, {volume_35}", file=f)

