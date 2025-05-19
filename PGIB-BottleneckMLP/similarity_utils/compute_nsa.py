import torch
from similarity_metrics import NSALoss
from similarity_metrics import LNSA_loss
import numpy as np
from GNSA_adapted import NSALoss_adapted

device = 'cuda:0'

nsa = NSALoss()
lnsa = LNSA_loss(k=5)

nsa_adapted = NSALoss_adapted()

for epoch in range(300):
    emb_dict = torch.load(f'./embeddings/embeddings_epoch_{epoch}.pt')

    space1 = emb_dict['node_embs'].detach().to(device)
    space2 = emb_dict['layer_0'].detach().to(device)
    space3 = emb_dict['layer_1'].detach().to(device)
    space4 = emb_dict['layer_2'].detach().to(device)
    space5 = emb_dict['last_layer'].detach().to(device)

    category1_indices = torch.load(f"./indices_per_batch/category1_indices_epoch_{epoch}.pt")
    category2_indices = torch.load(f"./indices_per_batch/category2_indices_epoch_{epoch}.pt")
    category3_indices = torch.load(f"./indices_per_batch/category3_indices_epoch_{epoch}.pt")

    space1_cat1 = space1[category1_indices]
    space2_cat1 = space2[category1_indices]
    space3_cat1 = space3[category1_indices]
    space4_cat1 = space4[category1_indices]
    space5_cat1 = space5[category1_indices]

    space1_cat2 = space1[category2_indices]
    space2_cat2 = space2[category2_indices]
    space3_cat2 = space3[category2_indices]
    space4_cat2 = space4[category2_indices]
    space5_cat2 = space5[category2_indices]

    space1_cat3 = space1[category3_indices]
    space2_cat3 = space2[category3_indices]
    space3_cat3 = space3[category3_indices]
    space4_cat3 = space4[category3_indices]
    space5_cat3 = space5[category3_indices]

    if space1_cat1.size(0) <= 1:
        print(f"No active nodes in epoch {epoch}.")
        with open(f'./similarity_logs/cat1_train_tmp.txt', 'a') as f:
            print(f"Epoch {epoch}, NSA+LNSA: {None}, {None}, {None}, {None}, LNSA: {None}, {None}, {None}, {None}", file=f)
    else: 
        lnsa_cat1_space12 = lnsa(space1_cat1, space2_cat1)
        lnsa_cat1_space13 = lnsa(space1_cat1, space3_cat1)
        lnsa_cat1_space14 = lnsa(space1_cat1, space4_cat1)
        lnsa_cat1_space15 = lnsa(space1_cat1, space5_cat1)

        nsa_cat1_space12 = nsa(space1_cat1, space2_cat1) + lnsa_cat1_space12
        nsa_cat1_space13 = nsa(space1_cat1, space3_cat1) + lnsa_cat1_space13
        nsa_cat1_space14 = nsa(space1_cat1, space4_cat1) + lnsa_cat1_space14 
        nsa_cat1_space15 = nsa(space1_cat1, space5_cat1) + lnsa_cat1_space15

        nsa_adapted_cat1_space12 = nsa_adapted(space1_cat1, space2_cat1, space1, space2)
        nsa_adapted_cat1_space13 = nsa_adapted(space1_cat1, space3_cat1, space1, space3) 
        nsa_adapted_cat1_space14 = nsa_adapted(space1_cat1, space4_cat1, space1, space4)  
        nsa_adapted_cat1_space15 = nsa_adapted(space1_cat1, space5_cat1, space1, space5)        

        with open(f'./similarity_logs/cat1_train_tmp.txt', 'a') as f:
            print(f"Epoch {epoch}, NSA+LNSA: {nsa_cat1_space12}, {nsa_cat1_space13}, {nsa_cat1_space14}, {nsa_cat1_space15}, LNSA: {lnsa_cat1_space12}, {lnsa_cat1_space13}, {lnsa_cat1_space14}, {lnsa_cat1_space15}", file=f)

        with open(f'./similarity_logs/cat1_nsa_adapted.txt', 'a') as f:
            print(f"Epoch {epoch}, NSA: {nsa_adapted_cat1_space12}, {nsa_adapted_cat1_space13}, {nsa_adapted_cat1_space14}, {nsa_adapted_cat1_space15}", file=f)

    lnsa_cat2_space12 = lnsa(space1_cat2, space2_cat2)
    lnsa_cat2_space13 = lnsa(space1_cat2, space3_cat2)
    lnsa_cat2_space14 = lnsa(space1_cat2, space4_cat2)
    lnsa_cat2_space15 = lnsa(space1_cat2, space5_cat2)

    nsa_cat2_space12 = nsa(space1_cat2, space2_cat2) + lnsa_cat2_space12
    nsa_cat2_space13 = nsa(space1_cat2, space3_cat2) + lnsa_cat2_space13
    nsa_cat2_space14 = nsa(space1_cat2, space4_cat2) + lnsa_cat2_space14
    nsa_cat2_space15 = nsa(space1_cat2, space5_cat2) + lnsa_cat2_space15

    nsa_adapted_cat2_space12 = nsa_adapted(space1_cat2, space2_cat2, space1, space2)
    nsa_adapted_cat2_space13 = nsa_adapted(space1_cat2, space3_cat2, space1, space3) 
    nsa_adapted_cat2_space14 = nsa_adapted(space1_cat2, space4_cat2, space1, space4)  
    nsa_adapted_cat2_space15 = nsa_adapted(space1_cat2, space5_cat2, space1, space5) 

    with open(f'./similarity_logs/cat2_train_tmp.txt', 'a') as f:
        print(f"Epoch {epoch}, NSA+LNSA: {nsa_cat2_space12}, {nsa_cat2_space13}, {nsa_cat2_space14}, {nsa_cat2_space15}, LNSA: {lnsa_cat2_space12}, {lnsa_cat2_space13}, {lnsa_cat2_space14}, {lnsa_cat2_space15}", file=f)

    with open(f'./similarity_logs/cat2_nsa_adapted.txt', 'a') as f:
        print(f"Epoch {epoch}, NSA: {nsa_adapted_cat2_space12}, {nsa_adapted_cat2_space13}, {nsa_adapted_cat2_space14}, {nsa_adapted_cat2_space15}", file=f)

    lnsa_cat3_space12 = lnsa(space1_cat3, space2_cat3)
    lnsa_cat3_space13 = lnsa(space1_cat3, space3_cat3)
    lnsa_cat3_space14 = lnsa(space1_cat3, space4_cat3)
    lnsa_cat3_space15 = lnsa(space1_cat3, space5_cat3)

    nsa_cat3_space12 = nsa(space1_cat3, space2_cat3) + lnsa_cat3_space12
    nsa_cat3_space13 = nsa(space1_cat3, space3_cat3) + lnsa_cat3_space13
    nsa_cat3_space14 = nsa(space1_cat3, space4_cat3) + lnsa_cat3_space14
    nsa_cat3_space15 = nsa(space1_cat3, space5_cat3) + lnsa_cat3_space15

    nsa_adapted_cat3_space12 = nsa_adapted(space1_cat3, space2_cat3, space1, space2)
    nsa_adapted_cat3_space13 = nsa_adapted(space1_cat3, space3_cat3, space1, space3) 
    nsa_adapted_cat3_space14 = nsa_adapted(space1_cat3, space4_cat3, space1, space4)  
    nsa_adapted_cat3_space15 = nsa_adapted(space1_cat3, space5_cat3, space1, space5) 

    with open(f'./similarity_logs/cat3_train_tmp.txt', 'a') as f:
        print(f"Epoch {epoch}, NSA+LNSA: {nsa_cat3_space12}, {nsa_cat3_space13}, {nsa_cat3_space14}, {nsa_cat3_space15}, LNSA: {lnsa_cat3_space12}, {lnsa_cat3_space13}, {lnsa_cat3_space14}, {lnsa_cat3_space15}", file=f)

    with open(f'./similarity_logs/cat3_nsa_adapted.txt', 'a') as f:
        print(f"Epoch {epoch}, NSA: {nsa_adapted_cat3_space12}, {nsa_adapted_cat3_space13}, {nsa_adapted_cat3_space14}, {nsa_adapted_cat3_space15}", file=f)
