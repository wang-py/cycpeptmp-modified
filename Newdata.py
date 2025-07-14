#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import os
import pandas as pd
import numpy as np
from rdkit import Chem

from utils import utils_function
from utils import calculate_descriptors
from utils import generate_conformation
from utils import generate_atom_input
from utils import generate_monomer_input
from utils import generate_peptide_input

import torch
import torch.nn as nn
from model import model_utils
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Torch version: {torch.__version__}')
print(f'Device: {DEVICE}')


# In[3]:


config_path = 'config/CycPeptMP.json'
config = json.load(open(config_path,'r'))


# In[4]:


# Example cyclic peptide drugs, without experimentally determined permeability.
# Anidulafungin, Pasireotide
new_data = pd.read_csv('data/new_data/new_data.csv')

# Check duplicates
old_data = pd.read_csv('data/CycPeptMPDB_Peptide_All.csv', low_memory=False)
for i in range(len(new_data)):
    if utils_function.canonicalize_smiles(new_data.iloc[i]['SMILES']) in old_data['SMILES'].to_list():
        print(f'Your peptide: {i} ({new_data.iloc[i]["ID_org"]}) is already in the database.')


# ### 0. Divide peptide into monomers (substructures)
# + Divides __peptide bond__ and __ester bond__ in the __main chain__ and splits peptide into monomers.
# + The cleaved amide group or O atom of the amide-to-ester substitution was methylated (addition of CH3), and the carboxyl group was converted to an aldehyde (addition of H).
# + __Disulfide bond__ is not included in CycPeptMPDB data, but it may be better to consider it as a division target.
# + __Bonds in side-chain__ are not subject to division to fully represent the side-chain properties.

# In[ ]:





# In[5]:


# Save unique monomers
utils_function.get_unique_monomer(new_data, 'data/new_data/unique_monomer.csv')


# ### 1. Generate different peptide SMILES representations by SMILES enumeration as atom-level data augmentation

# In[6]:


utils_function.enumerate_smiles(new_data, config, 'data/new_data/enum_smiles.csv')


# ### 2. Generate 3D conformations for peptide and monomer

# + Peptide

# In[9]:


# os.mkdir('sdf/new_data/')

df_enu = pd.read_csv('data/new_data/enum_smiles.csv')

# WARNING: If there is too much data, you can manually split it into multiple files for parallel computation.
# For example:
# sub_file_num = 10
# sub_file_len = len(df_enu) // sub_file_num
# for i in range(sub_file_num):
#     df_enu.iloc[i*sub_file_len:(i+1)*sub_file_len].to_csv(f'sdf/new_data/peptide_{i}.csv', index=False)

generate_conformation.generate_peptide_conformation(config, df_enu, 'sdf/new_data/peptide.sdf')


# + Monomer

# In[10]:


df_monomer = pd.read_csv('data/new_data/unique_monomer.csv')
generate_conformation.generate_monomer_conformation(config, df_monomer, 'sdf/new_data/monomer.sdf')


# ### 3. Calculate 2D and 3D descriptors for peptide and monomer

# #### 3.1. RDKit (208 types 2D descriptors)

# In[11]:


# peptide
calculate_descriptors.calc_rdkit_descriptors(new_data['SMILES'].tolist(), 'desc/new_data/peptide_rdkit.csv')


# In[12]:


# monomer
calculate_descriptors.calc_rdkit_descriptors(df_monomer['SMILES'].tolist(), 'desc/new_data/monomer_rdkit.csv')


# #### 3.2. Mordred (1275 types 2D descriptors + 51 types 3D descriptors)

# + 2D

# In[13]:


# peptide
calculate_descriptors.calc_mordred_2Ddescriptors(new_data['SMILES'].tolist(), 'desc/new_data/peptide_mordred_2D.csv')


# In[14]:


# monomer
calculate_descriptors.calc_mordred_2Ddescriptors(df_monomer['SMILES'].tolist(), 'desc/new_data/monomer_mordred_2D.csv')


# + 3D

# In[15]:


# peptide
mols = Chem.SDMolSupplier('sdf/new_data/peptide.sdf')
calculate_descriptors.calc_mordred_3Ddescriptors(mols, 'desc/new_data/peptide_mordred_3D.csv')


# In[16]:


# monomer
mols = Chem.SDMolSupplier('sdf/new_data/monomer.sdf')
calculate_descriptors.calc_mordred_3Ddescriptors(mols, 'desc/new_data/monomer_mordred_3D.csv')


# #### 3.3. MOE (206 types 2D descriptors + 117 types 3D descriptors)
# + CycPeptMP used the commercial software __MOE__ to calculate some of the descriptors.
# + In particular, many of the selected 3D descriptors were computed by MOE.
# + Please manualy calculate these descriptors. I showed __utils/MOE_3D_descriptors.sh__ as an example.
# + For 2D descriptors:
#     + Please wash SMILES and use washed mols for calculation.
#         + for GUI: Molecule -> Wash -> Protonation: Dominant
# + For 3D descriptors:
#     + First, please calculate the charge for the RDKit conformations.
#         + for GUI: Compute -> Molecule -> Partial Charges
#     + 21 MOPAC descriptors of the 3D descriptors were not computed due to computational cost (AM_x, MNDO_, PM3_x)

# #### 3.4. Concatenation files

# In[20]:


calculate_descriptors.merge_descriptors(config, 'desc/new_data/', 'data/new_data/')


# ### 4. Generate input for three sub-models

# In[21]:


folder_path = 'model/input/new_data/'
set_name = 'new'


# + Atom model

# In[22]:


df_enu = pd.read_csv('data/new_data/enum_smiles.csv')
mols = Chem.SDMolSupplier('sdf/new_data/peptide.sdf')

generate_atom_input.generate_atom_input(config, new_data, df_enu, mols, folder_path, set_name)


# + Monomer model

# In[23]:


df_mono_2D = pd.read_csv('desc/new_data/monomer_2D.csv')
df_mono_3D = pd.read_csv('desc/new_data/monomer_3D.csv')

generate_monomer_input.generate_monomer_input(config, new_data, df_mono_2D, df_mono_3D, folder_path, set_name)


# + Peptide model

# In[28]:


df_pep_2D = pd.read_csv('desc/new_data/peptide_2D.csv')
df_pep_3D = pd.read_csv('desc/new_data/peptide_3D.csv')
df_enu = pd.read_csv('data/new_data/enum_smiles.csv')

generate_peptide_input.generate_peptide_input(config, new_data, df_enu, df_pep_2D, df_pep_3D, folder_path, set_name)


# ### 5. Prediction

# In[10]:


MODEL_TYPE = 'Fusion'
# OPTIMIZE: Augmentation times
REPLICA_NUM = 60

# Import input
set_name = 'new'
dataset_new = model_utils.load_dataset('model/input/new_data/', MODEL_TYPE, REPLICA_NUM, set_name)

for _ in dataset_new[0]:
    print(_.shape)


# In[11]:


# Set random seed for reproducibility
seed = config['data']['seed']
model_utils.set_seed(seed)

# Determined hyperparameters
best_trial = config['model']

for cv in range(3):
    # Load trained weights
    model_path = f'weight/{MODEL_TYPE}/{MODEL_TYPE}-{REPLICA_NUM}_cv{cv}.cpt'
    checkpoint = torch.load(model_path)
    model = model_utils.create_model(best_trial, DEVICE, config['model']['use_auxiliary'])
    model_state = checkpoint['model_state_dict']
    model.load_state_dict(model_state)
    model = nn.DataParallel(model)
    model.to(DEVICE)

    # OPTIMIZE: Batch size
    batch_size = len(dataset_new)
    dataloader_now = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size, shuffle=False)
    ids, exps, preds = model_utils.predict_valid(DEVICE, model, dataloader_now, None, istrain=False,
                                                 use_auxiliary=config['model']['use_auxiliary'], gamma_layer=config['model']['gamma_layer'], gamma_subout=config['model']['gamma_subout'])
    now_pred = pd.DataFrame(preds, columns=['pred'])
    now_pred['exp'] = exps
    now_pred['ID'] = ids

    # NOTE: Can save all predicted values of all replicas
    # now_pred.to_csv(f'predicted/{MODEL_TYPE}-{REPLICA_NUM}/{set_name}_cv{cv}_allrep.csv')

    # Take the average of all replicas
    now_pred = now_pred.groupby('ID').mean()
    now_pred.to_csv(f'predicted/new_data/{MODEL_TYPE}-{REPLICA_NUM}/{set_name}_cv{cv}.csv')


# In[12]:


pred_cv0 = pd.read_csv(f'predicted/new_data/{MODEL_TYPE}-{REPLICA_NUM}/{set_name}_cv0.csv')
pred_cv1 = pd.read_csv(f'predicted/new_data/{MODEL_TYPE}-{REPLICA_NUM}/{set_name}_cv1.csv')
pred_cv2 = pd.read_csv(f'predicted/new_data/{MODEL_TYPE}-{REPLICA_NUM}/{set_name}_cv2.csv')

pred_mean = (pred_cv0 + pred_cv1 + pred_cv2) / 3
pred_mean[['ID', 'pred']]


# In[ ]:




