# TP4 - INF8225

**Repo NON-OFFICIEL de l'implémentation du modèle FISR**

Article de l'auteur: [article](https://arxiv.org/abs/1912.07213).

**Reference**:  
> Soo Ye Kim*, Jihyong Oh*, and Munchurl Kim, "FISR: Deep Joint Frame Interpolation and Super-Resolution with a Multi-scale Temporal Loss", *AAAI Conference on Artificial Intelligence*, 2020. (* *equal contribution*)

**BibTeX**
```bibtex
@inproceedings{kim2020fisr,
  title={FISR: Deep Joint Frame Interpolation and Super-Resolution with a Multi-scale Temporal Loss},
  author={Kim, Soo Ye and Oh, Jihyong and Kim, Munchurl},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2020}
}
```

### Requirements
Notre code est implémenté sur les versions suivantes:  
* Python 3.8
* Tensorflow 2.10 
* CUDA 11.2.2  
* cuDNN 8.1.1  
* NVIDIA RTX 4080 GPU
* Windows 11

## Code de test
### Quick Start
1.Télécharger le code source à un dossier quelconque **\<source_path\>**.

2. Télécharger la base de donnée de test des auteurs à l'adresse suivante : [Ce lien]( https://www.dropbox.com/s/101g9kdobgwl8x6/test.zip?dl=0) et unzip le dossier 'test' dans **\<source_path\>/data/test**, vous pourrez alors obtenir une base de donnée en entré (LR LFR), des données de flux, des données déformées et des données en sorties (HR HFR) placés dans **\<source_path\>/data/test/LR_LFR**, **\<source_path\>/data/test/flow** , **\<source_path\>/data/test/warped**  et **\<source_path\>/data/test/HR_HFR**, respectivement. 
```
FISR
└── data
   └── test
       ├── flow
           ├── LR_Surfing_SlamDunk_test_ss1.flo
       ├── HR_HFR
           ├── HR_vid_1_fr_07171_seq_2.png
           ├── HR_vid_1_fr_07171_seq_3.png
           └── ...
       ├── LR_LFR
           ├── LR_vid_1_fr_07171_seq_1.png 
           ├── LR_vid_1_fr_07171_seq_3.png
           └── ...
       ├── warped
           ├── LR_Surfing_SlamDunk_test_ss1_warp.mat  
```
3. Télécharger des poids préentrainés par les auteurs [Ce lien](https://www.dropbox.com/s/hfzzddfocmmazso/FISRnet_exp1.zip?dl=0) et placés les dans **\<source_path\>/checkpoint_dir/FISRnet_exp1**.
```
FISR
└── checkpoint_dir
   └── FISRnet_exp1
       ├── checkpoint
       ├── FISRnet-122000.data-00000-of-00001
       ├── FISRnet-122000.index
       ├── FISRnet-122000.meta
           
```
4. Rouler **main.py** avec les arguments suivants: 

**(i) Pour tester sur les données 4K des auteurs : test dataset input:**  

```bash
python main.py --phase 'test' --exp_num 1 --test_data_path './data/test/LR_LFR' --test_flow_data_path './data/test/flow/LR_Surfing_SlamDunk_test_ss1.flo' --test_warped_data_path './data/test/warped/LR_Surfing_SlamDunk_test_ss1_warp.mat' --test_label_path './data/test/HR_HFR'
```

## Code d'entrainement
### Quick Start
Téléchargez l'ensemble de données d'entraînement des auteurs depuis [ce lien]( https://www.dropbox.com/s/n71hzqis6hpggcs/train.zip?dl=0) et décompressez le dossier 'train' dans <source_path>/data/train. Vous obtiendrez alors un ensemble de données d'entrée (LR LFR), deux données de flux (stride 1 & 2), deux données déformées (stride 1 & 2) et un ensemble de données de sortie (HR HFR) placés respectivement dans <source_path>/data/train/LR_LFR, <source_path>/data/train/flow, <source_path>/data/train/warped et <source_path>/data/train/HR_HFR.
 ```
FISR
└── data
   └── train
       ├── flow
           ├── LR_Surfing_SlamDunk_5seq_ss1.flo
           ├── LR_Surfing_SlamDunk_5seq_ss2.flo
       ├── HR_HFR
           ├── HR_Surfing_SlamDunk_5seq.mat
       ├── LR_LFR
           ├── LR_Surfing_SlamDunk_5seq.mat
       ├── warped
           ├── LR_Surfing_SlamDunk_5seq_ss1_warp.mat  
           ├── LR_Surfing_SlamDunk_5seq_ss2_warp.mat
```
3. Run **main.py** avec les options suivantes  
```bash
python main.py --phase 'train' --exp_num 11 --train_data_path './data/train/LR_LFR/LR_Surfing_SlamDunk_5seq.mat' --train_flow_data_path './data/train/flow/LR_Surfing_SlamDunk_5seq_ss1.flo' --train_flow_ss2_data_path './data/train/flow/LR_Surfing_SlamDunk_5seq_ss2.flo' --train_warped_data_path './data/train/warped/LR_Surfing_SlamDunk_5seq_ss1_warp.mat' --train_wapred_ss2_data_path './data/train/warped/LR_Surfing_SlamDunk_5seq_ss2_warp.mat' --train_label_path './data/train/HR_HFR/HR_Surfing_SlamDunk_5seq.mat'  --epoch 5
`--exp_num` devrait être ajusté au besoin.



