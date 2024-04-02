# *Review of current SOTA Domain Generalization methods*

<div align="center">
    <span class="author-block">
    <a href="https://mluniproject.github.io/" target="_blank">Niklas Baier</a><sup>1,2*</sup>,</span>
   
   
</div>



-----------------

### Abstract

We evaluate several novel DG approaches and see whether we could reproduce the results that were presented by the authors. We used the TerraIncognita Dataset for Verification. As a codebase we used UniDG. We provide a overview of our results.


## Our Results

Here is an overview of the reported Results 

| Algorithm       |                Dataset                |  Backbone  |   Reported_Results |Our_Results               
| ---------------- | :-----------------------------------: |  :-------: | :----------: | :---------:|
| ERM+UniDG       | TerraInc | ResNet-50 |52.9 | 55.0 | 
| ERM+UniDG       | TerraInc | ConvNext |69.6 | 64.52 | 
| ERM        | TerraInc |  ConvNext | - | 58.3 | 
| ERM        | TerraInc |  ResNet-50 | 47.2 | 45.15 | 
| LISA (within one Domain)        | TerraInc |  ResNet-50 | - | 17.15 | 
| LISA (within one Domain)        | TerraInc |  ConvNext | - | 47.35 | 
| SIMPLE      | TerraInc|Ensemble|57.0 | 46.49 | 
| SWAD     | TerraInc| ResNet-50 |50.0 | 50.79| 


## 🔧 Get Started

Environments Set up

```sh
git clone https://github.com/mluniproject/UniDG.git && cd UniDG
conda env create -f UniDG.yaml &&  conda activate UniDG
```

Datasets Preparation

```sh
python -m domainbed.scripts.download \
       --data_dir=./data
```

## ⏳UniDG- Training & Test-time adaptation

Train a model:

```sh
python -m domainbed.scripts.train\
       --data_dir=./data \
       --algorithm ERM \
       --dataset OfficeHome \
       --test_env 2 \
       --hparams "{\"backbone\": \"resnet50\"}" \
       --output_dir my/pretrain/ERM/resnet50
```

*Note that you can download pretrained checkpoints in the [Model Zoo](https://github.com/invictus717/UniDG#model-zoo-for-unidg).*

Then you can perform self-supervised adaptation:

```shell
python -m domainbed.scripts.unsupervised_adaptation \
       --input_dir my/pretrain/ERM/resnet50 \
       --adapt_algorithm=UniDG
```

## 📆UniDG- Collect Experimental Results

Then you can perform self-supervised adaptation:

```shell
python -m domainbed.scripts.collect_all_results\
       --input_dir=my/pretrain/ERM \
       --adapt_dir=results/ERM/resnet50 \
       --output_dir=log/UniDG/ \
       --adapt_algorithm=UniDG \
       --latex
```

## 📈 UniDG- Visualization results

For T-SNE visualization: 

```bash
python -m domainbed.scripts.visualize_tsne\
       --input_dir=my/pretrain/ERM \
       --adapt_dir=UniDG/results/ERM/resnet50 \
       --output_dir=log/UniDG/ \
       --adapt_algorithm=UniDG \
       --latex
```

For performance curves visualization: 

```bash
python -m domainbed.scripts.visualize_curves\
       --input_dir=my/pretrain/ERM \
       --adapt_dir=UniDG/results/ERM/resnet50 \
       --output_dir=log/UniDG/ \
       --adapt_algorithm=UniDG \
       --latex
```
## SIMPLE 
### Switch Branches
```
git switch SIMPLE
cd SeqML/SIMPLE
```
### Saving model predictions

Note that all pretrained models are not fine-tuned, so saving their predictions rather than take inference when needed can save a lot of time. Follow the steps given below to save the predictions of the pretrained models:

**Step 1:** Define the model pool by creating a .txt file that lists the names of the pre-trained models. This repository provides two examples: *'sample_list.txt'* and *'full_list.txt'*. 


**Step 2:** Set the path where the pretrained model cache will be saved to by running the following command:
```
export TORCH_HOME="./pytorch_pretrained_models/"
```


**Step 3:** Run the spec.py file with the following parameters to generate the predictions of all of the pretrained models in the model pool defined by the pretrain_model_list:
```
python -u spec.py --save_inference_only --dataset domainbed --domainbed_dataset PACS --pretrain_model_list modelpool_list/sample_list.txt --batch_size 256
```


### Running
To learn the specialized model-sample matching, use the following command as an example for the PACS dataset in DomainBed:
```
CUDA_VISIBLE_DEVICES=0 python spec.py --pretrain_model_list modelpool_list/sample_list.txt --dataset domainbed --domainbed_dataset PACS --domainbed_test_env 0'''

```

## LISA
### switch Branches
```
git switch Mixup

```
### Run Model
```
python -m domainbed.scripts.train       --data_dir=./data     --dataset TerraIncognita  --algorithm Inter_domain_adaptation        --test_env 0 --hparams "{\"backbone\": \"ConvNext\"}"     --output_dir my/pretrain/Inter_domain_adaptation/TerraIncognita/ConvNext/0 --steps 8000
```
## SWAD 
### Switch Branches
```
git switch SWAD

```

### Run Methods
```python train_all.py TR0 --dataset TerraIncognita --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py TR1 --dataset TerraIncognita --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py TR2 --dataset TerraIncognita --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir /my/datasets/path

```

## Citation
We used the work of  

@article{zhang2023unified,
      title={Towards Unified and Effective Domain Generalization}, 
      author={Yiyuan Zhang and Kaixiong Gong and Xiaohan Ding and Kaipeng Zhang and Fangrui Lv and Kurt Keutzer and Xiangyu Yue},
      year={2023},
      eprint={2310.10008},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@misc{cha_swad_2021,
	title = {{SWAD}: {Domain} {Generalization} by {Seeking} {Flat} {Minima}},
	shorttitle = {{SWAD}},
	url = {http://arxiv.org/abs/2102.08604},
	urldate = {2024-03-13},
	publisher = {arXiv},
	author = {Cha, Junbum and Chun, Sanghyuk and Lee, Kyungjae and Cho, Han-Cheol and Park, Seunghyun and Lee, Yunsung and Park, Sungrae},
	month = nov,
	year = {2021},
	note = {arXiv:2102.08604 [cs]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
	annote = {Comment: NeurIPS 2021 camera-ready},
	
}
@article{li_simple_2023,
	title = {{SIMPLE}: {SPECIALIZED} {MODEL}-{SAMPLE} {MATCHING} {FOR} {DOMAIN} {GENERALIZATION}},
	abstract = {In domain generalization (DG), most existing methods aspire to fine-tune a specific pretrained model through novel DG algorithms. In this paper, we propose an alternative direction, i.e., to efficiently leverage a pool of pretrained models without fine-tuning. Through extensive empirical and theoretical evidence, we demonstrate that (1) pretrained models have possessed generalization to some extent while there is no single best pretrained model across all distribution shifts, and (2) out-of-distribution (OOD) generalization error depends on the fitness between the pretrained model and unseen test distributions. This analysis motivates us to incorporate diverse pretrained models and to dispatch the best matched models for each OOD sample by means of recommendation techniques. To this end, we propose SIMPLE, a specialized model-sample matching method for domain generalization. First, the predictions of pretrained models are adapted to the target domain by a linear label space transformation. A matching network aware of model specialty is then proposed to dynamically recommend proper pretrained models to predict each test sample. The experiments on DomainBed show that our method achieves significant performance improvements (up to 12.2\% for individual dataset and 3.9\% on average) compared to state-of-the-art (SOTA) methods and further achieves 6.1\% gain via enlarging the pretrained model pool. Moreover, our method is highly efficient and achieves more than 1000× training speedup compared to the conventional DG methods with fine-tuning a pretrained model. Code and supplemental materials are available at https://seqml.github.io/simple.},
	language = {en},
	author = {Li, Ziyue and Ren, Kan and Jiang, Xinyang and Shen, Yifei and Zhang, Haipeng and Li, Dongsheng},
	year = {2023},
	
}
@inproceedings{yao_improving_2022,
	title = {Improving {Out}-of-{Distribution} {Robustness} via {Selective} {Augmentation}},
	url = {https://proceedings.mlr.press/v162/yao22b.html},
	abstract = {Machine learning algorithms typically assume that training and test examples are drawn from the same distribution. However, distribution shift is a common problem in real-world applications and can cause models to perform dramatically worse at test time. In this paper, we specifically consider the problems of subpopulation shifts (e.g., imbalanced data) and domain shifts. While prior works often seek to explicitly regularize internal representations or predictors of the model to be domain invariant, we instead aim to learn invariant predictors without restricting the model’s internal representations or predictors. This leads to a simple mixup-based technique which learns invariant predictors via selective augmentation called LISA. LISA selectively interpolates samples either with the same labels but different domains or with the same domain but different labels. Empirically, we study the effectiveness of LISA on nine benchmarks ranging from subpopulation shifts to domain shifts, and we find that LISA consistently outperforms other state-of-the-art methods and leads to more invariant predictors. We further analyze a linear setting and theoretically show how LISA leads to a smaller worst-group error.},
	language = {en},
	urldate = {2023-12-06},
	booktitle = {Proceedings of the 39th {International} {Conference} on {Machine} {Learning}},
	publisher = {PMLR},
	author = {Yao, Huaxiu and Wang, Yu and Li, Sai and Zhang, Linjun and Liang, Weixin and Zou, James and Finn, Chelsea},
	month = jun,
	year = {2022},
	note = {ISSN: 2640-3498},
	pages = {25407--25437},
}

```

## Acknowledge

This repository is based on [DomainBed](https://github.com/facebookresearch/DomainBed), [T3A](https://github.com/matsuolab/T3A), [timm](https://github.com/fastai/timmdocs). Thanks a lot for their great works!

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file.
