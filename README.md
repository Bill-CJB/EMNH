# NHDE
Code for NeurIPS2023 Paper: [Efficient Meta Neural Heuristic for Multi-Objective Combinatorial Optimization](https://github.com/bill-cjb/EMNH)

**Quick Start**

- To train a model, such as MOTSP with 20 nodes, set *TSP_SIZE=20* and *MODE=1* in *HYPER_PARAMS.py*, and then run *run.py* in the corresponding folder.
- To fine-tune and test a model, such as MOTSP with 20 nodes, set *TSP_SIZE=20* and *MODE=2* in *HYPER_PARAMS.py*, and then run *run.py* in the corresponding folder.
- Pretrained models for each problem can be found in the *result* folder.

**Reference**

If our work is helpful for your research, please cite our paper:
```
@inproceedings{chen2023efficient,
  title={Efficient Meta Neural Heuristic for Multi-Objective Combinatorial Optimization},
  author={Chen, Jinbiao and Wang, Jiahai and Zhang, Zizhen and Cao, Zhiguang and Ye, Te and Siyuan, Chen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}
```

