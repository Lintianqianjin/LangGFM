<div align="center">

# LangGFM: A Large Language Model Alone Can be a Powerful Graph Foundation Model

 <a href='https://arxiv.org/abs/2410.14961'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
<!-- [![GitHub last commit](https://img.shields.io/github/last-commit/Lintianqianjin/LangGFM)](https://github.com/Lintianqianjin/LangGFM/commits/main/) -->

<img src="assets/logo.png" alt="LangGFM Logo" width="300"/> 
</div>

<!-- 
# LangGFM
Official code of "A Large Language Model Alone Can be a Powerful Graph Foundation Model" -->

<!-- --- -->

Environment
---
For pip (with python 3.11): `requirements.txt`   
`pip install -r requirements.txt`


For conda `environment.yml`  
`conda env create -f environment.yml`


run commands in `LangGFM/`


To ensure compatibility with all graphs in the open world, treat all graphs as MultiDiGraph: 
* A directed graph class that can store multiedges. 
* Multiedges are multiple edges between two nodes. Each edge can hold optional data or attributes.
* A MultiDiGraph holds directed edges. Self loops are allowed.

Treat heterogeneous graph as homogeneous graph with edge_type and node_type as feature.

In an undirected graph, edges have no direction. An undirected edge can be represented as two directed edges in opposite directions. 



Please estimate the 10-based logarithm of the aqueous solubility ($\log S$) of the given compound directly from its structure. The calculation formula is as follows: $\log S = 0.16 - 0.63 \log P - 0.0062 MW + 0.066 RB - 0.74 Apolar$, where $S$ is Aqueous solubility (mol/L), $\log P$ is the octanol-water partition coefficient (lipophilicity indicator), $MW$ is Molecular weight (g/mol), $RB$ is the number of rotatable bonds (flexibility measure) and $Apolar$ is aromatic proportion, i.e., the proportion of heavy atoms in the molecule that are in anaromatic ring. 
