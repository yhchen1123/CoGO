# CoGO


Codes and models for the paper "CoGO: a contrastive learning framework to predict disease similarity based on gene network and ontology structure".
> CoGO: a contrastive learning framework to predict disease similarity based on gene network and ontology structure \
> Authors: Yuhao Chen, Yanshi Hu, Xiaotian Hu, Cong Feng, Ming Chen 

Contact: yuhaoc@zju.edu.cn. Any questions or discussions are welcomed!

## Abstract

**Motivation:** Quantifying the similarity of human diseases provides guiding insights to the discovery of micro-scope mechanisms from a macro scale. Previous work demonstrated that better performance can be gained by integrating multi-view data sources or applying machine learning techniques. However, designing an efficient framework to extract and incorporate information from different biological data using deep learning models remains unexplored.
**Results:** We present **CoGO**, a **Co**ntrastive learning framework to predict disease similarity based on **G**ene network and **O**ntology structure, which incorporates the gene interaction network and gene ontology (GO) domain knowledge using graph deep learning models. First, graph deep learning models are applied to encode the features of genes and GO terms from separate graph structure data. Next, gene and GO features are projected to a common embedding space via a non-linear projection. Then cross-view contrastive loss is applied to maximize the agreement of corresponding gene-GO associations and lead to meaningful gene representation. Finally, CoGO infers the similarity between diseases by the cosine similarity of disease representation vectors derived from related gene embedding. In our experiments, CoGO outperforms the most competitive baseline method on both AUROC and AUPRC, especially improves 19.57% in AUPRC (0.7733). Furthermore, we conduct a detailed case study of top similar disease pairs which is demonstrated by other studies. Empirical results show that CoGO achieves powerful performance in disease similarity problem.

## Contribution

1. We convert disease similarity prediction problem to multi-view graph representation learning problem.
   <div align="center"><img width="200" src="./images/multiview_net.png"/></div>
   <center>An example of multi-view network.</center>

2. We propose to incorporate the disease related molecular data and GO domain knowledge into disease similarity prediction problem. A contrastive learning based method is presented to learn their correlations. 
   <div align="center"><img width="700" src="./images/pipeline.png"/></div>
   <center>Overview of CoGO. In the training stage, GCN and RGCN are implemented to encode features of gene interaction network and GO graph. MLP is applied to map the output of GCN and RGCN to the common embedding space. Contrastive loss is used to maximize the agreement of corresponding genes and GO terms. In the inference stage, only trained GCN is preserved to calculate the gene embedding. And disease representation is derived from related gene embedding by average pooling.</center>

3. The proposed CoGO model achieves state-of-the-art performance in manual inspection data sets. Especially using Area Under Precision–Recall Curve (AUPRC) as evaluation metric. 
> We evaluate the disease similarity prediction performance using both AUROC and AUPRC. This is because ROC curves can present an overly optimistic view of an algorithm's performance when applied to imbalanced data sets. 
   <div align="center"><img width="600" src="./images/results.png"/></div>