# DecisionTreeBaseline
A collection and implementation of several variants of classical decision tree algorithm, which can serve as baselines or comparative studies of Oblique Decision Tree research. 



<div align="center">

![Language](https://img.shields.io/badge/language-python-blue)
![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
[![License](https://img.shields.io/github/license/maoqiangqiang/decisiontreebaseline)](https://github.com/maoqiangqiang/DecisionTreeBaseline/blob/main/LICENSE)

A Collection of Oblique Decision Tree Algorithms for Regression in Python

</div>

## Introduction 
We developed the `DecisionTreeBaseline` repository to serve as baselines for comparative studies in Oblique Decision Tree research. Our motivation for developing this repository was the inconvenience of comparing other authors' Oblique Decision Tree algorithms, including `HouseHolder-CART (HHCART)`, `Continuously-Optimized-Oblique-Tree (CO2)`, `BUTIF`[^1], `OC1`, `RandCART`, `RidgeCART`[^2], `Nonlinear-Decision-Tree` and `Linear-Tree`<a href='#Reference'>[1-7]</a>. While some GitHub repositories have incorporated these algorithms, there is still a gap in the availability of a collective package that includes all aforementioned oblique decision tree algorithm together for regression. 

The development of this repository was based on some online GitHub resources, which were adapted for regression with some modifications. We extend our gratitudes to the authors and developers for their valuable contributions. For further information, please refer to the [Acknowledgements](#acknowledgements) section.


[^1]: might be `BUTIA` in the article of "A bottom-up oblique decision tree induction algorithm" (not sure). 
[^2]: might be `CART-LC` that uses Ridge Regression for linear combination. 

## Script Description 
- `src` folder contains the scripts of oblique decision trees. 
  - `HHCART.py` 
  - `CO2.py`
  - `BUTIF.py`
  - `OC1Regression.py`
  - `RandCART.py`
  - `RidgeCART.py`
  - `NonLinearDTRegr.py`
- `test` folder contains the script of running these algorithms.
  - All methods in `src`.
  - `CART` from `scikit-learn`
  - `Linear-Tree` from `linear-tree`

## Examples 
The example can be implemented via `python test/test_baseline.py`. 


## License
This repository is published under the terms of the `MIT License`. See [LICENSE](https://github.com/maoqiangqiang/DecisionTreeBaseline/blob/main/LICENSE) for more details.

## Reference 
<span id='Reference'>

[1] D. C. Wickramarachchi, B. L. Robertson, M. Reale, C. J. Price, and J. Brown, “HHCART: An Oblique Decision Tree.” arXiv, Apr. 14, 2015. doi: 10.48550/arXiv.1504.03415.

[2] M. Norouzi, M. D. Collins, D. J. Fleet, and P. Kohli, “CO2 Forest: Improved Random Forest by Continuous Optimization of Oblique Splits.” arXiv, Jun. 24, 2015. doi: 10.48550/arXiv.1506.06155.

[3] R. C. Barros, R. Cerri, P. A. Jaskowiak, and A. C. P. L. F. de Carvalho, “A bottom-up oblique decision tree induction algorithm,” in 2011 11th International Conference on Intelligent Systems Design and Applications, Nov. 2011, pp. 450–456. doi: 10.1109/ISDA.2011.6121697.

[4] S. K. Murthy, S. Kasif, and S. Salzberg, “A System for Induction of Oblique Decision Trees.” arXiv, Jul. 31, 1994. doi: 10.48550/arXiv.cs/9408103.

[5] R. Blaser and P. Fryzlewicz, “Random Rotation Ensembles,” Journal of Machine Learning Research, vol. 17, no. 4, pp. 1–26, 2016.

[6] L. Breiman, Classification and Regression Trees. New York: Routledge, 2017. doi: 10.1201/9781315139470.

[7] A. Ittner and M. Schlosser, “Non-linear decision trees-NDT,” in ICML, Citeseer, 1996, pp. 252–257.

</span>

## Acknowledgements 
We appreciate the authors' contributions for their published GitHub repositories. 
- [AndriyMulyar/sklearn-oblique-tree](https://github.com/AndriyMulyar/sklearn-oblique-tree)
- [hengzhe-zhang/scikit-obliquetree](https://github.com/hengzhe-zhang/scikit-obliquetree)
- [TorshaMajumder/Ensembles_of_Oblique_Decision_Trees](https://github.com/TorshaMajumder/Ensembles_of_Oblique_Decision_Trees)
- [rmontanana/Course-work](https://github.com/rmontanana/Course-work)
- [cerlymarco/linear-tree](https://github.com/cerlymarco/linear-tree)
