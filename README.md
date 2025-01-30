# Statistical Significance of Feature Importance Rankings

This repository contains the code accompanying the paper, *"Statistical Significance of Feature Importance Rankings."* The provided tools enable robust analysis of feature importance rankings using SHAP values, with additional support for LIME-based methodologies.

---

## Installation

To set up the required environment for SHAP-based analysis and methodologies, follow these steps:

1. **Basic functionality**. Most of our codebase is devoted to analyzing SHAP values. To set it up, install the required packages with the following line in a Python or Conda virtual environment:
   
   ```bash
   pip install -r requirements.txt
   ```
2. **Optional: Enable LIME functionality**  
   * Install the [S-LIME](https://github.com/ZhengzeZhou/slime) package by Zhenghe Zhou.  
   * Replace the `slime/slime/lime_tabular.py` file in the S-LIME package with the version provided in this repository. Our sole modification has the `slime()` function also output a Boolean flag indicating whether the algorithm converged within the specified sample budget.  
   * Navigate to the S-LIME package directory, where `setup.py` is located. In the same environment, install the package:
    ```bash
    pip install .
    ```
## Key Components and Functions

The `HelperFiles/` directory contains functions for our methodology, experiments, and analysis.

### Retrospective Analysis

- **Estimate Shapley Values:**
  - Use `shapley_sampling()` or `kernelshap()` in `retrospective.py` to estimate SHAP values via Shapley Sampling or KernelSHAP.
- **Verify Rankings (Procedure 1):**
  - Validate the ranks of observed SHAP Values - or any variable importance measures, for that matter - using the `find_num_verified()` function in `helper.py`.
- **Assess Top-K Stability (Procedure 2):**
  - Evaluate the stability of the top-K set using the `test_top_k_set()` function, also in `helper.py`.
    
### Top-K Sampling Algorithms

The repository includes two algorithms for evaluating feature importance rankings and sets: **SPRT-SHAP** and **RankSHAP**. Both are implemented in `top_k.py`:  
  - Use the `sprtshap()` and `rankshap()` functions to apply these methods.  
  - The parameter *guarantee* determines whether the algorithm focuses on ranking stability or set stability.

## Usage and Examples

For detailed examples of the provided functions on real-world data, refer to the `vignette.ipynb` notebook. This vignette performs ranking and set stability analysis in both the retrospective and algorithmic cases. 
  - Retrospective analysis with Shapley Sampling and KernelSHAP.
  - Top-K algorithms with SPRT-SHAP on KernelSHAP, RankSHAP on Shapley Sampling, and our application of S-LIME.
