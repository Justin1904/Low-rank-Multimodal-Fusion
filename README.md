# Low-rank-Multimodal-Fusion
This is the repository for "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors", Zhun and Ying, et. al. ACL 2018. This repo will be populated shortly afterwards.

# Requirements
Python 2.7


PyTorch <= 0.3


CMU-MultimodalDataSDK (Compatible version currently not available): https://github.com/A2Zadeh/CMU-MultimodalSDK

# Important Notice
The current version of our code depends on the **legacy version** of CMU-MultimodalDataSDK (latest version can be found at https://github.com/A2Zadeh/CMU-MultimodalSDK) **to run the experiments**. However, the SDK has been updated since we last used it, and the data structures and API it offers has changed significantly. The legacy version of the Data SDK along with data in the older data structure is not available right now. In this case, we will either adapt our code for experiments to the latest version of the Data SDK or offer a serialized data object that can be directly used with our code in the near future. 


**Meanwhile, the implementation of the model in `model.py` does not depend on this so it can still be directly used on your own data.**
