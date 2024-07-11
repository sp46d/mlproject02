# Text summarization
In this project, I build my own text summarization app from scratch, using HuggingFace API. This project is designed as an end-to-end project, allowing me to practice the whole process of machine learning development, which includes setting up the environments, model development, deploying the model on cloud, and maintaining the model using CI/CD framework. 

This project generally follows the guideline provided by 'DSwithBappy' in his [YouTube video](https://youtu.be/p7V4Aa7qEpw?list=PLDcfxd4ep_-5N_B-SRYvrqrLD-Dd8WxUC), with some tweaks made by myself to make the app meet my needs.


## Workflows

1. Update config.yaml 
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update main.py
8. Update app.py

## TODO

Use colab to train the model and transfer the learning to the local machine:

- [ ] Implement HuggingFace PEFT API for training the model
  - Decide between freezing layers and using PEFT for training
- [ ] Download PubMed dataset
  - Select the subset of the dataset to train
- [ ] Train the model on Colab
- [ ] Modify the code to be able to use the pretrained parameters on colab (if not present, train the model locally)