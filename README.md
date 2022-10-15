## Hey there 👋, I'm Aziz Baran 👨‍💻

I'm an undergraduate Linguistics student.

I'm passionate about constantly improving myself in the fields of NLP, Machine learning and Deep learning with the aim of bringing the most effective solutions to different types of language-related real-world problems.

I'm interested in leveraging the state-of-the-art NLP transformer models like BERT, GPT-2 and mT5 to solve challenges occuring in natural languages. Therefore, I enjoy making various language models and applications and then deploying them in 🤗 Hugging Face so that people in need can use them to overcome their difficulties. These models and apps that I deployed, including "cst5", the Czech language model that was downloaded more than 500 times since its release, are available in my [🤗 Hugging Face profile](https://huggingface.co/azizbarank).

Currently, I'm learning [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers), an extension made by integrating adapters into the Hugging Face Transformer models via [AdapterHub](https://adapterhub.ml/). 

In my spare time, I keep up with the recent developments in the field of NLP/CL by reading articles and the latest published books.


## 🛠️ Skills

**Languages** 

   ![Python](https://img.shields.io/badge/Python-black?style=flat-square&logo=python&logoColor=ffdd54?)
  
**Natural Language Processing**

   ![Hugging Face Transformers](https://img.shields.io/badge/🤗_Transformers-black?style=flat-square&logo=Hugging_Face&logoColor=white) ![NLTK](https://img.shields.io/badge/NLTK-black?style=flat-square&logo=python&logoColor=blue) ![spaCy](https://img.shields.io/badge/spaCy-black?style=flat-square&logo=spacy&logoColor=blue) ![image](https://img.shields.io/badge/Gensim-black?style=flat-square&logo=&logoColor=blue)
  
**Machine Learning / Deep Learning** 

   ![scikit-learn](https://img.shields.io/badge/scikit--learn-black?style=flat-square&logo=scikit-learn&logoColor=F7931E?) ![NumPy](https://img.shields.io/badge/Numpy-black?style=flat-square&logo=numpy&logoColor=777BB4) ![Pandas](https://img.shields.io/badge/pandas-black?style=flat-square&logo=pandas&logoColor=2C2D72) ![PyTorch](https://img.shields.io/badge/PyTorch-black?style=flat-square&logo=PyTorch&logoColor=%23EE4C2C.svg) ![W&B](https://img.shields.io/badge/Weights_&_Biases-black?style=flat-square&logo=WeightsAndBiases&logoColor=FFBE00?)
  
**IDEs & Notebooks** 
  
  ![Spyder](https://img.shields.io/badge/Spyder%20Ide-black?style=flat-square&logo=spyder%20ide&logoColor=FF0000) ![Jupyter](https://img.shields.io/badge/Jupyter_Lab-black?style=flat-square&logo=Jupyter&logoColor=F37626) ![Google Colab](https://img.shields.io/badge/Colab-black?style=flat-square&logo=googlecolab&color=black) ![Jupyter Notebook](https://camo.githubusercontent.com/9e480c584c43933793430e771351727de61ea44580dd08cb37d30c350d290377/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f2d4a7570797465722532304e6f7465626f6f6b2d626c61636b3f7374796c653d666c61742d737175617265266c6f676f3d4a757079746572)
  
**Other Technologies & Tools** 

   ![Anaconda](https://img.shields.io/badge/Anaconda-black?style=flat-square&logo=anaconda&logoColor=342B029.svg) ![Conda](https://img.shields.io/badge/conda-black?&style=flat-square&logo=anaconda&logoColor=342B029.svg) ![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-black?style=flat-square&logo=Streamlit&logoColor=FF4B4B)


## 📃 Projects

* [Turkish Sentiment Analyser](https://github.com/azizbarank/Turkish-Sentiment-Analyser) - [Hugging Face](https://huggingface.co/azizbarank/distilbert-base-turkish-cased-sentiment) - [Web App](https://huggingface.co/spaces/azizbarank/Turkish-Sentiment-Analysis)

  Fine-tuned the distilled Turkish BERT model on a review classification dataset for sentiment analysis. The final model achieved 86% accuracy and was deployed to Hugging Face Spaces using Streamlit as an interactive web app. The app provides a no-code way for people to see whether a particular review is "positive" or "negative". 

* [distilroberta-base-sst-2-distilled](https://github.com/azizbarank/distilroberta-base-sst-2-distilled) - [Hugging Face](https://huggingface.co/azizbarank/distilroberta-base-sst2-distilled)

  Created a DistilRoBERTa model that is fine-tuned on the GLUE SST-2 dataset by the method of task specific knowledge distillation applied to the original teacher RoBERTa model. After hyperparameter tuning with Optuna, the final model achieves **92% accuracy** (original RoBERTa's being **94.8%**) on SST-2 despite much less parameters and it is twice as fast as the original teacher model. Therefore, the resulted model can be used directly and more efficiently for sentiment analysis tasks in English.

* [Toxic Comment Detector](https://github.com/azizbarank/Toxic-Comment-Detector) - [Web App](https://huggingface.co/spaces/azizbarank/Toxic-Comment-Detection-App)
 
  Binary classification project to predict whether a comment is toxic or not. Three machine learning models of Multinomial Naive Bayes, Logistic Regression, and Support Vector Machine were used. The best model was a Naive Bayes classifier with TF-IDF Vectorizer with the F1 and Recall scores of **0,85** and **0,88**, respectively. The application uses this model to predict the toxicity of comments.
 
* [cst5](https://github.com/azizbarank/Czech-T5-Base-Model) - [Hugging Face](https://huggingface.co/azizbarank/cst5-base)

  cst5 is a tiny T5 model for the Czech language that is based on the smaller version of Google's mT5 model. cst5 is meant to help people in doing experiments for the Czech language by enabling them to use a lightweight model, rather than the 101 languages-covering massive mT5. cst5 was obtained by retaining only the Czech and English embeddings of the mT5 model, during which the total size was reduced from **2.2GB** to **0.9GB** as a result of shrinking the original "sentencepiece" vocabulary from **250K** to **30K** tokens and parameters from **582M** to **244M**. cst5, thus, allows people to do fine-tuning for further downstream tasks in the Czech language with less size requirement and without any loss in quality from the original multilingual model.

* [Financial Sentiment Analysis with Machine Learning, LSTM, and BERT Transformer](https://github.com/azizbarank/Financial-Sentiment-Analysis-with-Machine-Learning-LSTM-and-BERT-Transformer)

  Financial sentiment analysis project to predict if a given financial text is to be considered as positive, negative or neutral. Machine learning, LSTM, and BERT transformer were used during the process. The best result was obtained with BERT. It achieved the accuracy score of **0.77**.

## 💻 My Blog Posts

* [4 Highly Practical Books to Read for Applied NLP](https://azizbarank.github.io/post/d/)
* [Dealing with Lack of Computational Resources in NLP](https://azizbarank.github.io/post/b/)
* [Simple Guide to Building and Deploying Transformer Models with Hugging Face](https://azizbarank.github.io/post/c/)
* [Transformers as Feature Extractors](https://azizbarank.github.io/post/a/)
