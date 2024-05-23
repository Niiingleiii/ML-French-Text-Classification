# Machine Learning 101: Our Machine Learning Adventure from Novices to Innovators
Team Nvidia: Ning Lei, Emily Sun Reed

## Repository Index: 
- **Kaggle Competition Rank:** 13th/20
- **Best Result:** 0.603
- [**Main Colab Notebook**](https://raw.githubusercontent.com/Niiingleiii/ML-French-Text-Classification/main/Nvidia_Main_Notebook.ipynb): The main python notebook for this assignment. Note that this main notebook captures the codes of our main models. During the development process, we experimented with many more codes. However, for simplicity, we have not included all the modes in the main notebook. The main notebook focuses on the core models that are central to our project.
- [**YouTube Video**](https://www.youtube.com/watch?v=g2VQgumq7r4): The Youtube video for this assignment.
- **Streamlit**
![Streamlit](https://github.com/Niiingleiii/ML-French-Text-Classification/blob/main/images/streamlit1.png)![Streamlit](https://github.com/Niiingleiii/ML-French-Text-Classification/blob/main/images/streamlit2.png)

## 1.Introduction

Hello! We are two passionate students, recent graduates with Bachelor's degrees in Business and Economics. Despite our academic backgrounds, we have always been drawn to the world of technology and language learning. As non-native French speakers, we've experienced firsthand the challenges of finding suitable reading materials that match our proficiency levels. This personal struggle, combined with our curiosity, led us to participate in a Kaggle competition for our machine learning class.

## 2.The Challenge

The task was presented through a Kaggle competition. The goal was to develop a machine learning model that predicts the difficulty level of French-written texts for English speakers. This model could then be used in a recommendation system to suggest texts, such as recent news articles, that are appropriate for a user’s language level. For instance, presenting a text at the B2 level to a learner at the A1 level would be counterproductive, as it would be too difficult to understand. Conversely, providing texts that are too easy wouldn't contribute much to learning progress. Our mission was to find the perfect balance.

## 3.The Journey Begins

As we delved into this project, we couldn't help but feel a mix of excitement and apprehension. The world of machine learning and natural language processing was relatively new to us, and we felt a bit out of our depth. The technical jargon, complex algorithms, and sophisticated models were intimidating, to say the least. We questioned whether our backgrounds in Business and Economics had prepared us for such a daunting task. Yet, the drive to overcome our language learning challenges and create something impactful kept us going.

We were not very confident at the start, and the fear of failure loomed large. But we decided to face our fears head-on, viewing this as an opportunity to learn and grow. With determination and a shared passion for language learning, we embarked on this journey, ready to tackle each obstacle one step at a time

![Start of the Journey](https://raw.githubusercontent.com/Niiingleiii/ML-French-Text-Classification/main/images/resized_rotated_image.webp)

## 4.Tackling Basics, Tokenization, Advanced Classifiers, and Feature Engineering


To start, we decided to use basic classifier models. This meant diving into the world of Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Random Forest. Without a deep understanding of our data, we applied these models directly, hoping to establish a simple baseline. 

### 4.1.Exploring Tokenizers
For the choice of tokenzier, we turned to TF-IDF (Term Frequency-Inverse Document Frequency, a numerical statistic that reflects the importance of a word in a document relative to a collection of documents) and then a more sophisticated tool, CamemBERT tokenization, a pre-trained language model for French. Despite not fully understanding CamemBERT's complexities, we did not realize it could also function as a classifier and assumed it was solely for tokenization. We used CamemBERT only because we had heard from classmates that BERT-like models were top-notch for text processing. Our initial results, using CamemBERT tokenization and Random Forest classifier, yielded a baseline accuracy of 50%.

### 4.2.Exploring Advanced Classifiers

Seeking better results, we explored more advanced classifiers known for their performance in text classification tasks:

- **LGBMClassifier:** A gradient boosting framework that builds models in a sequential manner to minimize errors.
- **XGBClassifier:** Another gradient boosting algorithm, focusing on performance and speed.
- **CatBoostClassifier:** Specifically designed to handle categorical data efficiently and prevent overfitting.

Unfortunately, the performance of these models did not meet our expectations, pushing us to rethink our approach.

### 4.3.Diving Deeper into Data Input

Realizing that the quality of input data might be the issue, we started reading articles on how to enhance model understanding of language complexity. Inspired by a helpful [Medium article](https://towardsdatascience.com/linguistic-complexity-measures-for-text-nlp-e4bf664bd660), we began cleaning our dataset by removing punctuation and capital letters, and adding basic features like the number of characters and average word length. This small step improved our accuracy to 54.3%, a sign that we were on the right track.

### 4.4.Adding Complex Features

Encouraged by our progress, we decided to add more complex textual features inspired by an [article](https://www.researchgate.net/publication/363085243_Quantifying_French_Document_Complexity) on quantifying French document complexity. We integrated:

- **Lexical Richness Metrics:** Such as Types-Token Ratio (TTR), Mean Segmental Type-Token Ratio (MSTTR), Moving-Average Type-Token Ratio (MATTR), and Measure of Textual Lexical Diversity (MTLD).
- **Vocabulary Complexity Metrics:** Assessing the richness and diversity of vocabulary.
- **Syntactic Complexity Measures:** Including Mean Length of Sentence (MLS), Clauses per Sentence (C/S), Mean Length of Clause (MLC), and T-units.
- **Part-of-Speech (POS) Features:** Analyzing the grammatical structure.
- **Readability Scores:** Such as the Kincaid-McCandless (KM) score and BINGUI index.

However, this ambitious addition led to a significant drop in performance. We suspected multicollinearity, where new features overlapped with existing ones, confusing the model. Using Variance Inflation Factor (VIF) analysis, we confirmed high multicollinearity among our features. Simplifying our feature set to include only the number of characters and average length brought our model's performance back to a more reasonable level.

### 4.5.Data Augmentation

To further improve, we explored data augmentation techniques, which involve generating new data from existing data to enhance the training set. We experimented with:

- **Back Translation:** Translating sentences into another language and then back to French, creating variety without changing the meaning.
- **Word Replacement:** Replacing words with synonyms to generate diverse sentence structures.

Adding 7000 new sentences through word replacement significantly boosted our validation accuracy to 80%. Thrilled, we uploaded our model to Kaggle, only to find a marginal improvement to 54.5% on the test set. This discrepancy between validation and test performance highlighted an overfitting issue, where our model performed well on the training data but poorly on unseen data.

## 5.Leveraging BERT for Improved Accuracy

After experimenting with both basic and advanced models combined with contextual features, the best result we achieved was an accuracy of 0.545. While this was a significant improvement, we aimed for better performance. To achieve this, we decided to explore BERT models.

### 5.1.Understanding BERT

BERT, which stands for Bidirectional Encoder Representations from Transformers, is a revolutionary model in the field of natural language processing. Unlike traditional models, BERT is based on transformers, a deep learning architecture where every output element is connected to every input element, and the weightings between them are dynamically calculated based on their connection. This bidirectional approach allows BERT to understand the context of a word based on all of its surrounding words in a sentence, rather than just the words that precede or follow it.

### 5.2.Exploring BERT Models

In our search for the best model, we discovered numerous BERT variants available in the Hugging Face community. Each variant offered unique strengths, tailored to different types of text data and tasks. We experimented with several BERT models, including DistilBERT, a smaller and faster version of BERT, and found that CamemBERT, a BERT model specifically trained on French data, worked the best for our task. See below for the performance of various BERT models we tried using the Weights and Biases tool:

- **DistilBERT**: A distilled version of BERT that is faster and smaller but retains much of BERT's performance capabilities.
- **RoBERTa**: A robustly optimized BERT approach that modifies key hyperparameters and training strategies to improve performance.
- **CamemBERT**: A BERT model specifically pre-trained on a large French corpus, making it particularly well-suited for our task.

Please see below for graphs of different BERT models we tried over Weights & Biases, an AI developer platform. 

![Weights and Biases Performance Tracking](https://github.com/Niiingleiii/ML-French-Text-Classification/blob/main/images/Weights%26Biases.jpeg)
### 5.3.Fine-tuning CamemBERT

Given our relatively small dataset—4,800 original entries plus 7,000 augmented entries—we faced the challenge of fine-tuning the CamemBERT model without overfitting. To address this, we implemented the following strategies:

- **Parameter Reduction**: We reduced the number of parameters in the model to make it more manageable and to prevent overfitting on our limited data.
- **Layer Freezing**: By freezing some of the layers in the model, we limited the number of parameters that needed to be updated during training. This approach allowed us to retain the pre-trained knowledge from the original CamemBERT model while fine-tuning only the top layers for our specific task.

### 5.4.Ensembling for Enhanced Performance
To further improve our model's accuracy, we employed ensembling techniques. Ensembling involves combining the predictions of multiple models to produce a final prediction. This method leverages the strengths of each individual model and can lead to more robust and accurate results. By creating an ensemble of several well-performing CamemBERT models, we initially achieved an accuracy of 0.601. This approach helped us mitigate the weaknesses of any single model.

### 5.5.Continuous Tuning and Improvement
Not satisfied with stopping there, we continued to fine-tune the CamemBERT model and explore various hyperparameter configurations. Through meticulous tuning and optimization, we eventually reached our best accuracy of 0.603 and ranked 13th among 20 teams. This fine-tuning process involved adjusting learning rates, batch sizes, and other critical parameters to extract the maximum potential from the CamemBERT model.

![Kaggle Competition Result](https://github.com/Niiingleiii/ML-French-Text-Classification/blob/main/images/KaggleResult.jpg)

### 5.6.BERT - Conclusion
Integrating BERT into our project marked a significant leap forward in our efforts to predict the difficulty of French texts accurately. Through careful model selection, fine-tuning, and ensembling, we harnessed the power of BERT's contextual understanding to achieve better results than we had with traditional models alone. Our journey with BERT models, particularly CamemBERT, led us to achieve a best accuracy of 0.603, reflecting the potential of advanced NLP techniques in language learning applications. As we continue to refine and test our models, we are excited about the potential to make meaningful contributions to the field of language learning.

## 6.Results
| Metric    | Logistic Regression | KNN      | Decision Tree | Random Forest | Random Forest (best parameter) | Extra Trees | LightGBM | XGBoost | Catboost | Best Model (seed 42) |
|-----------|---------------------|----------|---------------|---------------|-------------------------------|-------------|----------|---------|----------|----------------------|
| Accuracy  | 0.374576            | 0.413559 | 0.598305      | 0.729661      | 0.742373                      | 0.741525    | 0.704237 | 0.722881| 0.673729 | 0.724576             |
| Precision | 0.357355            | 0.435093 | 0.598663      | 0.730144      | 0.743800                      | 0.740819    | 0.702897 | 0.721097| 0.672656 | 0.726683             |
| Recall    | 0.362858            | 0.407529 | 0.596806      | 0.728259      | 0.741678                      | 0.739245    | 0.701322 | 0.720420| 0.670246 | 0.724576             |
| F1        | 0.343414            | 0.399650 | 0.596925      | 0.727885      | 0.741665                      | 0.739451    | 0.701451 | 0.720414| 0.671055 | 0.724277             |

### Metrics Explanation

1. **Accuracy**: Accuracy is the ratio of correctly predicted instances to the total instances. It is a general measure of the model's performance, indicating how often the model makes correct predictions overall.
  
2. **Precision**: Precision measures the accuracy of positive predictions. It is the ratio of true positive predictions to the total positive predictions (true positives + false positives). High precision indicates that the model makes very few false positive errors.
  
3. **Recall**: Recall, or sensitivity, measures the model's ability to identify all relevant instances. It is the ratio of true positive predictions to the total actual positives (true positives + false negatives). High recall indicates that the model captures most of the relevant instances, making few false negative errors.
  
4. **F1 Score**: The F1 score is the harmonic mean of precision and recall, providing a balance between the two. It is useful for evaluating models where there is an uneven class distribution or when both precision and recall are important.

### Analysis

- **Logistic Regression** and **KNN** have relatively low scores across all metrics, indicating that these simpler models struggle with the task at hand.
- **Decision Tree** performs better, but still lags behind the more complex models.
- **Random Forest** and **Random Forest (best parameter)** show significant improvements, with the best parameter version achieving the highest scores in accuracy (0.742373), precision (0.743800), recall (0.741678), and F1 score (0.741665).
- **Extra Trees**, **LightGBM**, **XGBoost**, and **Catboost** also perform well, with scores close to the best Random Forest model, but they do not consistently outperform it.
- **Best Model (seed 42)** has comparable scores to these advanced models, with slightly lower metrics but still showing strong overall performance.

### Best Model (seed 42) Performance

The Best Model (seed 42) is considered the best because it maintains high metrics while exhibiting low overfitting, resulting in better performance on the test set. The consistency of the metrics (accuracy, precision, recall, and F1 score) across training and test data indicates that the model generalizes well to unseen data, making it a robust choice for practical applications.


![Confusion Matrix](https://github.com/Niiingleiii/ML-French-Text-Classification/blob/main/images/Confusion_Matrix.png)

The confusion matrix further demonstrates the model's robustness, showing high correct classification rates and relatively low misclassification rates. 

- **A1 (182 correctly classified, 30 misclassified as A2, etc.)**: This indicates that the model performs well in correctly identifying A1 instances but has some misclassification errors, especially with A2 and B1.
- **A2 (115 correctly classified, 32 misclassified as A1, etc.)**: The model struggles a bit more here, with a significant number of A2 instances being misclassified as A1 or B1.
- **B1 (127 correctly classified, 36 misclassified as A2, etc.)**: B1 has a substantial number of misclassifications, indicating room for improvement in distinguishing B1 from similar categories.
- **B2 (137 correctly classified, some misclassified as B1, C1, etc.)**: The model shows a high number of correct classifications for B2 but also some confusion with adjacent categories like B1 and C1.
- **C1 and C2 (160 and 134 correctly classified, respectively)**: These categories show strong performance, with high numbers of correct classifications and fewer misclassifications.


## 7.Remaining Questions

Throughout this project, we encountered several challenges and questions that left us pondering the nuances of machine learning theory versus practical application. One of the most prominent issues we faced was overfitting.

In our experiments, some models yielded exceptionally high accuracy on the training data but failed to generalize well to the validation set. This phenomenon, known as overfitting, occurs when a model learns the noise and details in the training data to the extent that it negatively impacts its performance on new data. Despite our efforts to mitigate overfitting through techniques such as parameter reduction and layer freezing, it was a persistent challenge.

In our machine learning classes, we were taught the importance of finding the best fit—training the model just enough to capture the underlying patterns in the data without learning the noise. The theory emphasizes monitoring the validation loss and stopping the training process when the validation loss starts to increase again, indicating the onset of overfitting. However, in practice, we observed that the relationship between training and validation loss wasn't always straightforward.

For instance, there were instances where a slight increase in validation loss didn't necessarily correlate with a significant drop in model performance on unseen data. On the contrary, some models that exhibited minor overfitting still performed remarkably well on the test set. This observation led us to question the rigidity of theoretical guidelines when applied to real-world data.

![Overfitting](https://raw.githubusercontent.com/Niiingleiii/ML-French-Text-Classification/main/images/Overfitting.webp)

These experiences raised several questions for us:

1. **How to Balance Overfitting and Model Performance**: When is it acceptable to allow a model to slightly overfit if it results in better generalization on the test set?
2. **Practical vs. Theoretical Guidance**: How should we reconcile the theoretical teachings of machine learning with the practical challenges and anomalies we encounter?
3. **Hyperparameter Tuning**: Given the extensive trial-and-error involved in tuning, what strategies can help streamline this process while maintaining model robustness?

## 8.End of Our Journey - Reflections

Looking back on our journey through this project, we realize how much we have learned and grown. We want to encourage others who might be embarking on a similar path. Start early to give yourself time to experiment, iterate, and learn from mistakes. Don’t be afraid of coding, even if your background is in a different field. We embraced coding despite our Business and Economics backgrounds, and it became a powerful tool in solving complex problems.

Throughout the project, we did our best and stayed committed. We maintained a positive attitude, pushing ourselves without succumbing to stress. Staying positive was crucial; it helped us navigate the challenges and stay motivated.

Initially, we thought this project would be incredibly difficult, especially given our non-technical backgrounds. However, we persevered and achieving an accuracy above 0.6 has been incredibly empowering. This experience has shown us that with determination and effort, we can overcome any challenge. We feel empowered and are excited to continue our journey in machine learning.

![End of our Journey](https://raw.githubusercontent.com/Niiingleiii/ML-French-Text-Classification/main/images/Nvidia%20Team.webp)





