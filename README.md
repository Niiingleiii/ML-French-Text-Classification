# Introduction

Hello! We are two passionate students, recent graduates with Bachelor's degrees in Business and Economics. Despite our academic backgrounds, we have always been drawn to the world of technology and language learning. As non-native French speakers, we've experienced firsthand the challenges of finding suitable reading materials that match our proficiency levels. This personal struggle, combined with our curiosity, led us to participate in a Kaggle competition for our machine learning class.

# The Challenge

The task was presented through a Kaggle competition. The goal was to develop a machine learning model that predicts the difficulty level of French-written texts for English speakers. This model could then be used in a recommendation system to suggest texts, such as recent news articles, that are appropriate for a user’s language level. For instance, presenting a text at the B2 level to a learner at the A1 level would be counterproductive, as it would be too difficult to understand. Conversely, providing texts that are too easy wouldn't contribute much to learning progress. Our mission was to find the perfect balance.

# The Journey Begins

As we delved into this project, we couldn't help but feel a mix of excitement and apprehension. The world of machine learning and natural language processing was relatively new to us, and we felt a bit out of our depth. The technical jargon, complex algorithms, and sophisticated models were intimidating, to say the least. We questioned whether our backgrounds in Business and Economics had prepared us for such a daunting task. Yet, the drive to overcome our language learning challenges and create something impactful kept us going.

We were not very confident at the start, and the fear of failure loomed large. But we decided to face our fears head-on, viewing this as an opportunity to learn and grow. With determination and a shared passion for language learning, we embarked on this journey, ready to tackle each obstacle one step at a time

![Start of the Journey](https://raw.githubusercontent.com/Niiingleiii/ML-French-Text-Classification/main/resized_rotated_image.webp)  

# Leveraging BERT for Improved Accuracy

After experimenting with both basic and advanced models combined with contextual features, the best result we achieved was an accuracy of 0.545. While this was a significant improvement, we aimed for better performance. To achieve this, we decided to explore BERT models.

## Understanding BERT

BERT, which stands for Bidirectional Encoder Representations from Transformers, is a revolutionary model in the field of natural language processing. Unlike traditional models, BERT is based on transformers, a deep learning architecture where every output element is connected to every input element, and the weightings between them are dynamically calculated based on their connection. This bidirectional approach allows BERT to understand the context of a word based on all of its surrounding words in a sentence, rather than just the words that precede or follow it.

## Exploring BERT Models

In our search for the best model, we discovered numerous BERT variants available in the Hugging Face community. Each variant offered unique strengths, tailored to different types of text data and tasks. We experimented with several BERT models, including DistilBERT, a smaller and faster version of BERT, and found that CamemBERT, a BERT model specifically trained on French data, worked the best for our task. See below for the performance of various BERT models we tried using the Weights and Biases tool:

- **DistilBERT**: A distilled version of BERT that is faster and smaller but retains much of BERT's performance capabilities.
- **RoBERTa**: A robustly optimized BERT approach that modifies key hyperparameters and training strategies to improve performance.
- **CamemBERT**: A BERT model specifically pre-trained on a large French corpus, making it particularly well-suited for our task.

![Weights and Biases Performance Tracking](link_to_performance_graph) *(Placeholder for actual performance graph)*

## Fine-tuning CamemBERT

Given our relatively small dataset—4,800 original entries plus 7,000 augmented entries—we faced the challenge of fine-tuning the CamemBERT model without overfitting. To address this, we implemented the following strategies:

- **Parameter Reduction**: We reduced the number of parameters in the model to make it more manageable and to prevent overfitting on our limited data.
- **Layer Freezing**: By freezing some of the layers in the model, we limited the number of parameters that needed to be updated during training. This approach allowed us to retain the pre-trained knowledge from the original CamemBERT model while fine-tuning only the top layers for our specific task.

## Ensembling for Enhanced Performance

To further improve our model's accuracy, we employed ensembling techniques. Ensembling involves combining the predictions of multiple models to produce a final prediction. This method leverages the strengths of each individual model and can lead to more robust and accurate results. We created an ensemble of several well-performing CamemBERT models, allowing them to "compete" and collaborate in making predictions. This approach helped us achieve higher accuracy by mitigating the weaknesses of any single model.

## Conclusion

Integrating BERT into our project marked a significant leap forward in our efforts to predict the difficulty of French texts accurately. Through careful model selection, fine-tuning, and ensembling, we harnessed the power of BERT's contextual understanding to achieve better results than we had with traditional models alone. This journey into the world of BERT not only improved our model's performance but also deepened our understanding of advanced natural language processing techniques. As we continue to refine and test our models, we are excited about the potential to make meaningful contributions to the field of language learning.




