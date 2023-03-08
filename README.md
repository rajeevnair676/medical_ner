# Medical NER model

A model trained on understanding the medical terms and conditions using Spacy
The model is trained on Corona2.json file which is available in Kaggle, and the link is attached below:
https://www.kaggle.com/datasets/finalepoch/medical-ner

1. Install the dependencies from requirements.txt file
2. If you need to train the model on a new data, run the train_model.py file, and if you need to convert the data into the format spacy accepts, run data_load.py
3. The NER_med.ipynb is the notebook file to illustrate the training process
4. To run and test the model, execute the main.py from CMD, and enter the sentence to identify the entities. The model would identify the entities like medicine, medical condition, pathogen etc.
