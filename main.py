import spacy
from data_load import load_transform

data_path = 'Corona2.json'
model_path = "model"
spacy_data = load_transform(data_path)

    

if __name__ == "__main__":
    # do something only if the script is run directly
    model = spacy.load(model_path)
    text1 = str(input("Enter the text to extract the entities: "))
    doc = model(text1)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])