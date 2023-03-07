import spacy
import random
import pickle
from tqdm import tqdm
from pathlib import Path
from spacy.training.example import Example
from data_load import load_transform

PATH = 'Corona2.json'
spacy_data = load_transform(PATH)


model = None
output_dir=Path("model")
n_iter=100

#load the model
if model is not None:
    nlp = spacy.load(model)  
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')  
    print("Created blank 'en' model")

#set up the pipeline
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')

for _, annotations in spacy_data:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(spacy_data)
        losses = {}
        for text, annotations in tqdm(spacy_data):
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update(
                [example],  
                drop=0.5,  
                sgd=optimizer,
                losses=losses)
        print(losses)

# Save the model
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)