import json
from pathlib import Path
import pickle

PATH = 'Corona2.json'

def load_transform(PATH):
    with open(PATH) as cs:
        data = json.load(cs)
    spacy_data=[]
    for element in data['examples']:
        ent_dict={}
        ent_list=[]
        index=[]
        for ent in element['annotations']:
            if ent['human_annotations']!=[] and ent['start'] not in index and ent['end'] not in index:
                index.append(ent['start'])
                index.append(ent['end'])
                ent_list.append((ent['start'],ent['end'],ent['tag_name']))
        ent_dict['entities']=ent_list
        spacy_data.append((element['content'],ent_dict))
    
    return spacy_data

spacy_data = load_transform(PATH)

# Save the data as pkl
with open('spacy_data.pkl', 'wb') as file:
        pickle.dump(spacy_data, file)