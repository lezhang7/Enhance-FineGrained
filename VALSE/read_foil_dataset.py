import json

def read_foils(foils_path):
    if "original-foil-dataset" in foils_path:
        foils_data = read_foil_dataset(foils_path)
    else:
        with open(foils_path) as json_file:
            foils_data = json.load(json_file)
    return foils_data

def read_foil_dataset(foils_path):
    """
    Read in the data of the original foil dataset and convert it on the fly to our format (dict/json).
    """
    with open(foils_path) as json_file:
        foil_dataset = json.load(json_file)

    foils_data = {}  # our format

    for foil in foil_dataset['annotations']:
        # For unimodal models, we always need foil, non-foil pairs to compare perplexity.
        if foil['foil'] == True:  # we have a foil not foil pair
            # recover the original sentence
            orig_sentence = foil['caption'].replace(foil['foil_word'], foil['target_word'])
            image_id = foil['image_id']
            foils_data[foil["foil_id"]] = {'dataset': 'FOIL dataset',
                                          'dataset_idx': foil["foil_id"],
                                          'original_split': 'test',
                                          'linguistic_phenomena': 'noun phrases',
                                          'image_file': f'COCO_val2014_{str(image_id).zfill(12)}.jpg', # COCO_val2014_000000522703.jpg all are "val"
                                          'caption': orig_sentence,
                                          'foils': [foil['caption']],
                                          'classes': foil['target_word'],
                                          'classes_foil': foil['foil_word'],
                                          }

    return foils_data