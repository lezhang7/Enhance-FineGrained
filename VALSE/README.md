# VALSE :dancer:

:dancer: VALSE: A Task-Independent Benchmark for Vision and Language Models Centered on Linguistic Phenomena. https://aclanthology.org/2022.acl-long.567/

📰 News: Accepted at ACL 2022 Main.

## Papers with Code entries for capturing future results on the benchmark
* Paper: https://paperswithcode.com/paper/valse-a-task-independent-benchmark-for-vision-1
* Dataset: https://paperswithcode.com/dataset/valse
   * We encourage the community to **submit their results** there to keep track of the progress on this benchmark. Disclaimer: Since we often cannot run models produced by the community, we have to trust the results reported by the respective model authors.

## Data Instructions
Please find the data in the `data` folder. The dataset is in `json` format and contains the following relevant fields:
* A reference to the image in the original dataset: `dataset` and `image_file`.
* The valid sentence, the caption for VALSE: `caption`.
* The altered caption, the `foil`.
* The annotator's votes (3 annotators per sample): `mturk`.
    * The subentry `caption` counts the number of annotators who chose the caption, but/and not the foil, to be the one describing the image.
    * The subentry `foil` counts how many of the three annotators chose the foil to be (also) describing the image.
    * For more information, see subsec. 4.4 and App. E of the [paper](https://aclanthology.org/2022.acl-long.567/).

:bangbang: Please be aware that the jsons are containing both valid (meaning: validated by annotators) and non-validated samples. In order to work only with the **valid set**, please consider filtering them:

> We consider a **valid foil** to mean: at least two out of three annotators identified the caption, but not the foil, as the text which accurately describes the image.

This means that the valid samples of the dataset are the ones where `sample["mturk"]["caption"] >= 2`.

Example instance:
```python
{
    "actions_test_0": {
        "dataset": "SWiG",                        # dataset from where the image and caption originate from
        "original_split": "test",                 # the split of the original dataset in which the sample belonged to
        "dataset_idx": "exercising_255.jpg",      # the sample id in the original dataset
        "linguistic_phenomena": "actions",        # the linguistic phenomenon targeted
        "image_file": "exercising_255.jpg",       # the image filename (in the original dataset)
        "caption": "A man exercises his torso.",  # image caption
        "classes": "man",                         # the word of the caption that was replaced
        "classes_foil": "torso",                  # the foil word / phrase
        "mturk": {                                # Amazon Mechanical Turk annotation (validation) results
            "foil": 0,                            # how many annotators voted that the foil describes the image
            "caption": 3,                         # how many annotators voted that the caption only (and not the foil) to describe the image
            "other": 0
        },
        "foil": "A torso exercises for a man."    # foil where one word / phrase is exchanged in the original caption such that the foil caption does not describe the image anymore
    }, ...
}
```

## Images
For the images, please follow the downloading instructions of the respective original dataset. The provenance of the original images is mentioned in the json files in the field `dataset`.

# Reference
Please cite our [:dancer: VALSE paper](https://aclanthology.org/2022.acl-long.567/) if you are using this dataset.

```
@inproceedings{parcalabescu-etal-2022-valse,
    title = "{VALSE}: A Task-Independent Benchmark for Vision and Language Models Centered on Linguistic Phenomena",
    author = "Parcalabescu, Letitia  and
      Cafagna, Michele  and
      Muradjan, Lilitta  and
      Frank, Anette  and
      Calixto, Iacer  and
      Gatt, Albert",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.567",
    pages = "8253--8280",
    abstract = "We propose VALSE (Vision And Language Structured Evaluation), a novel benchmark designed for testing general-purpose pretrained vision and language (V{\&}L) models for their visio-linguistic grounding capabilities on specific linguistic phenomena. VALSE offers a suite of six tests covering various linguistic constructs. Solving these requires models to ground linguistic phenomena in the visual modality, allowing more fine-grained evaluations than hitherto possible. We build VALSE using methods that support the construction of valid foils, and report results from evaluating five widely-used V{\&}L models. Our experiments suggest that current models have considerable difficulty addressing most phenomena. Hence, we expect VALSE to serve as an important benchmark to measure future progress of pretrained V{\&}L models from a linguistic perspective, complementing the canonical task-centred V{\&}L evaluations.",
}
```
