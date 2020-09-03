# Just-News

The aim of the project is to analyze the language used by famous american newspapers using distributional methods.

Our research focuses on the use of *subjective language*, opposed to a more *neutral*\* and distant writing style.

In other words, we are trying to determine which newspapers tend to rely on emotions and morals, rather than merely reporting facts.

## Data
We conduct our research on the following newspapers:
* New York Times
* CNN 
* ABC News
* Breitbart
* Slate
* The Federalist
* NewsMax

Our dataset consists of around 40000 articles, plus ~20000 Wikipedia pages, that we use as *control corpus*, under the hypothesis that it uses neutral language.

## Phases 
Here are presented the phases of our work.
* **Data retrieval.** Consists in getting the data from various sources (mainly web-scraping). The code related to this phase is not included in the repo. *A link shall be added once some clean up process has been executed*.
* **Data cleaning and preprocessing.** Consist of creating a dataset suitable for analysis. (CADE_testing.ipynb).
* **CADE training.** We make use of [CADE](https://github.com/vinid/cade) in order to obtain aligned word-embeddings, one for each newspaper. (CADE_testing.ipynb).
* **Score induction.** In order to obtain a *subjectivity score* for the words in our vocabulary, we apply a score induction procedure. This procedure allow us to propagate the score coming from a an [annotated lexicon](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/) to all the words in the vocabulary.
* **Score propagation.** How do we determine the subjectivity score of an entire article, starting from the score of the words in it? This process will require some manual *benchmarking*. 
* **Final results and visualization.**
* **(Extra)** Following the indications contained in [this article](https://deepai.org/publication/analytical-methods-for-interpretable-ultradense-word-embeddings), we could push on the the aspect of interpretability of results.

### Modifications to SWEAT
* `enrich` and `lexicon_refinement` inside `utils.py` need to be updated to be used in more than 2 models.
* Also, we must try to relax some of the constraints in the `lexicon refinement` function. For example, we might set a lower value of `zipf_cutoff`. Another option is to set `topn = n` inside the building of the comparative framework, thus maintaining the n nearest across the various representations, instead of just one.

## Note
\* We are aware of the fact that there is no such thing as a *neutral way* of reporting news, and that every journal and journalist tend to have its own views. Our research, in fact, focuses on the use of language only.

