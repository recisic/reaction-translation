# Reaction Translator

Translating 'reagents and reactants' to 'products'.

## Paper

[ArXiv] [Linking the Neural Machine Translation and the Prediction of Organic Chemistry Reactions](http://arxiv.org/abs/1612.09529)

Finding the main product of a chemical reaction is one of the important problems of organic chemistry. This paper describes a method of applying a neural machine translation model to the prediction of organic chemical reactions. In order to translate 'reactants and reagents' to 'products', a gated recurrent unit based sequence-to-sequence model and a parser to generate input tokens for model from reaction SMILES strings were built. Training sets are composed of reactions from the patent databases, and reactions manually generated applying the elementary reactions in an organic chemistry textbook of Wade. The trained models were tested by examples and problems in the textbook. The prediction process does not need manual encoding of rules (e.g., SMARTS transformations) to predict products, hence it only needs sufficient training reaction sets to learn new types of reactions.

## Links
The code in this repository is partly based on:
* [Reaction Fingerprints](https://github.com/jnwei/neural_reaction_fingerprint)
* [TensorFlow's Translate Model](https://github.com/tensorflow/tensorflow/tree/v0.10.0/tensorflow/models/rnn/translate)
* [SMILES Railroad Diagram](https://metamolecular.com/cheminformatics/smiles/railroad-diagram/index.xhtml)
