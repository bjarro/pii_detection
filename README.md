# Overview

##  Description
Attempt at PII Detection competition in Kaggle

## Reasons for doing PII Detection:
 - Learn / review NLP
 - Learn / experiment with SOTA architectures (BERT, DeBERTa)
 - Learn Hugging Face ecosystem (Transformers, Datasets, Model Hub, Trainer API)

## Output Goals:
 - [x] Create baseline Model from existing PII
	 - [x] Large Model
	 - [ ] Small Model 
 - [x] Fine-tune baseline model
 - [x] Develop pipeline (PII_Util.py)
 - [x] Iterative Improvements
	 - [x] From Large Model
		 - [x] Baseline (0.77) - Best model I can find in Model Hub (Adapted to dataset)
		 - [x] Fine-tuned v0 (0.82) - Last layer unfrozen
		 - [ ] Fine-tuned v1 () - More layers unfrozen
			 - [x] Does not fit in free GPU
		 - [ ] Fine-tuned v1 () - Incorporate new dataset (OpenPII)
			 - [x] In Progress
	 - [ ] From Small Model
		 - [ ] Baseline


## Files:
- PII_Util.py - main code
	 - Adapter class
		 - For adapting different formats/outputs/tokenization of models and datasets
	 - Preprocessing and Postprocessing
		 - Conversions and Alignment
			 - BIO <-> Non-BIO format
			 - Word <-> Subword
			 - Spans -> Labels
		 - Vectorized functions
		 - Positive Thresholding
	 - Trainer API usage
	 - Metrics computation
		 - Confusion Matrix
		 - Precision, Recall
		 - F-beta
	 - Visualizations
		 - Confusion matrix
		 - Class Contributions
	 - Record/Dataset search (to refactor)
 - Notebooks
	 - I - EDA / Datasets
		 - Original Dataset - Main dataset from Kaggle
		 - Polyglot - Only contains short documents and limited labels (general NER use-case only)
		 - OpenPII - Very applicable, although many documents are not in paragraph form (e.g. list of credentials)
	 - III - Pipeline
		 - Identical code, iterative improvements, aimed at different problems/experiments of the current pipeline
		 - Refactored common code to PII_Util.py
	- IV - Results - Results of the current model/approach
	- V - Compare - Comparisons of scores between 2 models/approach
- Docs
	- Notes / scratchpads of initial ideas, questions before studying NER/PII or NLP pipelines and libraries in general

## Pretrained models used
- Large model: https://huggingface.co/Yanis/microsoft-deberta-v3-large_ner_conll2003-anonimization_TRY_1
- Small model:


# Notes

## Questions:
What is the main problem we are trying to solve:
 - Difficult to detect named entities from a list of tokens?
	 - Although looks very simple and seems like can be done algorithmically (using regex), current models struggle with specific cases:
		 - Identifying Names - Lots of False Positives
		 - Distinguishing between similar classes:
			 - Email and Personal URL
			 - Phone Number and ID NUM
		 - class imbalance or lack of exposure to specific classes
			 - Street Address, Username, ID NUM
 - Difficult to identify if named entities are PII or not 
	 - Not all Named Entities are PII. Names in the public domain (e.g., Albert Einstein) are not PII and must be identified via:
		 - contextual analysis?
		 - lookup table (Global or dataset specific)

## Challenges:
- Initial learning curve and familiarity with NLP pipelines:
	- General familiarity
		- Tokenizers and types
		- Special Tokens
		- Attention Mask
		- Error handling and Special cases
			- Token Limit
	- Conversion of inputs and alignment
		- Non-BIO <-> BIO 
		- Word <-> Subword
			- How to distribute words to subwords (reorder or distribute)
				- Most code available online simply distributes -> (B-) labels can be assigned to multiple consecutive subwords
				- Need to test how it affects final score
	- Familiarity with assumptions
		- Consistent tokenizer, formatting
		- 'O' not included in F-Beta score
	- Familiarity with what to measure
- Vectorization and Optimization
	- Custom Metrics and Integration
- Compute availability, efficiency and workflow

## Outputs:
- Baseline model (adapted model without training
- Fine-tuned model (v1)
- Evaluations and visualizations to see which classes needs improvement
	 - Confusion Matrix
	 - Class contributions
	 - Confusion Matrix contributions
		 - Should we prioritize precision or recall
		 - Should we prioritize increasing TP or decreasing FP or FN
 - Vectorized Pipelines:
	- Vectorized Pipeline for conversions of inputs and alignment (preprocessing and postprocessing)
	- Vectorized Pipeline for thresholding (postprocessing)
- Spans to Tokens (Not yet vectorized)
- These enables:
	- quickly adapt other models and datasets regardless of tokenizer used (word or subword) or label formatting (BIO or non-BIO)
	- quickly test and evaluate different pretrained models and datasets
	- quickly evaluate the effects of certain methods on training and validation score
- Model Debugging and Dataset Checking:
	- Word label visualizations
	- Get entries where model failed (to refactor)
	- Get target documents from dataset (to refactor)
		- containing desired labels by density or by count

## Insights:
 - Even though recall is generally more important since beta=5, in our current case raising the precision (decreasing FP) yields a better score.
	 - Each single decrease in FP yields a higher score compared to each single increase in TP
 - Misidentified classes:
	 - B_EMAIL as B_URL_Personal
	 - B_PHONE_NUMBER as B_ID_NUM

## Methods:
 - Stratifying train, validation, test by clustering input dataset
	 - Make sure each set has similar distributions of classes
 - Writing Dataset-Model Adapters
	 - Write mappings:
	 - Test and Set optimal positive threshold
 - Debugging mismatches
	 - Check confusion matrix for mismatch
	 - Query dataset with model predictions for specific mismatch
	 - Visualize, Inspect and document

## Personal Learnings:
 - NLP concepts and architectures (High level)
	 - Encoder / Decoder
	 - BERT, LSTM, etc.
	 - Attention
	 - Vocabulary and tokenization
 - Hugging Face and spacy ecosystem
	 - spacy (nlp, doc, displacy)
	 - transformers (base, datasets, trainer, metrics)
	 - online repository (models, datasets)
 - Metrics
	 - F-Beta (micro averaging)
	 - Custom optimized implementation
 - Types of tokenizers (word, subword)
 - Types of training (pretraining, fine-tuning, transfer learning)
 - Kaggle
	 - Saving and reusing output
		 - How to use datasets and models
	 - Using GPU and TPU
	 - Submissions
 - Pipelines
	 - Querying / Filtering dataset
		 - Problematic cases (where model fails)
	 - Preprocessing and Postprocessing
		 - Formatting and Mappings
		 - Tokenization
			 - word/subword tokenization
			 - Token and label alignment (Reorder vs Distribute)
			 - Handling special tokens
		 - **Positive Thresholding**
		 - Vectorization using numpy
			 - using diff array - where - cumsum - accumulate
			 - grouping - split by cumsum
	 - Training (Practice)
		 - Memory considerations (Model Size, Batch Size, Dataset Size)
		 - Error handling
		 - Runtime considerations
 - NLP Tasks
	 - Named Entity Recognition
	 - PII



