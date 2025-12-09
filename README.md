# TuringTestProject
CS4100 Project

Christian Larouche, Shourya Ravula

# abstract
The project challenges whether modern ML systems can effectively determine the difference between human and AI generated text. In our 2 part pipeline, we combined a TF-IDF cosine similarity search engine with a fine-tuned transformer classifier. Our TF-IDF part finds the most similar texts in its dataset to the entered text, effectively identifying style and alphabetic patterns between the two. Our transformer part uses supervised classification through context embeddings to find similar patterns. Results show our transformer model has great accuracy on the evaluation set, (with issues with overfitting), overall this suggests that linguistic changes between man and machine are detectable.

# Overview
## what is the problem?
AI is one of the hottest fields right now, with LLM's like gpt-5 and Claude Sonnet being used everywhere from classrooms to offices. Those same LLMs have allowed machines to make text that is almost identical to human writings. This raises major concerns for plagiarism, misinformation, and bot detection on social media. While many ai detection services already exist--like GPT Zero and Zero GPT--many of them report false positives, inaccuracies, and lack insight into the psychological and linguistic components at play. Current detectors are unreliable, stunned by parapgrasing, and flip flop based on a few word changes. We hope to learn whether further computer models, using classical IR and modern deep learning, can tell man from machine.

## why is it interesting
Our project offers a horse to the Turing Test arms race. We need to keep up with LLM models as they become more life like, in order to maintain honesty. AI work is increasing relied on in schools and work, journalism depends on verified reports and authentic figures, and understanding the features the go into language helps combat this increase while understanding the factors that inform it. 

## proposed approach
We use a 2 part pipeline to detect AI texts through a TF-IDK search engine and a fine tuned transformer model. The TF-IDF retrieves similar texts through cosine similarity from a labelled AI-Human dataset. This allows semantic signals to be interpretable through nearest neighbors and term importance. Our transformer classifies text as AI and human using token level context embeddings and learned style and structure differences. Our interpreter then explains the classification to the user using natural language and evidence from the other model. 

## rationale and related work
Most free AI detectors achieve high levels of accuracy using lexical, sentence level, and punctuation rules, despite failing when parapgrasing is used. We use retrieval and contextual classification to complement these strengths and offer a hyrbid design.

## key components and limitations
We use TF-IDF tokenizer, cosine similarty search, document level transformer classifier, explanation generation, and visualization tools for similarity and features. Our dataset is limited to a single, small corpus (<5,0000 essays). We don't test paraphrasing ai texts, ourtransformer is smaller, and out results arent yet generalizable to real world settings (ai sentence vs essay vs book).

# approach
## methodology overview
Our TF-IDF tokenizes text and regex-based word extractor, computers TF, DF and IDF, doess cosine similarity search, then returns top-k nearest neighbors with labels and scores of similarity. Our transformer model using uses a tokenizer to cinvert text into token IDs, trains using supervised labels of AI vs Human, and then has linear classification on top of context embeds. 

## design choices
We used a balanced dataset over a deeper dataset to avoid label skew, we had a GPU accerlation to reduce transformer training time, and our tokenizer perserved basic linguistic components and grammer. 

## assumptions and limitations
Our models assume textlength is long (at least long enough for stlistic features to show). It also assumes our balanced dataset is indicative of real world scenarios and proportions. TF-IDF cant gain long term context, our transformer severelly overfits as we ran out of time to establish regularization or random forresting. We also didn't implement defenses for parapphrasing. 

# experiments
## dataset
Our balanced_ai_human_prompts.csv used a balanced distribution of human and ai essays, responses, and paragraphs. the columns were text, generated?. Mean length of texts is 1670.9, vocab size is 31,788, and number of entries is 2750

## implementation details
we used custom TF-IDF search engine from the starter code, a distilbert base uncased transformer model loaded through huggingFace transformers and a qwen2.5 transformer classifier, then we used a minimal interpretability LM to communicate the reasons for classification. We trained on learning_rate=2e-5, num_train_epochs=2, weight_decay=0.01. We used a Google colab environment with CUDA enabled PyTorch, huggingface transformers and data, in a python 3.1 environment. Our model architecture was a base transformer encoder, mean pooling/cls token, and then a linear classifier layer for binary outputs. 


# results
## main
eval_loss: 9.09e-05, eval_accuracy: 1.0, eval_f1: 1.0 
This indicates overfitting on the dataset. Our data seeimingly identifies the AI Human data perfectly. However, we also found that our TF-IDF was also successful at determining human and ai text. 

## supplementary
Our TF-IDF determines AI text though a template of cosine similarity. Our transformers catch hidden indicators by looking at over consistent sentence structures, inconsistent prose, lack of semantic variation, and a lack of individual phrasings and niche terms. Our parameters allow for high accuracy due to data simplemness. 

# discussion
Compared to other detectors like GPTZero, originality.ai, or zero GPT, our models provide interpretable reasoning, has transparent processes, and achieves strong accuracy on specific data. We are limited in our generality as we lack the tools to train on giant datasets. Potential issues thus arise, as we risk overfitting to short, clean data and lack the robustness of paraphrasing or other plagiarism tactics. In the future, we could train on adversarial situations, add stylometry, use deep retrieval or internet search, expand dataset to LONG essays, short social media comments, and news articles. We could also try perplecity based signaling. 


# conclusion 
We made a viable alternative to current AI detectors and threw our hats in the ring for the Turing Test. We integrated information retriveal with modern transformer modelling, allowing for the interpreyable retrieval system to combine with modern classification that achieves great accuracy. Limitation relent, especially given the use of parapgrasing AI text to provide life like text. However, we have overall showed that ML can distinguisg human writing from AI generations in specific conditions. Hopefully this adherence to academic integrity and authenticity continues in the future, as ai becomes more advanced and people are greater incentivized to take the easy way.


# references
Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense — Krishna et al. (2023) 
arXiv

Distinguishing AI-Generated and Human-Written Text Through Psycholinguistic Analysis — Opara et al. (2025) 
Emergent Mind

A Lightweight Approach to Detection of AI-Generated Texts Using Stylometric Features — Aityan et al. (2025) 
arXiv

Testing of detection tools for AI-generated text — Weber-Wulf et al. (2023) SpringerLink













