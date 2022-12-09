# Paraphrase_detection
Install the model from https://huggingface.co/sentence-transformers/all-mpnet-base-v2,
preferably in a virtualenv.

I get the embeddings of questions and save it in sentence_vectors.npy.
I used l2 norm as a measure of similarity.
I used a threshold of 1.2 to classify if the paraphrased question is comparable to given questions or a general question.

