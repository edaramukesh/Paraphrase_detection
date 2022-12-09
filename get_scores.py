from sentence_transformers import SentenceTransformer
import numpy as np
import torch


questionsList = ["What are credits?",
"How many credits do we need to complete in our degree?",
"Minimum number of credits per semester?",
"Maximum number of credits per semester?",
"Recommended number of Credits from 2nd semester",
"What are course baskets?",
"How many course baskets are there?",
"What are minimum credit requiremenet for each basket?",
"Do these baskets have Categories?",
"What are the Categories of University Core(UC)?",
"What are the Categories of Program Core(PC)?",
"What are the Categories of Program elective(PE)?",
"What are the Categories of Program elective(PE)?",
"Are there minimum credit requirement for each category?",
"Can you do more than 160 credits?.",
"Will they charge for taking extra credits? ",
"What is the charged amount per extra credit?",
"What are the minimum number of credits to be selected in Engineering Clinics Basket?",
"What are the minimum number of credits to be selected in Engineering Basket?",
"What are the minimum number of credits to be selected in English Basket?",
"What are the minimum number of credits to be selected in Humanities Basket?",
"What are the minimum number of credits to be selected in Soft skills Basket?",
"What are the minimum number of credits to be selected in Management Basket? ",
"What are the minimum number of credits to be selected in Science Basket?",
"What is the fee structure for b.tech?",
"What is the hostel fee structure ?",
"What are the degree programs offered in Vit?",
"What are the courses offered in Engineering Programmes?",
"What are the courses offered in UG Programmes?",
"What are the courses offered in Integrated Programmes?",
"What are the courses offered in PG Programmes?",
"What are the courses offered in Research Programmes?"]

def loadModel():
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    # print(model.device)
    return model

def getEmbeddings(path = "sentence_vectors.npy"):
    embeddings = np.load(path)
    return embeddings

def createEmbeds(model,sentences):
    embeddings = model.encode(sentences)
    np.save('sentence_vectors',embeddings)

def compareSimilarity(model,embeddings,question):
    q_embed = model.encode([question])
    similarity_list = []
    for idx,i in enumerate(embeddings):
        # print(f"{sentences[idx]}: ",np.linalg.norm(i-q_embed))
        similarity_list.append(np.linalg.norm(i-q_embed))
    return similarity_list


if __name__ == "__main__":
    """
    # Add questions to sentence_vectors.pny 
    
    model = loadModel()
    with open("questions.txt","r") as f:
        ques = f.readlines()
    sentences = ["'"+i.strip()+"'" for i in ques]
    createEmbeds(model,sentences)

    """

    # """
    
    # inference

    model = loadModel()
    embeddings = getEmbeddings()
    question = input("Enter the question: ")
    while True:
        similarityList = compareSimilarity(model,embeddings,question)
        min_sim = min(similarityList)
        if min_sim<1.2:
            print(questionsList[similarityList.index(min_sim)])
        else:
            print("Didn't match boii.")
        question = input("Enter the question: ")
        if question.strip() == "" or question.strip() == "bye":
            break

    # """


