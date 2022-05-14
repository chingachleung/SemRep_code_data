
import spacy
import amrlib
amrlib.setup_spacy_extension()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
from scipy import stats

"""
def retrieve_fn_from_lu (word):

    #fns = fn.lus(word)
    # get frame with lus matching the lemma given
    #nltk.download('framenet_v17')
    from nltk.corpus import framenet as fn
    fns = fn.frames_by_lemma(word)

    for f in fns:
        #print(fn)
        #print("\n")
        # return the keys(lus) from a framenet
        id = f.ID
        frame = f.name
        lus = f.lexUnit.keys()
        fes = f.FE.keys()
        print(id, frame, lus,fes)
        print("\n")

    return fns
"""

def amr_parsing(sentences):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentences)
    graphs = doc._.to_amr()
    with open("amrs_test.txt", 'a') as file:
        print("lengths of the graphs:",len(graphs))
        if len(graphs) > 1:
            print("exceeded one")
            return False
        else:
            for graph in graphs:
                file.write(graph)
                file.write("\n")
                file.write("\n")

def amr_parsing2(sentences):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentences)
    graphs = doc._.to_amr()
    with open("amrs2.txt", 'a') as file:
        if len(graphs) > 1:
            return False
        else:
            for graph in graphs:
                file.write(graph)
                file.write("\n")
                file.write("\n")

def get_sent_embeddings(sent):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sent, convert_to_tensor=True)
    return embeddings

def process_pairs(file):
    with open(file,'r') as data:
        with open("gold_labels.txt",'w') as newf:
            with open("embeddings_similarities.txt", "w") as newf2:
                for line in data:
                    fields = line.strip().split('\t')
                    newf.write(fields[0])
                    newf.write("\n")
                    amr_parsing(fields[1])
                    amr_parsing2(fields[2])
                    emb1 = get_sent_embeddings(fields[1])
                    emb2 = get_sent_embeddings((fields[2]))
                    newf2.write(get_cosine_similarity(emb1,emb2))
                    newf2.write("\n")
                    
def get_cosine_similarity(sent1, sent2):
    similarity = cosine_similarity(sent1.reshape(1,-1), sent2.reshape(1,-1))
    return str(similarity[0][0])

def get_lines_with_labels(file):
    with open (file,'r') as data:
        with open("sts_data_new.tsv",'a') as new_file:
            for line in data:
                fields = line.strip().split('\t')
                if len(fields)==3:
                    new_file.write(line)

def test_sts_file(file):
    with open(file,'r') as data:
        labels = []
        sent1 = []
        sent2 = []
        for line in data:
            fields = line.strip().split("\t")
            labels.append(fields[0])
            sent1.append(fields[1].strip())
            sent2.append(fields[2].strip())

    return labels, sent1, sent2
def test_amr_parsing(sentence):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    graphs = doc._.to_amr()
    for graph in graphs:
        print(graph)
        print("#####")

def test_amr_file(file):
    with open(file, 'r') as data:
        amrs = []
        for line in data:
            if "# ::snt" in line:
                amrs.append(line[8:].strip())
    return amrs

def master_amr_parsing(sent1, sent2):
    nlp = spacy.load('en_core_web_sm')
    doc1=nlp(sent1)
    doc2=nlp(sent2)
    graphs1 = doc1._.to_amr()
    graphs2 = doc2._.to_amr()
    with open("AMR1_f.txt", 'a') as file:
        with open("AMR2_f.txt", 'a') as file2:
            if len(graphs1) != 1 or len(graphs2) != 1:
                return False
            else:
                for graph in graphs1:
                    file.write(graph)
                    file.write("\n")
                    file.write("\n")
                for g in graphs2:
                    file2.write(g)
                    file2.write("\n")
                    file2.write("\n")

def process_pair2(file):
    with open(file, 'r') as data:
        with open("gold_labels4.txt", 'a') as newf:
            with open("embeddings_similarities4.txt", "a") as newf2:
                for i,line in enumerate(data):
                    print("current line:", i)
                    fields = line.strip().split('\t')
                    parse_result = master_amr_parsing(fields[1], fields[2])
                    if parse_result == False:
                        continue
                    else:
                        newf.write(fields[0])
                        newf.write("\n")
                        emb1 = get_sent_embeddings(fields[1])
                        emb2 = get_sent_embeddings((fields[2]))
                        newf2.write(get_cosine_similarity(emb1, emb2))
                        newf2.write("\n")

    #print("lenght of the correct parse: ", correctly_parse)

def get_final_sentences(file):
    with open(file, 'r') as data:
        with open("amr2_texts.txt","w") as newf:
            for line in data:
                if "# ::snt" not in line:
                    continue
                else:
                    newf.write(line[8:])
def get_bert_score():
    from bert_score import score
    import tensorflow as tf
    with open("amr1_texts.txt","r") as f1:
        cands = [line.strip() for line in f1]
    with open("amr2_texts.txt","r") as f2:
        refs = [line.strip() for line in f2]

    P, R, F1 = score(cands, refs, lang='en', verbose=True)
    tf.print(F1)
    with open("bertscore_scores.txt",'w') as file:
        for f in F1:
            file.write(str(f.item()))
            file.write("\n")

def get_correlation_score(gold, pred):
    with open(gold, 'r') as file:
        with open(pred, 'r') as file2:
            gold_labels = [float(i) for i in file]
            pred_labels = [float(i) for i in file2]
    #print(gold_labels, "\n", pred_labels)
    result = stats.spearmanr(gold_labels, pred_labels)
    print(result)

def create_dataframe():
    #covert list of scoress into a data frame
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    with open("smatch_scores.txt",'r') as data:
        smatch = [float(i) for i in data]
    with open("s2match_scores.txt","r") as data2:
        s2match = [float(i) for i in data2]
    with open("embeddings_similarities4.txt","r") as data3:
        sentbert = [float(i) for i in data3]
    with open("bertscore_scores.txt",'r') as data4:
        bertscore = [float(i) for i in data4]

    # load the amr sentences into two lists
    with open("amr1_texts.txt", 'r') as file1:
        amrs1 = [line for line in file1]
    with open("amr2_texts.txt", 'r') as file2:
        amrs2 = [line for line in file2]

    with open("gold_labels4.txt",'r') as file3:
        labels = [l for l in file3]

    data = pd.DataFrame(list(zip(s2match, sentbert, bertscore)), columns = ["s2match", "sentbert", "bertscore"])
    s2match_scaler = StandardScaler()
    sentbert_scaler = StandardScaler()
    bertscore_scaler = StandardScaler()

    data["s2match"] =s2match_scaler.fit_transform(data['s2match'].values.reshape(-1,1))
    data["sentbert"] = sentbert_scaler.fit_transform(data['sentbert'].values.reshape(-1, 1))
    data["bertscore"] = bertscore_scaler.fit_transform(data['bertscore'].values.reshape(-1, 1))

    deviations = data.std(axis=1)
    # print(deviations)
    # print(len(deviations))
    std_dict = {}
    for i, item in enumerate(deviations):
        std_dict[item] = i

    # reoder the dict from the highest to lowest STD sentence pairs
    new_dict = dict(sorted(std_dict.items(), key=lambda item: item[0], reverse=True))

    highstd = list(new_dict.items())[:100]
    lowstd = list(new_dict.items())[-100:]
    #extremes = highstd + lowstd



    final_all_data = pd.DataFrame(list(zip(amrs1, amrs2, labels, s2match,sentbert,
                                          bertscore,data["s2match"],data["sentbert"],data["bertscore"],deviations
                                          )), columns = ["sent1", "sent2", "labels",
                                                                                   "s2match", "sentbert", "bertscore",
                                                                                   "scaled_s2match", "scaled_sentbert",
                                                                                   "scaled_bertscore","standard deviation"])
    #covert to a csv file:
    final_all_data.to_csv("final_all_data.csv")

    # write the high std and low std sentences into two files separately


def get_statistics(file) :
    from scipy import stats
    with open(file,'r') as data:
        newlist =[float(l) for l in data]

    print(stats.describe(newlist))
