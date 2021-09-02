from flask import Flask, request, render_template
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')   #Import wordnet from the NLTK
nltk.download('averaged_perceptron_tagger')
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def function():
    sentence=request.form['sentence']
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device ",device)
    model = model.to(device)

    sentence.lower()
    text =  "paraphrase: " + sentence + " </s>"
    max_len = 256

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=5 # Number of sentences to return
    )

    para_sents1 = []

    for i,line in enumerate(beam_outputs):
        paraphrase = tokenizer.decode(line,skip_special_tokens=True,clean_up_tokenization_spaces=True)
        para_sents1.append(paraphrase)
        #print(f"{i+1}. {paraphrase}")

    syn = list()
    ant = list()

    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    para_sents = []
    s_temp = sentence

    count = 0
    for t in range(len(tags)) :
        temp = tags[t]
        if temp[1] == "NN" or (temp[1] == "JJ" and temp[0]!= "i"):
            for synset in wordnet.synsets(temp[0]):
                for lemma in synset.lemmas():
                    syn.append(lemma.name())
                    word_to_replace = lemma.name()
                    s = s_temp.replace(temp[0], word_to_replace)
                    para_sents.append(s)
                    s_temp = sentence

    concat_sents = para_sents1 + para_sents     
    final = []
    [final.append(x) for x in concat_sents if x not in final]
    dict={}
    for i in range(len(final)):
        dict[i]=final[i]
    return dict
