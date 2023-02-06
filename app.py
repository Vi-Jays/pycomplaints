import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.utils import pad_sequences

AIModel = 'ai/ngv2.h5'
maxlen = 469
tknzr = 'ai/ngv2_tokenizer.pickle'
if AIModel == 'ai/ngv1.h5':
    maxlen = 452
    tknzr = 'ai/ngv1_tokenizer.pickle'

DB_SCHEMA = 'complaints.db'
SENTS_IMG = 'res/faces-sentiment.png'

st.set_page_config(
    page_title='IPHSENT',
    layout='wide',
)


@st.cache(allow_output_mutation=True)
def aimodel():
    load = load_model(AIModel)
    with open(tknzr, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return load, tokenizer


def preprocessing(text, _maxlen, nzr):
    text = text.lower()

    text = nzr.texts_to_sequences([text])

    text = pad_sequences(text, maxlen=_maxlen)

    return text


#########################################
def create_db():
    conn = sqlite3.connect(DB_SCHEMA)
    c = conn.cursor()

    c.execute(
        '''CREATE TABLE IF NOT EXISTS complaints(id TEXT NOT NULL, reclame TEXT NOT NULL, classificado_como TEXT NOT NULL, grau_de_confianca TEXT NOT NULL)''')

    conn.commit()
    conn.close()


def submit_r(_id, reclame, classif, grau):
    conn = sqlite3.connect(DB_SCHEMA)
    c = conn.cursor()

    c.execute("INSERT INTO complaints(id, reclame, classificado_como, grau_de_confianca) VALUES (?,?,?,?)",
              (_id, reclame, classif, grau))

    conn.commit()
    conn.close()


def fetch_data():
    conn = sqlite3.connect(DB_SCHEMA)
    c = conn.cursor()

    c.execute(
        "SELECT id, reclame as 'Texto', classificado_como as 'Sentimento', grau_de_confianca as 'Confiança' FROM complaints")
    complaints = c.fetchall()

    conn.close()

    return complaints


#########################################


def iniciar_page():
    imgs = Image.open(SENTS_IMG)

    with st.container():
        st.write("""<h3>O que é IPHSENT ?</h3>
    <p>Trata-se de um protótipo funcional que 
    realiza a <b>classificação de polaridades 
    (sentimentos positivos e negativos)</b> 
    contidas num <b>texto</b>, graças ao método de PLN 
    (Processamento de Linguagem Natural), 
    sub-área da IA (Inteligência Artificial).</p>
    <p>Análise de sentimento, também conhecida como 
    mineração de opiniões, é um campo de estudo dentro do 
    PLN (Processamento de Linguagem Natural) que se 
    concentra em identificar e categorizar 
    informações subjetivas em texto escrito. 
    O objetivo é determinar a atitude, emoção ou 
    opinião do escritor em relação a um determinado 
    assunto ou entidade. Isso é conseguido usando 
    uma combinação de algoritmos de aprendizado de 
    máquina, dicionários e outras técnicas de PLN 
    para analisar dados de texto e extrair 
    informações relevantes. O resultado final é 
    frequentemente uma pontuação de sentimento, 
    que varia de negativo a positivo, e pode ser usado 
    em várias aplicações, como atendimento ao 
    cliente, marketing e monitoramento de mídias sociais.</p>
    """, unsafe_allow_html=True)

        st.image(imgs, width=360)


def reclames_page():
    _, md, _ = st.columns([2, 15, 2])
    complaints = fetch_data()

    df = pd.DataFrame(complaints, columns=['Autor', 'Reclame', 'Sentimento', 'Confiança'])
    with md:
        st.write(df)


def posts_page(ml, tk):
    with st.form('sbmt_form', clear_on_submit=True):
        sigu_id = st.text_input("ID SIGU:", max_chars=10)
        reclame = st.text_area("Escreva aqui", max_chars=100)

        btn = st.form_submit_button("Enviar")

    if btn:
        if sigu_id != "" and reclame != "" and len(sigu_id) == 10 and len(reclame) > 4:
            with st.spinner("A processar..."):
                ####################################################
                text = preprocessing(reclame, maxlen, tk)
                yproba = ml.predict(text)
                ypred = np.argmax(yproba, axis=1)

                if ypred == 1:
                    classif = 'Positivo'
                    conf = "{:.2f}".format(yproba[0][1] * 100)
                    conf = str(conf)

                else:
                    classif = 'Negativo'
                    conf = "{:.2f}".format(yproba[0][1] * 100)
                    conf = str(conf)
                ####################################################
                submit_r(sigu_id, reclame, classif, conf)

                st.success("Enviado com sucesso!")
        else:
            st.error("Deve preencher os campos devidamente.")


def main():
    model, tokenizer = aimodel()
    st.title("IPHSENT", anchor=None)
    _, md_col, _ = st.columns([3, 3, 3], gap="small")

    with md_col:
        page = st.selectbox("Menu:", ["Iniciar", "Postar", "Resultados"])

    if page == "Iniciar":
        iniciar_page()
    elif page == "Resultados":
        reclames_page()
    else:
        with md_col:
            posts_page(model, tokenizer)


if __name__ == "__main__":
    if not os.path.exists(DB_SCHEMA):
        create_db()

    main()
