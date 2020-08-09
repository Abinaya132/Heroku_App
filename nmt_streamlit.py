import streamlit as st
from googletrans import Translator
from vaderSentiment.vaderSentiment import  SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer() # initialize sentiment analyzer 
translator = Translator() # initialize

st.markdown('# DeathNote')

st.markdown('## Part 1: Sentimental Analysis')

st.markdown('## Part 2: French to English ')

st.write('Sentimental Analysis  and French to English App ')


# The sentiment analyzer function
def sentiment_analyzer_scores(sentence):
   trans = translator.translate(sentence).text # extracting   translation text
   score = analyser.polarity_scores(trans) # analyzing the text
   score = score['compound']
   if score >= 0.05:
      return 'The sentiment of your text is Positive'
   elif score > -0.5 and score < 0.05:
      return 'The sentiment of your text is Neutral'
   else:
      return 'The sentiment of your text is Negative'
   return score


sentence = st.text_area('Write your sentence') # we take user inputif st.button(‘Submit’): # a button for submitting the form
result = sentiment_analyzer_scores(sentence) # run our function  on it
st.balloons() # show some cool animation
st.success(result) # show result in a Bootstrap panel

####################### NMT PART ##############################

import torch 
import nmt 

hidden_size = 256
encoder1 = nmt.EncoderRNN(nmt.input_lang.n_words, hidden_size).to(nmt.device)
attn_decoder1 = nmt.AttnDecoderRNN(hidden_size, nmt.output_lang.n_words, dropout_p=0.1).to(nmt.device)
encoder1.load_state_dict(torch.load('data/encoder.dict'))
attn_decoder1.load_state_dict(torch.load('data/decoder.dict'))

sentence = st.text_area('Write your sentence in French ') # we take user inputif st.button(‘Submit’): # a button for submitting the form

res_nmt , plt = nmt.evaluateAndShowAttention(sentence ,encoder1, attn_decoder1)

st.write('The corresponding English Text is : ')
st.success(res_nmt) # show result in a Bootstrap panel

st.write('The Attention scores for the translation is : ')
st.pyplot()