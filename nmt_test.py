import torch 
import nmt 

#load all saved variables 

hidden_size = 256
encoder1 = nmt.EncoderRNN(nmt.input_lang.n_words, hidden_size).to(device)
attn_decoder1 = nmt.AttnDecoderRNN(hidden_size, nmt.output_lang.n_words, dropout_p=0.1).to(device)
encoder1.load_state_dict(torch.load('data/encoder.dict'))
attn_decoder1.load_state_dict(torch.load('data/decoder.dict'))

nmt.evaluateAndShowAttention("elle a cinq ans de moins que moi ." ,encoder1, attn_decoder1)

nmt.evaluateAndShowAttention("elle est trop petit ." ,encoder1, attn_decoder1)

nmt.evaluateAndShowAttention("je ne crains pas de mourir ." ,encoder1, attn_decoder1)

nmt.evaluateAndShowAttention("c est un jeune directeur plein de talent ." ,encoder1, attn_decoder1)
