import torch

def sample_translation(encoder, decoder, dataset, sentence, device):
    model_input = torch.cat([dataset.eng2embed[word][:300].unsqueeze(0) for word in sentence.split()], dim=0).unsqueeze(0)
    encoded_source = encoder(model_input.to(device))
    yhat = decoder(encoded_source)[0].argmax(dim=1)
    output = " ".join([dataset.idx2spa[idx.item()] for idx in yhat])
    return output

def init_weights(model):
    for p in model.parameters():
        torch.nn.init.normal_(p, 0, 0.02)