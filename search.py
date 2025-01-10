import torch

from models import Full_Model
from tokenizer import LaTeXTokenizer
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import LogSoftmax
from torch.nn.utils.rnn import pad_sequence


device = 'cuda'
tokenizer = LaTeXTokenizer()
tokenizer.id_to_token = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[UNK]', 4: '{', 5: '}', 6: '_', 7: '1', 8: ')', 9: '(', 10: '^', 11: '2', 12: '=', 13: '\\frac', 14: 'x', 15: '0', 16: '-', 17: 'i', 18: ',', 19: 'n', 20: 't', 21: 'a', 22: '+', 23: '3', 24: 'r', 25: '.', 26: 'm', 27: 'd', 28: '4', 29: '5', 30: '6', 31: 'k', 32: 's', 33: '7', 34: '8', 35: '9', 36: 'c', 37: 'p', 38: 'e', 39: '|', 40: 'f', 41: 'y', 42: '\\cdot', 43: '\\sqrt', 44: 'A', 45: 'b', 46: 'l', 47: 'o', 48: '[', 49: ']', 50: 'g', 51: '\\partial', 52: 'z', 53: 'j', 54: '/', 55: 'v', 56: 'X', 57: 'T', 58: 'R', 59: '&', 60: 'u', 61: 'P', 62: '\\\\', 63: 'S', 64: 'B', 65: 'C', 66: 'E', 67: '\\prime', 68: 'N', 69: '\\begin', 70: '\\end', 71: '\\alpha', 72: '\\int', 73: 'F', 74: 'V', 75: 'q', 76: '\\pi', 77: 'L', 78: 'h', 79: '\\sum', 80: 'M', 81: '\\theta', 82: '\\mu', 83: '\\in', 84: 'H', 85: 'I', 86: '\\hat', 87: 'D', 88: '\\rightarrow', 89: 'G', 90: '\\lambda', 91: '*', 92: '\\sigma', 93: '\\{', 94: ':', 95: '\\}', 96: 'w', 97: '\\overline', 98: '\\infty', 99: '\\mathbb', 100: 'K', 101: '\\beta', 102: '\\omega', 103: '\\rho', 104: '\\epsilon', 105: 'Z', 106: 'Q', 107: 'Y', 108: '\\le', 109: '\\gamma', 110: 'U', 111: '\\times', 112: '\\phi', 113: '\\rangle', 114: '\\Delta', 115: '<', 116: '\\tilde', 117: '\\psi', 118: '\\delta', 119: '\\nu', 120: 'W', 121: '\\tau', 122: 'O', 123: '\\varphi', 124: 'J', 125: '\\langle', 126: '\\vec', 127: '!', 128: '>', 129: '\\nabla', 130: '\\ge', 131: '\\prod', 132: '\\Omega', 133: ';', 134: '\\eta', 135: '\\Gamma', 136: '\\approx', 137: '\\xi', 138: '\\Phi', 139: '\\dot', 140: '\\pm', 141: '\\otimes', 142: '\\circ', 143: '\\wedge', 144: '\\equiv', 145: '\\hbar', 146: '\\chi', 147: '\\underline', 148: '\\zeta', 149: '\\kappa', 150: '\\ne', 151: '\\forall', 152: '\\Sigma', 153: '\\sim', 154: '\\subseteq', 155: '\\Psi', 156: '\\notin', 157: '\\cap', 158: '\\Lambda', 159: '\\mapsto', 160: '\\neg', 161: '\\cup', 162: '\\oplus', 163: '\\Rightarrow', 164: '\\dagger', 165: '\\vee', 166: '\\subset', 167: '\\backslash', 168: '\\rfloor', 169: '\\Pi', 170: '\\lfloor', 171: '\\|', 172: '\\Theta', 173: '\\exists', 174: '\\cong', 175: '\\emptyset', 176: '\\propto', 177: '\\perp', 178: '\\vdash', 179: '\\bigcup', 180: '\\bullet', 181: '\\simeq', 182: '\\leftarrow', 183: '\\aleph', 184: '\\%', 185: '\\vartheta', 186: '\\ll', 187: '\\#', 188: '\\oint', 189: '\\angle', 190: '\\top', 191: '\\leftrightarrow', 192: '\\bigoplus', 193: '\\iint', 194: '\\bigcap', 195: '\\vdots', 196: '\\lceil', 197: '\\rceil', 198: '\\iff', 199: '\\gg', 200: '\\odot', 201: '\\varpi', 202: '\\Leftrightarrow', 203: '\\models', 204: '\\longrightarrow', 205: '\\ominus', 206: '\\iota', 207: '?', 208: '\\upsilon', 209: '\\mp', 210: '\\Xi', 211: '\\varsigma', 212: '\\hookrightarrow', 213: '\\supseteq', 214: '\\supset', 215: '\\subsetneq', 216: '\\triangleq', 217: '\\bigwedge', 218: '\\div', 219: '\\Upsilon', 220: '\\Vdash', 221: '\\&', 222: '\\bigvee', 223: '\\ni', 224: '\\rightleftharpoons', 225: '\\triangle', 226: '\\_', 227: '\\not', 228: '\\bigcirc', 229: '\\sqsubseteq', 230: '\\triangleleft'}
model = Full_Model(vocab_size=231, d_model=256, nhead=8, dim_FF=1024, dropout=0, num_layers=3).to(device)
model.load_state_dict(torch.load(r'runs\Exp9E2.pt', map_location=device, weights_only=True))
transform = transforms.Compose([transforms.Resize((512, 384)),transforms.ToTensor()])


# input is a PIL image, returns beam_width amount of both L2R and R2L sequences
def beam_search(input, model, tokenizer, transform, device, beam_width, alpha, end_token):
    with torch.no_grad():
        vocab_size = len(tokenizer.id_to_token)
        logSM = LogSoftmax(dim=-1)
        features = model.encoder(transform(input).unsqueeze(0).to(device)).detach()
        features_batch = features.repeat_interleave(beam_width, dim=0) # From size [1, 192, 256] to [beam_width, 192, 256]
        tgt_mask = torch.triu(torch.ones(200, 200) * float("-inf"), diagonal=1).to(device)

        start_token = 1 if end_token == 2 else 2
        beams = torch.ones([1, 1], dtype=torch.long).to(device) * start_token
    
        # Root expansion step
        output = model.decoder(features, beams, tgt_mask[:1, :1])
        beam_scores, indices = torch.topk(logSM(output), beam_width, dim=-1) # Size [1, 1, beam_width]
        beams = torch.cat([beams.repeat_interleave(beam_width, dim=1).unsqueeze(2), indices.permute(0, 2, 1)], dim=-1) # [1, beam_width, 2]
        beam_scores = beam_scores.view(beam_width, 1) # Fits [beam_width, vocab] shape in for loop

        completed_beams = torch.zeros((beam_width, 200), dtype=torch.long).to(device)
        completed_probs = torch.zeros((beam_width), dtype=torch.float32).to(device)
        start_slice = 0 # For completed beams
        for i in range(2, 200):
            tgt = beams.squeeze()
            output = model.decoder(features_batch, tgt, tgt_mask[:i, :i])

            # Get updated probabilities and store as beam_width amount of beam scores
            probs = logSM(output[:, -1, :]) # [beam_width, vocab]
            probs += beam_scores # [beam_width, vocab] + [beam_width, 1] 

            # Freeze beams and store final values if needed
            # COULD DO: .nonzero(asTuple=True) below to clean up conditionals here
            if i > 2 and (end_indices.dim() == 0 or (end_indices.dim() == 1 and end_indices.shape[0] > 1)): # Accounts for single scalar index and range of indices (none true means empty tensor with dim=1)
                size = int(end_indices) if end_indices.dim() == 0 else end_indices.shape[0]
                end_slice = start_slice+size  
                end_beams = torch.index_select(beams, dim=1, index=end_indices).squeeze() # [beam_width, current_length]
                end_probs = beam_scores.squeeze()[end_indices] 
                if end_slice >= beam_width: 
                    end_slice = beam_width
                    end_beams = end_beams[0:(beam_width-(start_slice)), :]
                    end_probs = end_probs[0:(beam_width-(start_slice))] 

                completed_beams[start_slice:end_slice, :i] = end_beams
                completed_probs[start_slice:end_slice] = end_probs * (1 / (i**alpha))
                probs[end_mask, :] = -torch.inf 
                if end_slice >= beam_width:
                    break
                else:
                    start_slice += size

            # Update beams with next token - since probabilities are flattened, need to find corresponding beam and its next index w.r.t original vocab
            beam_scores, indices = torch.topk(probs.view(beam_width*vocab_size, 1), beam_width, dim=0) # Flattened
            # Need permute to change from [beam_width, 1, len] to [1, beam_width, len]
            beams = torch.cat([beams[torch.arange(1).unsqueeze(-1), indices//vocab_size], (indices%vocab_size).unsqueeze(2)], dim=-1).permute(1, 0, 2) 

            # Check if there was <EOS> or <BOS> generated
            end_mask = (beams[0, :, -1] == end_token).squeeze()
            end_indices = torch.nonzero(end_mask).squeeze()             

        return completed_beams, completed_probs 


def reverse_beams(tensor):
    result = tensor.clone()  # Clone the tensor to avoid modifying the original
    for i in range(tensor.size(0)):  # Iterate over each row
        non_zero_indices = tensor[i].nonzero(as_tuple=True)[0]  # Get non-zero indices
        non_zero_values = tensor[i][non_zero_indices]  # Extract non-zero values
        reversed_values = non_zero_values.flip(0)  # Reverse the non-zero values
        result[i][non_zero_indices] = reversed_values  # Place them back
    return result


# TO DO: maintain probabilities for each token (rather than sum at each step)
# This will allow use of CE loss here instead of nested python loop
# Also will get rid of parameterized b factor
def bidirectional_search(input, model, tokenizer, transform, device, beam_width, alpha, b):
    L2R_beams, L2R_probs = beam_search(input, model, tokenizer, transform, device, beam_width, alpha, 2) 
    R2L_beams, R2L_probs = beam_search(input, model, tokenizer, transform, device, beam_width, alpha, 1)
    reversed_L2R, reversed_R2L = reverse_beams(L2R_beams), reverse_beams(R2L_beams)

    # Compare L2R with reversed R2L, and R2L with reversed L2R
    for i in range(beam_width):
        difference_L, difference_R = 0, 0
        for j in range(beam_width):   
            difference_L += (L2R_beams[i]!=reversed_R2L[j]).int().sum().item()
            difference_R += (R2L_beams[i]!=reversed_L2R[j]).int().sum().item()
        L2R_probs[i] -= torch.tensor(difference_L * b)
        R2L_probs[i] -= torch.tensor(difference_R * b)

    probs_L, indices_L = torch.topk(L2R_probs, 1)
    probs_R, indices_R = torch.topk(R2L_probs, 1)
    best = 1 if probs_L > probs_R else 0
    if best:
        res = L2R_beams[indices_L]
    else:
        res = reverse_beams(R2L_beams[indices_R])
    res = res[res!=0]
    latex_out = tokenizer.decode(t for t in res.tolist())
    return latex_out

input = Image.open(r'C:\Users\edmun\Desktop\VSCode Projects\HME_Training\data\excerpt_cache\valid\00184588166a4dac.png') 
result = bidirectional_search(input, model, tokenizer, transform, device, 5, 1, 0.1)
print(result)