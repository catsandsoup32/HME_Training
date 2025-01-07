import torch

from models import Full_Model
from tokenizer import LaTeXTokenizer
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import LogSoftmax

device = 'cuda'
tokenizer = LaTeXTokenizer()
tokenizer.id_to_token = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[UNK]', 4: '{', 5: '}', 6: '_', 7: '1', 8: ')', 9: '(', 10: '^', 11: '2', 12: '=', 13: '\\frac', 14: 'x', 15: '0', 16: '-', 17: 'i', 18: ',', 19: 'n', 20: 't', 21: 'a', 22: '+', 23: '3', 24: 'r', 25: '.', 26: 'm', 27: 'd', 28: '4', 29: '5', 30: '6', 31: 'k', 32: 's', 33: '7', 34: '8', 35: '9', 36: 'c', 37: 'p', 38: 'e', 39: '|', 40: 'f', 41: 'y', 42: '\\cdot', 43: '\\sqrt', 44: 'A', 45: 'b', 46: 'l', 47: 'o', 48: '[', 49: ']', 50: 'g', 51: '\\partial', 52: 'z', 53: 'j', 54: '/', 55: 'v', 56: 'X', 57: 'T', 58: 'R', 59: '&', 60: 'u', 61: 'P', 62: '\\\\', 63: 'S', 64: 'B', 65: 'C', 66: 'E', 67: '\\prime', 68: 'N', 69: '\\begin', 70: '\\end', 71: '\\alpha', 72: '\\int', 73: 'F', 74: 'V', 75: 'q', 76: '\\pi', 77: 'L', 78: 'h', 79: '\\sum', 80: 'M', 81: '\\theta', 82: '\\mu', 83: '\\in', 84: 'H', 85: 'I', 86: '\\hat', 87: 'D', 88: '\\rightarrow', 89: 'G', 90: '\\lambda', 91: '*', 92: '\\sigma', 93: '\\{', 94: ':', 95: '\\}', 96: 'w', 97: '\\overline', 98: '\\infty', 99: '\\mathbb', 100: 'K', 101: '\\beta', 102: '\\omega', 103: '\\rho', 104: '\\epsilon', 105: 'Z', 106: 'Q', 107: 'Y', 108: '\\le', 109: '\\gamma', 110: 'U', 111: '\\times', 112: '\\phi', 113: '\\rangle', 114: '\\Delta', 115: '<', 116: '\\tilde', 117: '\\psi', 118: '\\delta', 119: '\\nu', 120: 'W', 121: '\\tau', 122: 'O', 123: '\\varphi', 124: 'J', 125: '\\langle', 126: '\\vec', 127: '!', 128: '>', 129: '\\nabla', 130: '\\ge', 131: '\\prod', 132: '\\Omega', 133: ';', 134: '\\eta', 135: '\\Gamma', 136: '\\approx', 137: '\\xi', 138: '\\Phi', 139: '\\dot', 140: '\\pm', 141: '\\otimes', 142: '\\circ', 143: '\\wedge', 144: '\\equiv', 145: '\\hbar', 146: '\\chi', 147: '\\underline', 148: '\\zeta', 149: '\\kappa', 150: '\\ne', 151: '\\forall', 152: '\\Sigma', 153: '\\sim', 154: '\\subseteq', 155: '\\Psi', 156: '\\notin', 157: '\\cap', 158: '\\Lambda', 159: '\\mapsto', 160: '\\neg', 161: '\\cup', 162: '\\oplus', 163: '\\Rightarrow', 164: '\\dagger', 165: '\\vee', 166: '\\subset', 167: '\\backslash', 168: '\\rfloor', 169: '\\Pi', 170: '\\lfloor', 171: '\\|', 172: '\\Theta', 173: '\\exists', 174: '\\cong', 175: '\\emptyset', 176: '\\propto', 177: '\\perp', 178: '\\vdash', 179: '\\bigcup', 180: '\\bullet', 181: '\\simeq', 182: '\\leftarrow', 183: '\\aleph', 184: '\\%', 185: '\\vartheta', 186: '\\ll', 187: '\\#', 188: '\\oint', 189: '\\angle', 190: '\\top', 191: '\\leftrightarrow', 192: '\\bigoplus', 193: '\\iint', 194: '\\bigcap', 195: '\\vdots', 196: '\\lceil', 197: '\\rceil', 198: '\\iff', 199: '\\gg', 200: '\\odot', 201: '\\varpi', 202: '\\Leftrightarrow', 203: '\\models', 204: '\\longrightarrow', 205: '\\ominus', 206: '\\iota', 207: '?', 208: '\\upsilon', 209: '\\mp', 210: '\\Xi', 211: '\\varsigma', 212: '\\hookrightarrow', 213: '\\supseteq', 214: '\\supset', 215: '\\subsetneq', 216: '\\triangleq', 217: '\\bigwedge', 218: '\\div', 219: '\\Upsilon', 220: '\\Vdash', 221: '\\&', 222: '\\bigvee', 223: '\\ni', 224: '\\rightleftharpoons', 225: '\\triangle', 226: '\\_', 227: '\\not', 228: '\\bigcirc', 229: '\\sqsubseteq', 230: '\\triangleleft'}
model = Full_Model(vocab_size=231, d_model=256, nhead=8, dim_FF=1024, dropout=0, num_layers=3).to(device)
model.load_state_dict(torch.load(r'runs\Exp9E2.pt', map_location=device, weights_only=True))
transform = transforms.Compose([transforms.Resize((512, 384)),transforms.ToTensor()])


def compare_beam_pairs():
    pass

# input is a PIL image, returns beam_width amount of both L2R and R2L sequences
def BD_beam_search(input, model, tokenizer, transform, device, beam_width, alpha):
    with torch.no_grad():
        vocab_size = 231 # len(tokenizer.vocab)
        logSM = LogSoftmax(dim=-1)
        features = model.encoder(transform(input).unsqueeze(0).to(device)).detach()
        features_batch = features.repeat_interleave(beam_width, dim=0) # From size [1, 192, 256] to [beam_width, 192, 256]
        tgt_mask = torch.triu(torch.ones(200, 200) * float("-inf"), diagonal=1).to(device)

        beams_L = torch.ones([1, 1], dtype=torch.long).to(device) 
        beams_R = beams_L * 2 # R2L sequence starts with <EOS> token

        # Root expansion step
        output_L, output_R = model.decoder(features, beams_L, tgt_mask[:1, :1]), model.decoder(features, beams_R, tgt_mask[:1, :1])
        beam_scores_L, indices_L = torch.topk(logSM(output_L), beam_width, dim=-1) # Size [1, 1, beam_width]
        beam_scores_R, indices_R = torch.topk(logSM(output_R), beam_width, dim=-1)
        beams_L = torch.cat([beams_L.repeat_interleave(beam_width, dim=1).unsqueeze(2), indices_L.permute(0, 2, 1)], dim=-1) # [1, beam_width, 2]
        beams_R = torch.cat([beams_R.repeat_interleave(beam_width, dim=1).unsqueeze(2), indices_R.permute(0, 2, 1)], dim=-1)
        beam_scores_L, beam_scores_R = beam_scores_L.view(beam_width, 1), beam_scores_R.view(beam_width, 1) # Fits [beam_width, vocab] shape in for loop

        for i in range(2, 200):
            tgt_L, tgt_R = beams_L.squeeze(), beams_R.squeeze()
            output_L, output_R = model.decoder(features_batch, tgt_L, tgt_mask[:i, :i]), model.decoder(features_batch, tgt_R, tgt_mask[:i, :i])

            # Get updated probabilities and store as beam_width amount of beam scores
            probs_L, probs_R = logSM(output_L[:, -1, :]), logSM(output_R[:, -1, :]) # [beam_width, vocab]
            probs_L += beam_scores_L # [beam_width, vocab] + [beam_width, 1] 
            probs_R +=  beam_scores_R

            # Make sure completed beams do not expand
            if i > 2:
                probs_L[~active_mask_L, :] = float("-inf")
                probs_R[~active_mask_R, :] = float("-inf")

            # Update beams with next token - since probabilities are flattened, need to find corresponding beam and its next index w.r.t original vocab
            beam_scores_L, indices_L = torch.topk(probs_L.view(beam_width*vocab_size, 1), beam_width, dim=0) # Flattened
            beam_scores_R, indices_R = torch.topk(probs_R.view(beam_width*vocab_size, 1), beam_width, dim=0)
            beams_L = torch.cat([beams_L[torch.arange(1).unsqueeze(-1), indices_L//vocab_size], (indices_L%vocab_size).unsqueeze(2)], dim=-1) 
            beams_R = torch.cat([beams_R[torch.arange(1).unsqueeze(-1), indices_R//vocab_size], (indices_R%vocab_size).unsqueeze(2)], dim=-1) 

            # Check if there was <EOS> or <BOS> generated
            active_mask_L = (beams_L[:, :, -1] != 2).squeeze()
            active_mask_R = (beams_R[:, :, -1] != 1).squeeze()
        

            if i > 5:
                break


        return 


input = Image.open(r'C:\Users\edmun\Desktop\VSCode Projects\HME_Training\data\excerpt_cache\test\000a4e8ca49c5a1c.png') 
BD_beam_search(input, model, tokenizer, transform, device, 5, 1)