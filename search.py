import torch


def beam_search(features, beam_width, model, tokenizer, transform, device, bidirectional=False):
    input = transform(input).unsqueeze(0).to(device) # -> [B, C, H, W]
    beams = torch.ones([1, 1], dtype=torch.long).to(device)
    if bidirectional:
        beams_r = torch.ones([1, 1], dtype=torch.long).to(device)
        beams_r[0, 1] = 2 # <EOS>
    tgt_mask = torch.triu(torch.ones(200, 200) * float("-inf"), diagonal=1).to(device)

    with torch.no_grad():
        output = model(features, beams, tgt_mask[:1, :1])
        probs, indices = torch.topk(torch.log(output), beam_width, dim=-1) # Add log probabilities instead of multiply small numbers

        beams = torch.cat

        for i in range(2, 200):
            tgt = beams.view(beam_width, i)
            output = model.decoder(features, tgt, tgt_mask[:i, :i])            


            # Batch size of beam width
            
        