from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch


def wndw(input, win_len):
    out = []
    idx = []
    step_size = int(win_len / 2)
    le = len(input)
    base_idx = 0
    end_idx = win_len
    # print(le)
    while (end_idx < le):
        # print(base_idx, end_idx)
        tmp = input[base_idx:end_idx]
        idx.append(base_idx)
        out.append(" ".join(tmp))

        base_idx += step_size
        end_idx += step_size

        # print(out)
    return out, idx

#Cosine similarty funtion via pytorch
def cos_sim_calc(input_sequence, chap_num, vid_num, win_len):
    path = "by_video/ch{}_{}.text"
    list_of_lists = []
    try:
        with open(path.format(chap_num, vid_num)) as f:
            for line in f:
                list_of_lists.append(line)
    except:
        return " Video not found", 0,0
    with open(path.format(chap_num, vid_num)) as f:
        for line in f:
            list_of_lists.append(line)
    list_of_lists, idx = wndw(list_of_lists, win_len)

    list_of_lists.insert(0, input_sequence)

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    sentences = list_of_lists
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    #Collecting the tensors in one tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().numpy()


    cos_dis = cosine_similarity([mean_pooled[0]], mean_pooled[1:])
    matched_text = sentences[cos_dis.argmax() + 1]
    start_id_match = idx[cos_dis.argmax()] / 3 - win_len + 1
    similarity_val = cos_dis.max()
    return matched_text ,similarity_val , start_id_match

