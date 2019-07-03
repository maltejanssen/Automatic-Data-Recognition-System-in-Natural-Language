import torch

def lossFn(outputs, labels):
    """

    :param Variable outputs: log softmax output of the model
    :param Variable labels: indices of labels (-1 if padding) [0, 1, ... num_tag-1]
    :return Variable loss:  cross entropy loss for all tokens in the batch
    """
    # reshape labels to give a flat vector
    labels = labels.view(-1)
    # remove padding
    mask = (labels >= 0).float()

    # covert padding into positive, because of negative indexing errors -> but ignore with mask
    labels = labels % outputs.shape[1]

    # num of non mask tokens
    num_tokens = int(torch.sum(mask).item())  # .data[0]

    return -torch.sum(outputs[range(outputs.shape[0]), labels] * mask) / num_tokens


