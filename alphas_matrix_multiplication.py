import torch



#src = torch.tensor([0, 0, 1, 4, 4])
#trg = torch.tensor([2, 10])
#dictionary = {1:1, 2:2, 4:10}

def get_alpha_matrix(src, trg, dictionary):
    """
    Assuming:
    src is of length M
    trg is of length N
    largest english token in translation dictionary is E
    largest french token in translation dictionary is F
    we first create a matrix of size E x F to show alignments in the dictionary
    then for each translation pair
        - onehotencode src to a matrix of size M x E 
        - onehotencode trg to a matrix of size N X F (then transpose to get F x N)
    finally do matrix multiplication on 
    (M x E) @ (E x F) @ (F x N) = M x N which is what we expected
    """
    # convert dictionary into a matrix -- this needs to be generated only once

    max_row = max(dictionary.keys())+1
    max_col= max(dictionary.values())+1
    EF = torch.zeros(max_row, max_col).cuda()
    for k, v in dictionary.items():
        EF[k, v] = 1

    # for each sample
    # convert src to one hot encoded matrix
    src = src.view(-1, 1)
    ME = torch.zeros(len(src), max_row).cuda()
    ME.scatter_(1, src, 1)

    # convert target to one hot encoded matrix
    trg = trg.view(-1, 1)
    FN = torch.zeros(len(trg), max_col).cuda()
    FN = torch.transpose(FN.scatter_(1, trg, 1), 0, 1)
    return ME @ EF @ FN


if __name__ == '__main__':
    src = torch.tensor([0, 0, 1, 4, 4])
    trg = torch.tensor([2, 10])
    dictionary = {1:1, 2:2, 4:10}

    print(get_alpha_matrix(src, trg, dictionary))
