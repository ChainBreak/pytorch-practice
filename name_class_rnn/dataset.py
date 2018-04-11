#!/usr/bin/env python
import torch.utils.data
import glob
import os
from io import open
from torch.autograd import Variable
import unicodedata
import string
import torch




class Dataset(torch.utils.data.Dataset):

    def __init__(self,dir_path):
        self.name_lang_pairs = []
        self.lang_index_map = {}


        file_paths = glob.glob(dir_path)
        file_paths.sort()

        self.n_lang = len(file_paths)
        self.n_letters = len(all_letters)

        for index,file_path in enumerate(file_paths):
            language =  os.path.splitext(os.path.basename(file_path))[0]
            self.lang_index_map[index] = language
            self.lang_index_map[language] = index
            with open(file_path,encoding='utf-8') as f:
                names = f.readlines()
                for name in names:
                    ascii_name = str( unicodeToAscii( name.strip()))
                    self.name_lang_pairs.append((ascii_name,language))

        print"Loaded %i names for %i languages"%(len(self.name_lang_pairs),self.n_lang)


    def __len__(self):
        return len(self.name_lang_pairs)

    def __getitem__(self,index):
        name,lang = self.name_lang_pairs[index]

        #<line_length x 1 x n_letters>
        name_tensor = lineToTensor(name)

        lang_index = self.lang_index_map[lang]
        lang_tensor = torch.LongTensor([lang_index])



        return {"name": name,"name_tensor": name_tensor, "lang": lang, "lang_tensor": lang_tensor}

all_letters = string.ascii_letters + " .,;'"


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
        )

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def letterToIndex(letter):
    return all_letters.find(letter)

if __name__ == "__main__":

    d = Dataset("data/names/*.txt")
    sample = d[0]
    print(sample["name_tensor"])
    print(sample["lang_tensor"])
