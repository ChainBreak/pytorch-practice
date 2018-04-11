#!/usr/bin/env python

import torch.nn as nn
from torch.autograd import Variable
import dataset
import torch.utils.data
import model
import matplotlib.pyplot as plt

def train(model,train_loader):

    filename = "rnn_state.pt"
    try:
        state=torch.load(filename)
        model.load_state_dict(state["state_dict"])
        #optimizer.load_state_dict(state["optimizer_dict"])
    except:
        # raise
        print("Could not load model file")
        state = {}
        state["train_loss_history"] = []
        state["test_loss_history"] = []
        state["epoch"] = 0

    criterion = nn.NLLLoss()
    lr = 0.005

    print_every = 5000
    plot_every = 1000
    n_epoch = 50
    train_loss = 0.0
    count = 0
    while state["epoch"] <n_epoch:

        n_batch = len(train_loader)


        model.train()
        for i_batch,batch_data in enumerate(train_loader,0):
            name_tensor = Variable(batch_data["name_tensor"])
            lang_tensor = Variable(batch_data["lang_tensor"])

            name_tensor = name_tensor.view(name_tensor.size()[1:])
            lang_tensor = lang_tensor.view(1)

            model.zero_grad()
            hidden = model.initHidden()
            n_letters = name_tensor.size()[0]
            for i in range(n_letters):
                output,hidden = model(name_tensor[i],hidden)


            loss = criterion(output,lang_tensor)
            loss.backward()

            train_loss += loss.data[0]

            for p in model.parameters():
                p.data.add_(-lr,p.grad.data)


            if count % plot_every == 0:
                train_loss_avg = train_loss / plot_every
                print("Epoch: %i/%i, Batch: %i/%i, Loss: %f, %s" % (state["epoch"],n_epoch,i_batch,n_batch,train_loss_avg,batch_data["lang"]))
                state["train_loss_history"].append( train_loss_avg)
                train_loss = 0.0
                plt.cla()
                plt.plot(state["train_loss_history"])
                plt.plot(state["test_loss_history"])
                plt.draw()
                plt.pause(0.1)

            count += 1


        print("\nEpoch: %i/%i Saved!" % (state["epoch"],n_epoch))
        state["state_dict"] = model.state_dict()
        # state["optimizer_dict"] = optimizer.state_dict()
        state["epoch"] += 1
        torch.save(state,filename)



def main():

    train_dataset = dataset.Dataset("data/names/*.txt")
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=1)


    input_size = train_dataset.n_letters
    hidden_size = 256
    output_size = train_dataset.n_lang

    rnn = model.RNN(input_size,hidden_size,output_size)

    train(rnn,train_loader)

if __name__ == "__main__":
    main()
