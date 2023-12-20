class my_mlp(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(my_mlp, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size[0])
        self.fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = torch.nn.Linear(hidden_size[1], num_outputs)
        #self.fc3 = torch.nn.Linear(hidden_size[1], hidden_size[2]) #for deeper MLP
        #self.fc4 = torch.nn.Linear(hidden_size[2], hidden_size[3])
        #self.fc5 = torch.nn.Linear(hidden_size[3], num_outputs)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu(hidden2)
        output = self.fc3(relu2)
        #hidden3 = self.fc3(relu2) #for deeper MLP
        #relu3 = self.relu(hidden3)
        #hidden4 = self.fc4(relu3)
        #relu4 = self.relu(hidden4)
        #output = self.fc5(relu4)
        return output

# accuracy is computed for mlp 
def testAccuracymlp(model, test_data, test_labels):
    model.eval()
    loss = torch.nn.MSELoss()
    output = model(test_data.T)
    error = loss(output,test_labels.T).item()
    return(error)

# utility function to create performance plots
def plots(results, save_dir='', filename='', show_plot=True):
    color_list = ['#ff0000']
    style_list = ['-']
    plot_curve_args = [{'c': color_list[0],
                        'linestyle': style_list[0],
                        'linewidth': 2}]
    font_size = 18
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    ax = axes[0]
    ax.set_title('training_error', loc='left', fontsize=font_size)
    for result, args in zip(results, plot_curve_args):
        ax.plot(np.arange(1, len(result['training_error']) + 1), result['training_error'], label=result['name'], **args)
        ax.set_xlabel(xlabel='step', fontsize=font_size)
        ax.set_ylabel(ylabel='error', fontsize=font_size)
        ax.tick_params(labelsize=12)
    ax = axes[1]
    ax.set_title('validation_error', loc='right', fontsize=font_size)
    for result, args in zip(results, plot_curve_args):
        ax.plot(np.arange(1, len(result['validation_error']) + 1), result['validation_error'], label=result['name'],
                **args)
        ax.set_xlabel(xlabel='step', fontsize=font_size)
        ax.set_ylabel(ylabel='error', fontsize=font_size)
        ax.tick_params(labelsize=12)

    if show_plot:
        plt.show()
    fig.savefig(os.path.join(save_dir, filename + '.png'))
    
batch_size = 50
num_epoch = 5
learning_rate = 0.01


#TRAINING STARTS

#model_mlp = my_mlp(250,[50,100,50,10],6) # model is assigned
model_mlp = my_mlp(2500,[50,10],6) # model is assigned
model_mlp.to(device)
model_mlp.train()
training_error = []
validation_error = []
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_mlp.parameters(),lr=learning_rate)
training_loss = 0.0
total = 0.0
train_data = train_data.to(device)
val_data = val_data.to(device)
train_labels = train_labels.to(device)
val_labels = val_labels.to(device)
for epoch in range(num_epoch):
    for i in range(int(train_data.size(1)/batch_size)):
        data = train_data[:,batch_size*i:batch_size*(i+1)].T
        labels = train_labels[:,batch_size*i:batch_size*(i+1)].T
        optimizer.zero_grad()  # zero the parameter gradients
        output = model_mlp(data)  # predict classes
        loss_at = loss(output, labels)  # compute the loss
        loss_at.backward()  # backpropagate the loss
        optimizer.step()  # adjust parameters based on the calculated gradients
        training_loss += loss_at.item() # extract the loss value
        total += labels.size(0)
        if i % 10 == 9:
          print("Training batch number: "+str(i)+" and Epoch: "+str(epoch))
          training_error.append(training_loss/total)
          validation_loss = testAccuracymlp(model_mlp,val_data,val_labels)
          validation_error.append(validation_loss)
                                  
my_mlp_data = (training_error,validation_error)
part_my_mlp = {}
part_my_mlp['name'] = 'my_mlp'
training_error = my_mlp_data[0]
validation_error = my_mlp_data[1]
part_my_mlp['training_error'] = training_error
part_my_mlp['validation_error'] = validation_error
# data workspace is saved as json
with open(str('part_my_mlp.json'), 'w') as json_file:
    json.dump(part_my_mlp, json_file)
with open(str('part_my_mlp.json')) as my_mlp_file:
    my_mlp_dic = json.load(my_mlp_file)
    
results = [my_mlp_dic]
plots(results,save_dir='./',filename='plots',show_plot=True)


out = model_mlp(val_data.T)
output_arr = out.cpu().detach().numpy()
output_angx = output_arr[:,0]
output_angy = output_arr[:,1]
output_angz = output_arr[:,2]
output_disx = output_arr[:,3]
output_disy = output_arr[:,4]
output_disz = output_arr[:,5]

plt.hist(output_angx, bins=np.linspace(output_angx.min(),output_angx.max()))
plt.xlabel('X Angle Error')
plt.ylabel('Number of Samples')
plt.title('Histogram of Estimated X Angle Error')
# plt.hist(output_angy, bins=np.linspace(output_angy.min(),output_angy.max()))
# plt.xlabel('Y Angle Error')
# plt.ylabel('Number of Samples')
# plt.title('Histogram of Estimated Y Angle Error')
# plt.hist(output_angz, bins=np.linspace(output_angz.min(),output_angz.max()))
# plt.xlabel('Z Angle Error')
# plt.ylabel('Number of Samples')
# plt.title('Histogram of Estimated Z Angle Error')
# plt.hist(output_disx, bins=np.linspace(output_disx.min(),output_disx.max()))
# plt.xlabel('X Distance Error')
# plt.ylabel('Number of Samples')
# plt.title('Histogram of Estimated X Distance Error')
# plt.hist(output_disy, bins=np.linspace(output_disy.min(),output_disy.max()))
# plt.xlabel('Y Distance Error')
# plt.ylabel('Number of Samples')
# plt.title('Histogram of Estimated Y Distance Error')
# plt.hist(output_disz, bins=np.linspace(output_disz.min(),output_disz.max()))
# plt.xlabel('Z Distance Error')
# plt.ylabel('Number of Samples')
# plt.title('Histogram of Estimated Z Distance Error')