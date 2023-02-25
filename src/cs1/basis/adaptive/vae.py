import random
from tqdm import tqdm
from qsi import io
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import HTML, display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
from torchviz import make_dot
from operator import itemgetter

def plot_signals(X, scaler=None):
    '''
    Plot signals

    Parameters
    ----------
    X : a batch of signals, shape (batch_size, n_features)
    '''
    X = X.squeeze().cpu().numpy()
    if scaler is not None:
        X = scaler.inverse_transform(X)  # NOTE: remember to reverse scaling
    for x in X:
        plt.figure(figsize=(5, 1))
        plt.plot(x.T)
        plt.axis('off')
        plt.show()

def build_vae(dataset_id = 'vintage'):
    '''
    Build and train two vae models , one with 2 hidden layers, one with 1 hidden layer.
    The trained weights are saved locally.
    '''

    display(HTML('<h2>Load dataset</h2>'))

    if dataset_id == 'vintage':
        X, y, X_names, desc, labels = io.load_dataset('vintage_526')
    elif dataset_id == 'vintage2':
        X, y, X_names, desc, labels = io.load_dataset('vintage_spi', y_subset=[0,3])
    else:
        X, y, X_names, labels = io.open_dataset(
         '7345X.5.csv', delimiter=',', has_y=True, labels=["5Y", "8Y", "16Y", "26Y"])
        #io.scatter_plot(X, y, labels = labels)

    nc = len(set(y)) # number of classes

    display(HTML('<h2>Train a LogisticRegressionCV on the dataset</h2>'))
    display(HTML('<p>We will use this model to evaluate the reconstructed data.</p>'))

    scaler = StandardScaler()  # MinMaxScaler() # StandardScaler()
    X = scaler.fit_transform(X)

    clf = LogisticRegressionCV(cv=5).fit(X, y)
    print('LogisticRegressionCV score on entire dataset:', clf.score(X, y))

    HTML('<h2>Train VAE</h2>')

    n = X.shape[1]
    batch_size = 32
    h_dim1 = 200
    h_dim2 = 50
    z_dim = 10

    save_path = dataset_id + '_vae_' + str([h_dim1, h_dim2, z_dim]) + '.pth'
    model1 = train_vae(X, y, batch_size=batch_size,
                      h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim)
    torch.save(model1.state_dict(), save_path)

    display(HTML('<h3>Model 1 (two hidden layers) saved to: ' + save_path + '</h3>'))

    input_vec = torch.zeros(1, n, dtype=torch.float, requires_grad=False).to('cuda')
    out = model1(input_vec)
    display(make_dot(out))  # plot graph of variable, not of a nn.Module

    ########### MODEL 2 ############

    h_dim2 = 0

    save_path = dataset_id + '_vae_' + str([h_dim1, h_dim2, z_dim]) + '.pth'
    model2 = train_vae(X, y, batch_size=batch_size,
                      h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim)
    torch.save(model2.state_dict(), save_path)

    display(HTML('<h3>Model 2 (one hidden layers) saved to: ' + save_path + '</h3>'))

    input_vec = torch.zeros(1, n, dtype=torch.float, requires_grad=False).to('cuda')
    out = model2(input_vec)
    display(make_dot(out))  # plot graph of variable, not of a nn.Module

    display(HTML('<h2>Show some generated signals from VAE (use Model 2)</h2>'))

    # peek generated signals
    with torch.no_grad():
        sample_size = 3
        # Generating 64 random z in the representation space
        z = torch.randn(sample_size, z_dim).cuda()
        # Evaluating the decoder on each of them
        sample = model2.decoder(z).cuda()
        plot_signals(make_grid(sample.view(sample_size, 1, n), padding=0),
                     scaler=scaler)  # Plotting the resulting signals

    display(HTML('<h3>To reload model, use this code: </h3><pre>' + '''
model = torchVAE(x_dim=n, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim)
model.load_state_dict(torch.load(save_path))
model.to('cuda') # load to GPU

# print model structure
for layer in model.named_modules():
    if 'fc' in layer[0]:
        print(layer)''' + '</pre>'))

    return X, y, scaler, clf, model1, model2

class torchVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(torchVAE, self).__init__()

        if h_dim2 > 0: # two hidden layers
            # Defining the encoder architecture
            self.fc1 = nn.Linear(x_dim, h_dim1)
            self.fc2 = nn.Linear(h_dim1, h_dim2)
            self.fc31 = nn.Linear(h_dim2, z_dim)
            self.fc32 = nn.Linear(h_dim2, z_dim)

            # Defining the decoder architecture
            self.fc4 = nn.Linear(z_dim, h_dim2)
            self.fc5 = nn.Linear(h_dim2, h_dim1)
            self.fc6 = nn.Linear(h_dim1, x_dim)
        else: # one hidden layer
            # Defining the encoder architecture
            self.fc1 = nn.Linear(x_dim, h_dim1)
            self.fc31 = nn.Linear(h_dim1, z_dim)
            self.fc32 = nn.Linear(h_dim1, z_dim)

            self.fc2 = None

            # Defining the decoder architecture
            self.fc4 = nn.Linear(z_dim, h_dim1)
            self.fc6 = nn.Linear(h_dim1, x_dim)

            self.fc5 = None

    def encoder(self, x):
        # Defining the encoder forward pass
        h = F.relu(self.fc1(x))
        if self.fc2 is not None:
            h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # Returning (mu, log_var)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # Returning z sample

    def decoder(self, z):
        # Defining the decoder forward pass
        h = F.relu(self.fc4(z))
        if self.fc5 is not None:
            h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        # Defining the global forward pass
        mu, log_var = self.encoder(x.view(-1, x.size(1)))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


def vae_loss_function(recon_x, x, mu, log_var):
    '''
    define loss as reconstruction error + KL divergence 
    '''
    recon_error = F.binary_cross_entropy(
        recon_x, x.view(-1, recon_x.size(1)), reduction='sum')
    KL = 0.5 * torch.sum(log_var.exp() - log_var + mu.pow(2) - 1)
    return recon_error + KL


def train_vae(X, y, batch_size=64, h_dim1=200, h_dim2=50, z_dim=10):
    '''
    h_dim2 : size of the 2nd hidden layer in the encoder. set to 0 to remove the layer.
    '''

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True)  # , random_state=0

    n = X.shape[1]  # 3400

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Building model
    # Sizes for the hidden layers are identical to those in the paper

    vae_model = torchVAE(x_dim=n, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim)
    if torch.cuda.is_available():
        vae_model.cuda()

    optimizer = optim.Adam(vae_model.parameters())

    # Training the VAE for N epochs
    for epoch in range(1, 30):
        vae_model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):  # Generating batch

            data = data.float().cuda()
            optimizer.zero_grad()  # Telling Pytorch not to store gradients between backward passes

            recon_batch, mu, log_var = vae_model(data)  # Forward pass
            loss = vae_loss_function(recon_batch, data, mu, log_var)  # Computing loss

            loss.backward()  # Performing automatic differentiation w.r.t weights of the networks
            train_loss += loss.item()  # Updating loss value
            optimizer.step()  # Perform parameters-update using gradients computed with .backward()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

        # Testing the VAE
        vae_model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.float().cuda()
                recon, mu, log_var = vae_model(data)

                # Adding batch loss to accumulator
                test_loss += vae_loss_function(recon, data, mu, log_var).item()

        # Normalizing with number of samples in the test set
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    return vae_model


def vae_reconstruct(model, PHI, xs, lr = 0.01, regularization = 0.1, iterations = 1000, N = 10, debug_mode = False):
    '''
    Reconstruct 1D signal using VAE
    
    Parameters
    ----------
    model : a pretrained VAE model  
    PHI : sensing matrix
    xs : sampled signal
    reg : set None or 0 to disable regularization  
    lr : learning rate  
    iterations : iterations for gradient descent
    N : due to randomness, we can perform N runs and get the best.
    '''
    
    z_dim = 10    
    for layer in model.named_modules():
        if layer[0] == 'fc31':
            z_dim = layer[1].out_features
            break
    
    z_best = torch.randn(1, z_dim).cuda()
    z_best.requires_grad_(False)
    
    for _ in range(N):  # Performing 10 random restarts, i.e. repeat for 10 initial random z's
        z = torch.randn(1, z_dim).cuda()
        z.requires_grad_(True)
        
        optimizer = optim.Adam([z], lr = lr)
        losses = []
        for _ in range(iterations):  # Performing iter gradient steps
            xsr = torch.mm(PHI, model.decoder(z).view(-1,1).cuda())  # xsr = PHI * Xr, i.e., reconstructed sampled signal, it should be close to xs
            
            if regularization is not None:
                loss = torch.pow(torch.norm(xsr - xs), 2) + regularization * torch.pow(torch.norm(z),2) # 正则化，抑制z向量
            else:
                loss = torch.pow(torch.norm(xsr - xs), 2)
                
            losses.append(loss.item()) # Storing loss history to determine best iterations

            optimizer.zero_grad()
            loss.backward()  # Using automatic differentiation on the loss
            optimizer.step()

        if debug_mode:
            plt.figure(figsize=(5,2))
            plt.plot(range(iterations), losses)
            plt.title('Loss ~ Iterations')
            plt.show()

        sample = model.decoder(z).cuda()
        with torch.no_grad():
            if loss < torch.pow(torch.norm(torch.mm(PHI,model.decoder(z_best).view(-1,1).cuda()) - xs),2):
                z_best = z  # Keeping z with smallest measurement error
    
    return z_best

def vae_cs(model, x, k = 0.1, PHI_flavor = 'gaussian', add_noise = True, lr = 0.01, regularization = 0.1, iterations = 1000, N = 10, debug_mode = False):
    '''
    Compressed sensing using VAE

    Parameters
    ----------
    model : a pretrained VAE model
    x : 1D signal
    k : sampling ratio, 0~1
    PHI_flavor : 'gaussian' or 'bernoulli'. The hardware implementation of 'gaussian' is more costly than 'bernoulli'.
    add_noise : whether add white noise

    Notes
    -----
    About 'bernoulli' and 'gaussian' sensing matrices: 
    i.e., random binary sensing matrix vs. random Gaussian sensing matrix. 
    A sensing matrix maps input vector to measurement vector through linear wighted summation of input. 
    What makes a specefic matrix good, is application dependent. Now, both distributions more or less satisfy RIP. 
    However hardware implementation of the Bernoulli matrix (binary or bipolar) is much much easier especially in analog domain. 
    A Bernoulli weight is either 0 or 1 (or -1/1 in case of polar Bernoulli), but a Gaussian wight is a floating point figure. 
    Multiplication of a flouting point number either in digital or analog, is resource consuming, while multiplication of a Bernoulli weight is feasible through implementation of a simple switch in analog domain or and instruction in digital.
    '''
    
    x = x.float() # shape : nx1
    n = x.size(0)
    ns = round(n*k) # dim of xs
    
    if PHI_flavor == 'gaussian' or PHI_flavor == 'normal':
    
        ######### Sensing matrix from Normal dist ############
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1/np.sqrt(n)])) # 构造N(0, 1/n)的感知矩阵
        PHI = normal.sample((ns,n)).squeeze().cuda() # squeeze对数据的维度进行压缩，去掉维数为1的的维度，默认是将所有为1的维度删掉
        # print('gaussian', PHI.size())
        
    else: # 'bernoulli'
        
        ######### Sensing matrix using the identity matrix ###########
        IDX = random.sample(range(n), ns)
        PHI = torch.Tensor( np.eye(n)[IDX] ).cuda().float()
        # print('bernoulli', PHI.size())
    
    PHI.requires_grad_(False)
        
    if add_noise:        
        normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.1/np.sqrt(n)]))
        noise = normal.sample((ns,)).cuda() # 白噪声
        noise.requires_grad_(False)
    
        xs = torch.mm(PHI, x) + noise
    else:
        xs = torch.mm(PHI, x)
        
    xs.cuda()
    xs.requires_grad_(False)
    
    z = vae_reconstruct(model, PHI, xs, lr = lr, regularization = regularization, iterations = iterations, N = N, debug_mode=debug_mode) # 依据Xs求解z
    xr = model.decoder(z).view(n,1).detach().cpu().numpy() # 利用decoder获取xr
    return xr


def VAE_Sensing_n_Recovery(model, x, scaler = None, k = 0.1,
                            PHI_flavor = 'gaussian', 
                            add_noise = True, 
                            lr = 0.01, regularization = 0.1, 
                            iterations = 1000, N = 10, show_plot = True, debug_mode = False):
    '''
    VAE sensing and recovery for one signal sample.
    
    Parameters
    ----------
    scaler : whether there is a rescaling pre-processor
    '''
    
    ##### Load model from local filesystem #########
    # model = torchVAE(x_dim=n, h_dim1=h_dim1, h_dim2=h_dim2, z_dim=z_dim)
    # model.load_state_dict(torch.load(model_path))
    # model.to('cuda') # load to GPU

    xr = vae_cs(model, torch.from_numpy(x).view(-1,1).cuda(), k = k, 
                            PHI_flavor = PHI_flavor, 
                            add_noise = add_noise, 
                            lr = lr, regularization = regularization, 
                            iterations = iterations, N = N, debug_mode=debug_mode)

    # print(xr.shape, x.shape)
    if scaler is not None:
        xr = scaler.inverse_transform( xr.reshape(1,-1) )[0]
        x = scaler.inverse_transform( x.reshape(1,-1) )[0]
   
    if show_plot:
        plt.figure(figsize=(9, 3))
        plt.plot(x.T, color='gray')
        plt.title('original signal')
        plt.show()

        plt.figure(figsize=(9, 3))
        plt.plot(xr.T, color='gray')
        plt.title('recovered signal')
        plt.show()


def vae_cs_grid_search(model, X, y,
ks = [0.01, 0.05, 0.1, 0.3],
PHI_flavors = ['gaussian', 'bernoulli'],
add_noises = [True, False],
lrs = [0.001, 0.01, 0.1],
regularizations = [0, 0.1, 1],
iterationss = [500, 1000],
Ns = [3, 10]):
    
    '''
    Return
    ------
    sorted_dic : sorted result from the best to the worst
    '''
    
    clf = LogisticRegressionCV(cv=5).fit(X, y)
    print('LogisticRegressionCV score on entire dataset:', clf.score(X, y))

    dic = {}
    best_acc = 0
    best_hparams = ()

    for k in ks:
        for PHI_flavor in PHI_flavors:
            for add_noise in add_noises:
                for lr in lrs:
                    for regularization in regularizations:
                        for iterations in iterationss:
                            for N in Ns:

                                print('\nGrid Search Loop:', k, PHI_flavor, add_noise, lr, regularization, iterations, N)


                                Xr = [vae_cs(model, torch.from_numpy(x).view(-1,1).cuda(), k = k, 
                                                        PHI_flavor = PHI_flavor, 
                                                        add_noise = add_noise, 
                                                        lr = lr, regularization = regularization, 
                                                        iterations = iterations, N = N) 
                                                for x in tqdm(X)]

                                Xr = np.squeeze(Xr)
                                acc = clf.score(Xr, y)
                                print('Acc:', acc)
                                dic[(k, PHI_flavor, add_noise, lr, regularization, iterations, N)] = acc
                                if best_acc < acc:                                    
                                    best_hparams = (k, PHI_flavor, add_noise, lr, regularization, iterations, N)
                                    print('Acc improved from {} to {}. Update best hparams : {}'.format(best_acc, acc, best_hparams ))       
                                    best_acc = acc                             


    sorted_dic = dict(sorted(dic.items(), key=itemgetter(1), reverse = True))
    return dic, best_hparams, best_acc, sorted_dic
