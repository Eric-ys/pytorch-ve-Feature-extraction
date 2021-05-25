import torch
# from models import BaseVAE
from torch import nn
from torch.nn import functional as F
# from .types_ import *
import torchvision
from torchvision import transforms
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torchvision.utils import save_image
#os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
#device = torch.device('cuda')
#device_ids = [0, 1]

class BetaVAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 256,
                 hidden_dims: None = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [3, 32, 32, 64, 64]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride = 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*16, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*16, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*16)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)   # 16 64 4 4
        result = torch.flatten(result, start_dim=1)    # 16 1024

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):   # 16 256
        result = self.decoder_input(z)     # 16 1024
        result = result.view(-1, 64, 4, 4)  #16 64 4 4
        result = self.decoder(result)    # 16 3 64 64
        result = self.final_layer(result)   # 16 3 128 128
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)     # 16 256
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      recon_x, x, mu, logvar,
                      M_N = 0.005) -> dict:
        self.num_iter += 1
        recons = recon_x
        input = x
        mu = mu
        log_var = logvar
        kld_weight = M_N   # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def get_data():
    # data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # train_data = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    transform = transforms.Compose(
        [torchvision.transforms.Resize((128, 128)),
         torchvision.transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])

    # dataset = torchvision.datasets.ImageFolder(root='../pytorch-vae-master/image1', transform=transform)
    dataset = torchvision.datasets.ImageFolder(root='D:\gansheyi\image1', transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        i=0
        for t, m, s in zip(tensor, self.mean, self.std):
            tmp = tensor[i]
            tmp = tmp.mul(s)
            tmp = tmp.add(m)
            tensor[i] = tmp
            i+=1
            # t = t.mul(s).add(m)
        return tensor

inver_transform_face = transforms.Compose([
    DeNormalize([0.5], [0.5]),
    lambda x: x.cpu().detach().numpy() * 255.,
    lambda x: x.transpose(1, 2, 0).astype(np.int8)
])

def get_test_data():
    transform = transforms.Compose(
        [torchvision.transforms.Resize((128, 128)),
         transforms.ToTensor(),
         ])

    # dataset = torchvision.datasets.ImageFolder(root='../pytorch-vae-master/image1', transform=transform)
    dataset = torchvision.datasets.ImageFolder(root='D:\gansheyi\image', transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
    )
    return test_loader

if __name__ == '__main__':

    # 超参数设置
    batch_size = 16
    lr = 1e-3
    epoches = 100
    # in_channels = 1
    # latent_dim = 20
    # hidden_dims = None
    model = BetaVAE()
    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda()

    train_data = get_data()

    # reconstruction_function = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    p=1
    original_image = get_test_data()
    for imag, _ in original_image:
        for i in imag:
            save_image(i, './vae_img1/original_image_{}.png'.format(p))
            p += 1

    for epoch in range(epoches):
        k=1

        for img, _ in train_data:
            # img = img.view(img.size(0), -1)
            #original_image = img
            img = Variable(img)
            if torch.cuda.is_available():
                # img = torch.nn.DataParallel(img, device_ids=device_ids)
                img = img.cuda()
            # forward
            # print(img[0].min())
            output, _, mu, logvar = model(img)   #img  16 3 128 128   output 16 1 32 32

            loss = model.loss_function(output, img, mu, logvar)['loss']# /img.size(0)  # ['loss']
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch==1):

                # pic = model.sample(16, torch.device('cpu'))
                if not os.path.exists('./vae_img1'):
                    os.mkdir('./vae_img1')
                print("ss",output[0].max())
                for i in range(output.size(0)):
                    im = output[i]
                    image = inver_transform_face(im)

                    # input_tensor = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('./vae_img2/image_{}.png'.format(k), image)
                    # save_image(image, './vae_img1/image_{}.png'.format(k))
                    #save_image(img, './vae_img2/original_image_{}.png'.format(p))
                    # p+=1
                    k+=1

        print("epoch=", epoch, loss.data.float())
        if (epoch+1) % 10 == 1:
            print("epoch = {}, loss is {}".format(epoch+1, loss.data))
    # p = 1
    # for img, _ in train_data:
    #     for img in original_image:
    #         save_image(img, './vae_img2/original_image_{}.png'.format(p))
    #         p += 1
    #         output, _, mu, logvar = model(img)
    #         save_image(image, './vae_img1/image_{}.png'.format(k))
    #         k += 1
    torch.save(model, './vae.pth')
#
   # transform = transforms.Compose(
   #     [torchvision.transforms.Resize((128, 128)),
   #      transforms.ToTensor()
   #     ])
   # orig_img =torchvision.datasets.ImageFolder(root='D:\gansheyi\image1', transform=transform)
   # test_img = torch.utils.data.DataLoader(dataset=orig_img)
   # j=0
   # k=0
   # for img in test_img:
   #     save_image(img,'./original_image/image_{}.png'.format(k))
   #     k+=1

   # transform1 = transforms.Compose(
   #     [torchvision.transforms.Resize((128, 128)),
   #      transforms.ToTensor(),
   #      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
   #      ])
   # test_img = torchvision.datasets.ImageFolder(root='D:\gansheyi\image1', transform=transform1)

   # for img in test_img:
   #     model = torch.load('./vae.pth')
   #     img = Variable(img)
   #     _,result = model(img)

   #     #  test_data = data.type(torch.FloatTensor)
   #     #  _,result = model(test_data)
   #     #  im_result = result.view(128,128)
   #     #  save_image(.detach().numpy(), './vae_img_test/image_{}.png'.format(j))
   #     #  save_image(im_result.detach().numpy(), './vae_img_test/image_{}.png'.format(j))
