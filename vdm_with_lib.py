import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
from torchvision import transforms as T, utils

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    num_frames = 10,
    timesteps = 300,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    './data_train',                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = 1,
    train_lr = 1e-4,
    save_and_sample_every = 10,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    results_folder = './results_test',
)

trainer.load(milestone=129)
trainer.train()

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

sampled_videos = diffusion.sample(batch_size = 1)
u_sampled_videos = sampled_videos.unbind(dim = 1)
for i in range(len(u_sampled_videos)):
    images = video_tensor_to_gif(u_sampled_videos[i], "result_"+str(i)+".gif")