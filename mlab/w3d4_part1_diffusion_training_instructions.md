
# Diffusion Models For Image Generation

Today you're going to implement and train a tiny diffusion model from scratch on FashionMNIST. Specifically, we'll be following the [2020 paper **Denoising Diffusion Probabilistic Models**](https://arxiv.org/pdf/2006.11239.pdf), which was an influential early paper in the field of realistic image generation. Understanding this paper will give you a solid foundation to understand state of the art diffusion models. I personally believe that diffusion models are an exciting research area and will make other methods like GANs obsolete in the coming years. To get a sense of how diffusion works, you'll first implement and train an even tinier model to generate images of color gradients.

The material is divided into three parts. In Part 1 we'll implement the actual equations for diffusion and train a basic model to generate color gradient images. In Part 2 we'll implement the U-Net architecture, which is a spicy mix of convolutions, MLPs, and attention all in one network. In Part 3, we'll train the U-Net architecture on FashionMNIST.

You're getting to be experts at implementing things, so today will be a less guided experience where you'll have to refer to the paper as you go. Don't worry about following all the math - the math might look intimidating but it's actually either less complicated than it sounds, or can safely be skipped over for the time being.

Diffusion models are an area of active research with rapid developments occurring - our goal today is to conceptually understand all the moving parts and how they fit together into a working system. It's also to understand all the different names for things - part of the difficulty is that diffusion models can be understood mathematically from different perspectives and since these are relatively new, different people use different terminology to refer to the same thing.

Once you understand today's material, you'll have a solid foundation for understanding state of the art systems involving diffusion models like:

- [GLIDE](https://arxiv.org/abs/2112.10741)
- [DALL-E 2](https://openai.com/dall-e-2/) by OpenAI
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion) by the University of Heidelberg
- [ImageGen](https://imagen.research.google/) by Google Brain.

## Table of Contents

- [What Even Is Diffusion?](#what-even-is-diffusion)
- [Today's Plan:](#todays-plan)
    - [Image Processing](#image-processing)
    - [Normalization](#normalization)
    - [Variance Schedule](#variance-schedule)
    - [Forward (q) function - Equation 2](#forward-q-function---equation-)
    - [Forward (q) function - Equation 4](#forward-q-function---equation-)
    - [Training Loop](#training-loop)
- [Sampling from the Model](#sampling-from-the-model)

## What Even Is Diffusion?

We're going to start by thinking about the input distribution of images. [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) is a dataset of 60K training examples and 10K test examples that belong to one of 10 different classes like "t-shirt" or "sandal". Each image is 28x28 pixels and in 8-bit grayscale. We think of those dataset examples as being samples drawn IID from some larger input distribution "the set of all FashionMNIST images".

One way to think about the input distribution is a mapping from each of the $256^{28*28}$ grayscale images to the probability that the image would be collected if we collected more training examples via an identical process as was used to obtain the 60K training examples.

Our goal in generative modeling is to take this input distribution and learn a very rough estimate of the probability in various regions. It should be near zero for images that look like random noise, and also near zero for a picture of a truck since that isn't part of the concept of "the set of all FashionMNIST images".

For our training examples, the fact that they were already sampled is evidence that their probability should be pretty high, but we only have information on 60K examples which is really not a lot compared to $256^{28*28}$. To have any hope of mapping out this space, we need to make some assumptions.

The assumption behind the forward process is that if we add Gaussian noise to an image from the distribution, on average this makes the noised image less likely to belong to the distribution. This isn't guaranteed - there exists some random noise that you could sample with positive probability that is exactly what's needed to turn your sandal into a stylish t-shirt.

The claim is that this is an empirical fact about the way the human visual system perceives objects - a sandal with a small splotch on it still looks like a sandal to us. As long as this holds most of the time, then we've successfully generated an additional training example. In addition, we know something about how the new example relates to the original example.

Note that this is similar but not the same as data augmentation in traditional supervised learning. In that setup, we make a perturbation to the original image and claim that the class label is preserved - that is, we would tell the model via the loss function that our noised sandal is exactly as much a sandal as the original sandal is, for any level of noise up to some arbitrary maximum. In today's setup, we're claiming that the noised sandal is less of a FashionMNIST member in proportion to the amount of noise involved.

Now that we know how to generate as much low-probability data as we want, in theory we could learn a reverse function that takes an image and returns one that is *more* likely to belong to the distribution.

Then we could just repeatedly apply the reverse function to "hill climb" and end up with a final image that has a relatively large probability. We know that deep neural networks are a good way to learn complicated functions, if you can define a loss function suitable for gradient descent and if you can find a suitable parameterization so that learning is smooth.

## Today's Plan:
- Implement the forward process to add noise to images
- Implement the loss function
- Train a toy diffusion model to reconstruct images of color gradients
- Implement the UNet neural network architecture
- Train the network from scratch on FashionMNIST

Ensure you have [the paper](https://arxiv.org/pdf/2006.11239.pdf) open to section 2 (Background). I find it helpful when facing a pile of notation to go through each symbol and try to make each one as concrete as possible. In particular, its useful to clearly distinguish between "a probability distribution" and "a sample from a probability distribution".

To start, we have $x_0$ which is a sample of data. $q(x_0)$ on the other hand is probability distribution - it's a function that you can pass any image to and get out a number telling you how likely that image is to occur in the input distribution.

Specifically, $x_0$ is an image treated as a flattened vector of some length `k = image width * image height * num color channels`. So one element of this vector is the intensity of some color at some position in the image. $x_1$ up to $x_T$ are also images, with some number of steps of noise added and capital $T$ is the maximum number of noise steps we're considering.

The notation $p(x_T) = N(x_T; 0, I)$ was new to me. The function $p$ refers to the reverse process, and in general $p$ is defined through Equation 1. But for the specific scenario that the input to $p$ is some image with T steps of noise added, the output of $p$ is a standard Gaussian. The 0 is a vector of `k` zeros, and the `I` is the identity matrix of shape `(k, k)` which means that each color component at each position is independent of everything else in the image. They dropped the subscript $\theta$ because $\theta$ refers to the parameters of a neural network, but in this specific case, the neural network isn't needed. The semicolon is just separating the left part $x_T$ which is an input to the function from the right part, which is constant.

Equation 1 then defines what $p$ means for all the other possible inputs. It means you start with that standard k-dimensional Gaussian and multiplying by a bunch of other Gaussians of the same dimensionality. Geometrically, you can think of your blob of probability slowly shifting around until we end up with a posterior distribution.

In the actual code, we will work with samples instead: starting with a sample from $p(x_T)$ (which to reiterate is just a standard k-dimensional Gaussian), we repeatedly pass the sample into the neural network to get a mean and variance term, which define a Gaussian that we can sample from to obtain a new sample from $p(x_t-1)$. Once we reach a sample from $p(x_0)$, we'll have an image that looks like a real $x_0$, if our neural network was giving us the right outputs.

Exercise: work through Equation 2 and explain what each symbol means in plain English. The specific values of $\beta$ are defined in Section 4. Try to explain why the square root term is there. If you feel stuck or intimidated, don't hesitate to look at the solution.

<details>

<summary>Solution - Equation 2</summary>

We're going to apply noise in a number of small steps, where each $\beta_t$ is a scalar controlling the variance of the Gaussian. Because we take the square root of $1 - \beta_t$, each $\beta_t$ must be less than or equal to 1.

Section 4 says that the noise schedule is fixed ahead of time and isn't learned during training.

In terms of distributions, the forward process is also a product of Gaussians just like the reverse process. The Gaussians here don't depend on the neural network - to obtain a distribution $q(x_t)$ all we need is the previous $x_{t-1}$ and the appropriate $\beta$. Again, all pixels and all color channels are noised independently.

In terms of samples, if we take a sample $x_0$ and repeatedly sample from the Gaussians, we'll end up with a sample $q(x_t)$ after $t$ steps. The square root term isn't justified, but I think the idea is to prevent the norm of $q(x_t)$ from growing too much.

</details>

Let's implement the forward process using what we know so far!


```python
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Optional, Union
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import utils
import wandb

MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")

```

### Image Processing

We'll first generate a toy dataset of random color gradients, and train the model to be able to recover them. This should be an easy task because the structure in the data is simple.


```python
def gradient_images(n_images: int, img_size: tuple[int, int, int]) -> t.Tensor:
    """
    Generate n_images of img_size, each a color gradient
    """
    (C, H, W) = img_size
    corners = t.randint(0, 255, (2, n_images, C))
    xs = t.linspace(0, W / (W + H), W)
    ys = t.linspace(0, H / (W + H), H)
    (x, y) = t.meshgrid(xs, ys, indexing="xy")
    grid = x + y
    grid = grid / grid[-1, -1]
    grid = repeat(grid, "h w -> b c h w", b=n_images, c=C)
    base = repeat(corners[0], "n c -> n c h w", h=H, w=W)
    ranges = repeat(corners[1] - corners[0], "n c -> n c h w", h=H, w=W)
    gradients = base + grid * ranges
    assert gradients.shape == (n_images, C, H, W)
    return gradients / 255


def plot_img(img: t.Tensor, title: Optional[str] = None) -> None:
    if IS_CI:
        return
    img = rearrange(img, "c h w -> h w c")
    plt.imshow(img.numpy())
    if title:
        plt.title(title)
    plt.show()


if MAIN:
    print("A few samples from the input distribution: ")
    img_shape = (3, 16, 16)
    n_images = 5
    imgs = gradient_images(n_images, img_shape)
    for i in range(n_images):
        plot_img(imgs[i])

```

### Normalization

Each input value is from [0, 1] right now, but it's easier for the neural network to learn if we center them so they have mean 0. Here are some helper functions to do that, and to recover our original image.

All our computations will operate on normalized images, and we'll denormalize whenever we want to plot the result.


```python
def normalize_img(img: t.Tensor) -> t.Tensor:
    return img * 2 - 1


def denormalize_img(img: t.Tensor) -> t.Tensor:
    return ((img + 1) / 2).clamp(0, 1)


if MAIN:
    plot_img(imgs[0], "Original")
    plot_img(normalize_img(imgs[0]), "Normalized")
    plot_img(denormalize_img(normalize_img(imgs[0])), "Denormalized")

```

### Variance Schedule

The amount of noise to add at each step is called $\beta$.

Compute the vector of $\beta$ according to Section 4 of the paper. They use 1000 steps, but when reproducing a paper it's a good idea to start by making everything smaller in order to have faster feedback cycles, so we're using only 200 steps here.



```python
def linear_schedule(max_steps: int, min_noise: float = 0.0001, max_noise: float = 0.02) -> t.Tensor:
    """Return the forward process variances as in the paper.

    max_steps: total number of steps of noise addition
    out: shape (step=max_steps, ) the amount of noise at each step
    """
    pass


if MAIN:
    betas = linear_schedule(max_steps=200)

```

### Forward (q) function - Equation 2

Implement Equation 2 in code. Literally use a for loop to implement the q function iteratively and visualize your results. After 50 steps, you should barely be able to make out the colors of the gradient. After 200 steps it should look just like random Gaussian noise.

Hint: you can use `torch.normal` or `torch.randn`.

<details>

<summary>I can still see the gradient pretty well after 200 steps.</summary>

The beta value indicates the variance of the normal distribution - did you forget a square root to convert it to standard deviation?

</details>


```python
def q_eq2(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    """Return the input image with num_steps iterations of noise added according to schedule.
    x: shape (channels, height, width)
    schedule: shape (T, ) with T >= num_steps

    out: shape (channels, height, width)
    """
    pass


if MAIN:
    x = normalize_img(gradient_images(1, (3, 16, 16))[0])
    for n in [1, 10, 50, 200]:
        xt = q_eq2(x, n, betas)
        plot_img(denormalize_img(xt), f"Equation 2 after {n} step(s)")
    plot_img(denormalize_img(t.randn_like(xt)), "Random Gaussian noise")

```

### Forward (q) function - Equation 4

Equation 2 is very slow and would be even slower if we went to 1000 steps. Conveniently, the authors chose to use Gaussian noise and a nice closed form expression exists to go directly to step t without needing a for loop. Implement Equation 4 and verify it looks visually similar to Equation 2.


```python
def q_eq4(x: t.Tensor, num_steps: int, betas: t.Tensor) -> t.Tensor:
    """Equivalent to Equation 2 but without a for loop."""
    pass


if MAIN:
    for n in [1, 10, 50, 200]:
        xt = q_eq4(x, n, betas)
        plot_img(denormalize_img(xt), f"Equation 4 after {n} steps")

```

Our image reconstruction process will depend on the noise schedule we use during training. So that we can save our noise schedule with our model later, we'll define a `NoiseSchedule` class that subclasses `nn.Module`.



```python
class NoiseSchedule(nn.Module):
    betas: t.Tensor
    alphas: t.Tensor
    alpha_bars: t.Tensor

    def __init__(self, max_steps: int, device: Union[t.device, str]) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.device = device
        pass

    @t.inference_mode()
    def beta(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        """
        Returns the beta(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        """
        pass

    @t.inference_mode()
    def alpha(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        """
        Returns the alphas(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        """
        pass

    @t.inference_mode()
    def alpha_bar(self, num_steps: Union[int, t.Tensor]) -> t.Tensor:
        """
        Returns the alpha_bar(s) corresponding to a given number of noise steps
        num_steps: int or int tensor of shape (batch_size,)
        Returns a tensor of shape (batch_size,), where batch_size is one if num_steps is an int
        """
        pass

    def __len__(self) -> int:
        return self.max_steps

```

Now we'll use this noise schedule to apply noise to our generated images. This will be the batched version of `q_eq4`.


```python
def noise_img(
    img: t.Tensor, noise_schedule: NoiseSchedule, max_steps: Optional[int] = None
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Adds a random number of steps of noise to each image in img.

    img: An image tensor of shape (B, C, H, W)
    noise_schedule: The NoiseSchedule to follow
    max_steps: if provided, only perform the first max_steps of the schedule

    Returns a tuple composed of:
    num_steps: an int tensor of shape (B,) of the number of steps of noise added to each image
    noise: the unscaled, standard Gaussian noise added to each image, a tensor of shape (B, C, H, W)
    noised: the final noised image, a tensor of shape (B, C, H, W)
    """
    (B, C, H, W) = img.shape
    pass


if MAIN:
    noise_schedule = NoiseSchedule(max_steps=200, device="cpu")
    img = gradient_images(1, (3, 16, 16))
    (num_steps, noise, noised) = noise_img(normalize_img(img), noise_schedule, max_steps=10)
    plot_img(img[0], "Gradient")
    plot_img(noise[0], "Applied Unscaled Noise")
    plot_img(denormalize_img(noised[0]), "Gradient with Noise Applied")

```

Later, we'd like to reconstruct images for logging purposes. If we pass the true noise to this function, it will compute the inverse of `noise_img()` above.

During training, we'll pass the predicted noise and we'll be able to visually see how close the prediction is.


```python
def reconstruct(noisy_img: t.Tensor, noise: t.Tensor, num_steps: t.Tensor, noise_schedule: NoiseSchedule) -> t.Tensor:
    """
    Subtract the scaled noise from noisy_img to recover the original image. We'll later use this with the model's output to log reconstructions during training. We'll use a different method to sample images once the model is trained.

    Returns img, a tensor with shape (B, C, H, W)
    """
    (B, C, H, W) = noisy_img.shape
    pass


if MAIN:
    reconstructed = reconstruct(noised, noise, num_steps, noise_schedule)
    denorm = denormalize_img(reconstructed)
    plot_img(img[0], "Original Gradient")
    plot_img(denorm[0], "Reconstruction")
    utils.allclose(denorm, img)

```

Now, we'll create a tiny model to use as our diffusion model. We'll use a simple two-layer MLP.

Note that we setup our `DiffusionModel` class to subclass `nn.Module` and the abstract base class (ABC). All ABC does for us is raise an error if subclasses forget to implement the abstract method `forward`. Later, we can write our training loop to work with any `DiffusionModel` subclass.

<details>
<summary>How should we handle num_steps in the forward pass?</summary>
You can scale num_steps down to [0, 1] and concatenate it to the flattened image.
</details>


```python
class DiffusionModel(nn.Module, ABC):
    img_shape: tuple[int, ...]
    noise_schedule: Optional[NoiseSchedule]

    @abstractmethod
    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        ...


@dataclass(frozen=True)
class TinyDiffuserConfig:
    img_shape: tuple[int, ...]
    hidden_size: int
    max_steps: int


class TinyDiffuser(DiffusionModel):
    def __init__(self, config: TinyDiffuserConfig):
        """
        A toy diffusion model composed of an MLP (Linear, ReLU, Linear)
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.img_shape = config.img_shape
        self.noise_schedule = None
        self.max_steps = config.max_steps
        pass

    def forward(self, images: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        """
        Given a batch of images and noise steps applied, attempt to predict the noise that was applied.
        images: tensor of shape (B, C, H, W)
        num_steps: tensor of shape (B,)

        Returns
        noise_pred: tensor of shape (B, C, H, W)
        """
        pass


if MAIN:
    img_shape = (3, 4, 5)
    n_images = 5
    imgs = gradient_images(n_images, img_shape)
    n_steps = t.zeros(imgs.size(0))
    model_config = TinyDiffuserConfig(img_shape, 16, 100)
    model = TinyDiffuser(model_config)
    out = model(imgs, n_steps)
    plot_img(out[0].detach(), "Noise prediction of untrained model")

```

### Training Loop

After a pile of math, the authors arrive at Equation 14 for the loss function and Algorithm 1 for the training procedure. We're going to skip over the derivation for now and implement the training loop at the top of Page 4.

Exercise: go through each line of Algorithm 1, explain it in plain English, and describe the shapes of each thing.

<details>

<summary>Solution - Line 2</summary>

The $x_0$ is just the original training data distribution, so we're just going to draw a minibatch from the training data of shape (batch, channels, height, width).

</details>

<details>

<summary>Solution - Line 3</summary>

We need to draw the number of steps of noise to add for each element of the batch, so the $t$ here will have shape (batch,) and be an integer tensor. Both 1 and T are inclusive here. Each element gets a different number of steps of noise added.

</details>

<details>

<summary>Solution - Line 4</summary>

$\epsilon$ is the sampled noise, not scaled by anything. It's going to add to the image, so its shape also has to be (batch, channel, height, width).

</details>

<details>

<summary>Solution - Line 5</summary>

$\epsilon_\theta$ is our neural network. It takes two arguments: the image with noise applied in one step of (batch, channel, height, width), and the number of steps (batch, ), normalized to the range [0, 1].

</details>

In Line 6 - it's unspecified how we know if the network is converged. We're just going to go until the loss seems to stop decreasing.

Now implement the training loop on minibatches of examples, using Adam as the optimizer. Log your results to Weights and Biases.


```python
def log_images(
    img: t.Tensor, noised: t.Tensor, noise: t.Tensor, noise_pred: t.Tensor, reconstructed: t.Tensor, num_images: int = 3
) -> list[wandb.Image]:
    """
    Convert tensors to a format suitable for logging to Weights and Biases. Returns an image with the ground truth in the upper row, and model reconstruction on the bottom row. Left is the noised image, middle is noise, and reconstructed image is in the rightmost column.
    """
    actual = t.cat((noised, noise, img), dim=-1)
    pred = t.cat((noised, noise_pred, reconstructed), dim=-1)
    log_img = t.cat((actual, pred), dim=-2)
    images = [wandb.Image(i) for i in log_img[:num_images]]
    return images


def train(
    model: DiffusionModel, config_dict: dict[str, Any], trainset: TensorDataset, testset: Optional[TensorDataset] = None
) -> DiffusionModel:
    wandb.init(project="diffusion_models", config=config_dict, mode="disabled" if IS_CI else "enabled")
    config = wandb.config
    print(f"Training with config: {config}")
    pass


if MAIN:
    config: dict[str, Any] = dict(
        lr=0.001,
        image_shape=(3, 4, 5),
        hidden_size=128,
        epochs=20 if not IS_CI else 1,
        max_steps=100,
        batch_size=128,
        img_log_interval=200,
        n_images_to_log=3,
        n_images=50000 if not IS_CI else 10,
        n_eval_images=1000 if not IS_CI else 10,
        device=t.device("cuda") if t.cuda.is_available() else t.device("cpu"),
    )
    images = normalize_img(gradient_images(config["n_images"], config["image_shape"]))
    train_set = TensorDataset(images)
    images = normalize_img(gradient_images(config["n_eval_images"], config["image_shape"]))
    test_set = TensorDataset(images)
    model_config = TinyDiffuserConfig(config["image_shape"], config["hidden_size"], config["max_steps"])
    model = TinyDiffuser(model_config).to(config["device"])
    model = train(model, config, train_set, test_set)

```

## Sampling from the Model

Our training loss went down, so maybe our model learned something. Implement sampling from the model according to Algorithm 2 so we can see what the images look like.



```python
def sample(model: DiffusionModel, n_samples: int, return_all_steps: bool = False) -> Union[t.Tensor, list[t.Tensor]]:
    """
    Sample, following Algorithm 2 in the DDPM paper

    model: The trained noise-predictor
    n_samples: The number of samples to generate
    return_all_steps: if true, return a list of the reconstructed tensors generated at each step, rather than just the final reconstructed image tensor.

    out: shape (B, C, H, W), the denoised images
    """
    schedule = model.noise_schedule
    assert schedule is not None
    pass


if MAIN:
    print("Generating multiple images")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 5)
    for s in samples:
        plot_img(denormalize_img(s).cpu())
if MAIN:
    print("Printing sequential denoising")
    assert isinstance(model, DiffusionModel)
    with t.inference_mode():
        samples = sample(model, 1, return_all_steps=True)
    for i, s in enumerate(samples):
        if i % (len(samples) // 20) == 0:
            plot_img(denormalize_img(s[0]), f"Step {i}")

```

Now that we've got the training working, on to part 2!
