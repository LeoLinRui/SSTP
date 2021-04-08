# Weekly Journal -- Project Barker

## Previous Work
In the past month, significant work has been put into the project. Although without the official approval of the committee, we’ve each been spending more than 20 hours per week to kickstart the project. Significant progress has been made, particularly in the computer science aspect of the project.

We have successfully trained StyleGAN2-ADA with multiple datasets we’ve collected and processed, including Time covers, Emojis, and Chinese Characters. We’ve benchmarked multiple variants of StyleGAN’s implementations’ (including ffhq128, 256, 512, and 1024; implemented in TensorFlow and Pytorch) performance (training time) on multiple setups including my home set-up (2x RTX2070) and multiple cloud server settings (RTX3080, RTX3090, Tesla P100, Tesla V100). The samples below are the generated images our model produced (no editing is done on the outputs.)

We’ve also written the code to generate transitioning videos from the models represented above (transition latent vector generation and streamlined model inference.) Here is an example of what the code is capable of producing.

### Summary
StyleGan-based Image Generation Models
- A working StyleGan Model to generate Time Covers at 1024x1024
- A working StyleGan Model to generate Emojis at 256x256

GPT-2-based Text Generation Models
- A working GPT-2 Model (700M) to generate New York Times articles
- A working GPT-2 Model (700M) to generate Russian interference tweets in 2016 election

### Samples
![fakes000320](https://user-images.githubusercontent.com/44302577/109961804-6499e000-7d25-11eb-94fb-e51e79bcc87e.png)

### Misc

## Week 1
Apr. 5 --> Apr. 12

We have decided on a direction change, switching from film production to designing a web application. The web app is a Twitter clone filled with generated contents. Most of the visual content that we can generate is atemporal, which makes it difficult to create a coherent video. Instead, we thought, what’s short, incoherent, and people are bombarded with? Tweets. So, we made a fake Twitter. We have made the app using the React framework in JavaScript; the backend is made with the Django framework in Python.
### Summary
#### Minh
#### Leo
### Change Log

### Samples
https://twitter-clone-sstp.ue.r.appspot.com/

### Misc
