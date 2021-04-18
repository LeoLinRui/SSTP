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
- A working GPT-2 Model (700M) to generate Russian interference tweets in 2016 election (from the Internet Research Agency)

### Samples
![fakes000320](https://user-images.githubusercontent.com/44302577/109961804-6499e000-7d25-11eb-94fb-e51e79bcc87e.png)

### Misc
- A toolbox for interacting with StyleGAN
- Scripts for processing texts and videos
- A rough draft of a digital video generation library

## Week 1
Apr. 5 --> Apr. 12

We have decided on a direction change, switching from film production to designing a web application. The web app is a Twitter clone filled with generated contents. Most of the visual content that we can generate is atemporal, which makes it difficult to create a coherent video. Instead, we thought, what’s short, disjointed, and with which people are bombarded? Tweets. So, we made a fake Twitter. We made the app using the React framework in JavaScript; the backend is made with the Django framework in Python; the [app](https://twitter-clone-sstp.ue.r.appspot.com/) is hosted on Google Cloud App Engine.
### Summary
#### Minh
- Generated ~3000 fake Tweets in the style of the Internet Research Agency.
- Created loader with (rudimentary) filtering to transfer Tweets from AI models to backend.
- Loaded the (~3000) Tweets to database.
- Finished backend (not frontend) of separate perspectives system. 
![Dataflow between backend and frontend to retrieve posts](https://user-images.githubusercontent.com/64875104/114307141-2c5b8f00-9a93-11eb-91bf-d93eea85b364.png)
- Added frontend support for media from Youtube, Facebook, SoundCloud, Vimeo, Dailymotion, and Twitch.
- Sacrificed three hours and my secondborn to the CSS gods. Media (videos and images) no longer breaks responsive design.

#### Leo
I have spent most of the time this week to continue working on a new video-generation model that I will call VFormer for now. It is a new model based on the transformer decoder structure that utilize self-attention for causal language modeling. I'm using the Reformer model (transfomrer with locality-sensitive hashing and a two-way network for improved computational efficiency on longer input sequences.) There's no quarantee that this will work but since we have completed the basics of our project before it officially started, some new research is needed if we want to further improve our project.

I also researched OpenAI's DALL-E and CLIP models for the possibility of adding a new feature to our project: automatic text-based image generation.

### Change Log
- Finished building Vformer's basic modules

### Samples
https://twitter-clone-sstp.ue.r.appspot.com/

### Misc
We met with Mr. Oxton's son, Timothy Oxton, who is a photographer currently working on his independent game. We had a 90-minute discussion about new digital technologies in art-making and the various aspects of designing a virtual experience for the viewer.

## Week 2
Apr. 12 --> Apr. 19

### Summary
#### Minh

#### Leo


### Change Log


### Samples
https://twitter-clone-sstp.ue.r.appspot.com/

### Misc
