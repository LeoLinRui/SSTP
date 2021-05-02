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
- Optimized generation resulting in 10x generation speed (as a result of previous terribleness, not new programming magic).
- Generated 50000 Tweets.
- Wrote pre-processing script for Leo's analysis, including a machine-learning algorithm to detect text encoding.
- Implemented scraper to scrape first 100 pages (6000 images) of a Getty Images search.
- Implemented Tweet loading queue. Bottoming out on a scroll while not loading any Tweets should (emphasis on "should") not happen anymore.
- Made script to automatically compile and deploy to GCP App Engine.

#### Leo
This week my work focused on figuring out ways to achieve image-generation for tweets based on the content of the tweets. First, I conducted an analysis on the vocabulary on the generated tweets, from which we determined that there is actually less variation in the nouns used in the generated tweets then we had previously imagined. I then decided to train an image generation model on Trump's photos, as it covers the most tweets from the Right Troll GPT model. I used open datasets as well as private ones, which I used custom scrapers (had to be reconfigured and debugged) and wrote processing tools (duplicate deletion via hashing, auto-padding, resizing, etc.) for. We were able to construct a dataset consisting of 30,000 Trump photos. I have obtained preliminary results on the generation of Trump's photo on StyleGAN. More fine-tuning is to be done. 

### Samples
https://twitter-clone-sstp.ue.r.appspot.com/

![image](https://user-images.githubusercontent.com/44302577/115158761-64ca2100-a0c2-11eb-89c9-54a16992da2e.png)

## Week 3
Apr. 19 --> Apr. 26

### Summary
#### Minh
Twitter clone backend:
- Generated ~60000 Tweets under multiple categories (HashtagGamer, Newsfeed, LeftTroll) of the Russian interference Twitter campaign and added to database.
- Optimized API requests system. Tweet-loading latency should be 3x faster (and quite a bit cheaper). 
  - No magic here. I just learned to include data from multiple models in one response.

Twitter clone frontend:
- Support for explanations. User can click on the username of a Tweet for the origin story of a the Tweet.
- Some UI/UX improvements including but not limited to: button-clicking sound effects, text scrambling for unused sidebar options, and flipping animation for explanation.
- Support for clickable hyperlinks in explanation. 
  -   Programmer milestone: Used Regex! (to detect links)
  -   This took too long; React really doesn't like passing HTML through a variable in JSX
  -   ![danger](https://user-images.githubusercontent.com/64875104/115997062-83cf2400-a596-11eb-84ac-4578e572beb3.png)

Data stuff:
- Finally learned to make proper HTTP requests from Python.
- Made script to download and stream The Internet Archive's Twitter corpus to Google Drive for processing (and hopefully generation).

made a nice espresso. not related. just wanted to share :)


#### Leo
This week's work includes 3 parts: 1. Fixing StyleGAN, 2. finding and pre-processing generic tweet data, and 3. finding a meme dataaset. 

The copy of StyleGAN on my computer is suddenly unable to build a cpp plug-in essential to the model's function. I've spent hours re-cloning my repo, updating CUDA and cuda toolkit, upgrading pytorch, and trying various methods found online, all without much success. I'll be spending more time on getting it to work again in the next week. 

After a long search, I was able to find a large, generic tweet dataset that includes the original text of the tweets (thus bypassing the strict Twitter API limit.) There are approximately 1TB of json files containing tweets from 2018 to 2021. After downloading some files and doing some analysis, we found that about 15% of these tweets are high-quality, usable data. I wrote the pre-processing pipeline, which filters out tweets in languages other than English, retweets. It also removes emojies from tweets and can take out tweets that are too short in order to improve training efficiency. The [code](https://github.com/LeoLinRui/SSTP/blob/main/utils/Twitter_Js_Load.py) utilizes multi-processing to speed up the pre-processing of the large dataset we have, which is pretty cool. 

I've also worked on looking for a high-quality and quantity meme dataset. I've found one containing over 50k links to images and have written a script to retrieve them. However, there seems to be some errors regarding the image files these urls points to. I'm currently working on toubleshotting these issues.

## Week 3
Apr. 26 --> May 3

### Summary
#### Minh
#### Leo
