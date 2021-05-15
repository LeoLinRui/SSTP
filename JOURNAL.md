# Weekly Journal -- [Project Barker](https://twitter-clone-sstp.appspot.com/)

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

## Week 4
Apr. 26 --> May 3

### Summary
#### Minh
- Work around our various mistakes in saving the processed generic tweets.
  -   Mistakes include but are not limited to: putting data in a row instead of column, appending a field separator after *every letter*, praying to false idols.
  -   Look at this terribleness ![terribleness](https://user-images.githubusercontent.com/64875104/116816698-e0db5480-ab17-11eb-859b-c7fb85ba9e30.png)
  -   Fixes would've been much quicker if I didn't try to edit gigabytes of data off of a cloud drive. Curse my data-loss paranoia (and my ISP).
- Trained NLP model on three batches of data. ("Batch" does have a technical meaning which I am electing to ignore).
  -   Actually not trivial because it keeps breaking thanks to our great job with saving/writing the data.
- Learned multithreading for data processing in Python using multiprocessing and p_tqdm libraries
  -   Also learned (the hard way) to not use it unnecessarily. Python does not love multithreading.


#### Leo
The meme dataset urls seems to be broken still and I couldn't figure out a way to solve that issue. I'm working on fining other good ones.
Most of my time this week is spent on dealing with generic tweets. I had to babysit the download for a decent while before I can do the processing. As Minh has mentioned in his journal entry, the processing keeps giving weird errors that we had to fix. I added a few new pieces to the pre-processing so we can better filter out short tweets and tweets that did not finish in one post. After all that debugging we are training the generic tweets generator now, which will make up the bulk of our final product. Below are some early examples of generic tweets we have generated.
![WeChat Image_20210503190939](https://user-images.githubusercontent.com/44302577/116869367-32ccc900-ac43-11eb-8da2-fa916169f40b.png)
After a few more attempts of fixing the StyleGan clone, I've given up. I got Colab Pro. I'm moving all the training data onto Google drive.

## Week 5
May. 3 --> May. 10

### Summary
#### Minh
Twitter clone app:
- Set up system and pipeline to support full-length generated news articles.
  -   Added new table in database for articles, which supports tweet-article pairing.
  -   Added new API endpoint to retrieve aforementioned news articles.
  -   Added frontend routing system to move from the tweet feed to individual news articles.
  -   Delegated aesthetic design of the news-site to Leo. We can now share in the suffering of CSS.
-   Removed (now-useless) API endpoints.
  -   Thanks to optimizations in the requesting system, what used to need four separate API calls only requires one.
  -   So much work... down the drain. It's a learning experience... I suppose.
-   Researched methods of hosting and streaming videos on the web.
  -   Apparently throwing around gigabytes of data all over the world on demand is complicated and expensive.
  -   YouTube will have to do.

Data/ML stuff:
- Retrained GenericTweet model after it got corrupted.
  -   Google probably isn't happy that I'm using so much GPU time.
- Aquired and processed dataset of CCTV videos.
- Used YOLOv5 (You Only Look Once version 5) to create computer-vision-generated bounding boxes.
  -   A message on both the ability of AI and the degree to which we are being watched.
  -   [sample](https://drive.google.com/file/d/10prLk079GQFHp5w9rC36VUVUDaDBHYvR/view?usp=sharing)
  -   CS nerds think of the best names.


#### Leo
- Set up the training environments and the pre-processing pipeline on Colab for training the Trump Images with StyleGAN Tensorflow. Completed ~60 hours of training and tweaking hyperparameters, achieving a satisfacoty result that can be used in production. Sample below: ![image](https://user-images.githubusercontent.com/44302577/117583609-a6d30980-b13a-11eb-81e0-951b7342fc57.png)
- Completed a few tests with Attention GAN for text-to-image generation, which will be used to generated images for a certain portion of the tweets in our project. Explored the possibility of using it on RunwayML browser app and desktop app, also looking into the possibility of utilizing the original repo. Sample: 
![image](https://user-images.githubusercontent.com/44302577/117583873-34fbbf80-b13c-11eb-9872-198655491ed7.png)
- Designed the logo for the project ("Barker") ![image](https://user-images.githubusercontent.com/44302577/117583620-bd796080-b13a-11eb-8560-000e26ba63e4.png)
- Designed the webpage for the generated news articles. (includes learning Adobe Dreamweaver, a little bit of HTML and CSS.) Sample: ![image](https://user-images.githubusercontent.com/44302577/117583790-bd2d9500-b13b-11eb-86e7-1a73b9a5cb21.png)
- Generated, with StyleGAN2 on FFHQ, 1000 faces for use as profile photos of the user of the tweet. Sample below:![image](https://user-images.githubusercontent.com/44302577/117583693-337dc780-b13b-11eb-939b-1f3afd3fb0b7.png)
- An extensive meeting with Minh (3+ hours) to recap the status of all current products, determine the final layout and graphic design details of the site, disucuss the production setup of image databases, and decide the design requirements for the various news article "card" sections on the site, which I will be designing next week.
- Downloaded and processed some more generic tweets data for final fine-tuning of the generic tweet model.


## Week 6
May. 10 --> May. 17

### Summary
#### Minh
Twitter clone app:
- Completed UI + backend for news-site
  -   Integrated Leo's Dreamweaver design into site
  -   It actually looks pretty good
- Improved UI/UX for mobile users
  -   Mobile users have a three-button top bar for the same functionality as desktop
    -   Functionality of the buttons still needs to be implemented
  -   Video player now *finally* scales properly
    -   After much praying
- Added option to not normalize text encodings in database loader
  -   Enconding normalizer didn't always work well
- Added 37k generic tweets (still AI-generated) to app
  -   The experience feels much more realistic (and off-putting)
  -   As expected, there is at least some profanity
- Registered and partly set up Firebase service to host our own images
  -   In progress. Not guaranteed to work.

Data/ML stuff:
- Trained two AI models to generate news articles
  -   Model 1, LeftNews: trained on CNN, BuzzFeed, and Vox news articles
  -   Model 2, RightNews: trained on Fox, Breitbart, and National Review articles
  -   Should be the last models

#### Leo
