# Deep Learning for Emotional Text-to-speech
A summary on our attempts at using Deep Learning approached for Emotional Text to Speech



# Emotional Text-to-speech with Tacotron and related models

## Motivation

Even though our project was cetered around using HMMs for Speech synthsis, we were encouraged to try out alternate approaches for the same. Hence, we started this separate thread of DL Approaches for Emotional TTS.

## Preliminaries

Before we start out with Deep Learning based approaches for TTS, it is essential to learn about the data resources and the possible models for that have been developed for neural TTS and can be adapted for Emotional TTS. In this section, we briefly talk about the resources we explored in terms of both datasets and existing neural architectures, with a link to more elaborate resources.

### Datasets

| Dataset | No. of Speakers | Emotions | No. of utterances | No. of unique prompts | Duration | Language | Comments | Pros | Cons|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **[RAVDESS](https://zenodo.org/record/1188976#.Xqw8ntMvPBI)** | 24 (12 female, 12 male) | 8 (calm, neutral, happy, sad, angry, fearful, surprise, and disgust) | 1440 | 2 | ~1 hour | English | This dataset consists of 24 actors (12 female, 12 male), speaking in 8 emotions (calm, neutral, happy, sad, angry, fearful, surprise, and disgust). Each speaker has 4 utterances for neutral emotion and 8 utterances for all other emotions, leading to 60 utterances per speaker | Easily available, smaller dataset. | Very limited utterances, poor vocabulary, same utterance in different voices |
| **[EMOV-DB](https://github.com/numediart/EmoV-DB)** | 5 (3 male, 2 female) | 5 (neutral, amused, angry sleepy, disgust) |  6914 (1568, 1315, 1293, 1720, 1018) | 1150 | ~7 hours | English, French (1 male speaker) |An attempt at a large scale corpus for Emotional speech. The Amused emotion contains non-verbal cues like chuckling, etc. which do not show up in the transcript. Similarly, Sleepiness has yawning sounds. | Only large scale emotional corpus that we found freely available. | Emotions covered are not very standard. The non-verbal cues make synthsis difficult. Also, not all emotions are available for all speakers. |
| **[LJ Speech](https://keithito.com/LJ-Speech-Dataset/)** | 1 (1 female) | NA (can be considered neutral) | 13100 | 13100 | 	23 hours 55 minutes 17 seconds | English | This is one of the largest corpuses for speech generation, with a rich vocabulary of over ~14k unique words. The sentences are taken from 7 non-fiction books. | Large scale corpus. Rich vocabulary. Abbreviations in text are expanded in speech. | No emotions. | 
| **[IEMOCAP](https://sail.usc.edu/iemocap/)** | 10 (5 female, 5 male) | 9 (anger, happiness, excitement, sadness, frustration, fear, surprise, other and neutral state) | 10039 | NA | 12.5 hours | English |This dataset consists of 10 actors (5 male; 5 female) of multi-modal dyadic conversations, which were scripted and enacted by a group of actors. | Variety of utterances, rich vocabulary, multi-modal input, emotion annotations | The access is very restricted and upon being granted access, we got a corpus archive file. |

### Relevant literature

- **[Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)**
  - An extremely influential paper in the are of Neural Text-to-speech. The idea can be abstracted to a simple encoder-decoder network, that takes as input the ground-truth audio and textual transcript. 
  - The reconstruction loss of the generated audio drives the training of the model. 
  - This was one of the architectures that we explored in this project. We also presented details about this paper in a class lecture. [\[slides\]](https://docs.google.com/presentation/d/1MnwhzJmzH689NivBOFdQ60BhUEr3WtGAC_UBML0dS2E/edit?usp=sharing)
- **[Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis
](https://arxiv.org/abs/1803.09017)**
  - This work was done by the same team that developed Tacotron.
  - The core idea was to improve the expressiveness of the generated speech, by incorporating "Style Tokens" which was basically an additional embedding layer for the ground-truth audio, which was used to condition the generated audio, so that transfer of "prosodic features" could occur. 
  - We also explored this model, and presented it for a class lecture. [\[slides\]](https://docs.google.com/presentation/d/1aug9OmIrd8nDY4BmyIK_z65D9DKP5cbrjgX4aDBngFU/edit?usp=sharing)
  - However, we did not explore this as extensively as the Tacotron, as it took a lot of time and resources to train.
- **[Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969)**
  - This work aimed at more efficient text-to-speech generation by using fully convolutional layers with guided attention.
  - We came across this work while looking for resources for efficient TTS systems that could be fine-tunes with low amount of data.
  
There are many more relevant papers that build up on the Vanilla Tacotron model. However, for the scope of our project, we restricted ourselves to these three papers.

## Approaches explored

### :x: Approach 1: Fine-tuning a Vanilla Tacotron model on RAVDESS that was pre-trained on LJ Speech

Our first approach was to train a vanilla Tacotron model from scratch on just one emotion (say, anger) and see if the generated voice has captured the prosodic features of that emotion.

#### Motivation

- We did not have acccess to any of the datasets described above except for RAVDESS and LJ Speech and had also never tried any of the Tacotron-flavored models before.
- Hence, we just wanted to play around initially and at least generate the results on LJ Speech, and analyse the quality of speech generated.
- The fine-tuning idea seemed natural after the pre-training was done, as the RAVDESS dataset was extremely limited, there was no point on training on it from scratch, as the vocabulary that the model was exposed to would be extremely low.
- We were hoping that at best, less amount of fine-tuning would lead to transfer of prosodic features to the model and at worst, after fine-tuning for a long interval, would lead to over-fitting on the dataset.

#### Observations 

- The alignment of the encoder and decoder states was completely destroyed in the first 1000 iterations of training itself.
- At test time, generated audio was initially empty. On further analysis, we discovered that this was because of the way the decoder stopped. 
- If the all the values in the generated frames were below a certain threshold, the decoder would stop producing the new frames. We observed that in our case, this was happening at the beginning itself.
- To fix this, we removed this condition and instead made the decoder produce sounds for a minimum number of iterations.
- We observed that for lesser iterations of finetuning (1-3k iterations) the audio was produced was complete noise, with no intelligible speech.
- If we fine-tune for long durations (~40k iterations), we observe that the model is able to generate angry speech for the utterances that are in the training set. However, even for utterances outside the training set, it speaks parts of the training set utterances only, indicating that the model has overfitted on this dataset.

#### Inference and next steps

- The observations presented above seemed to present a case of "*catastrophic forgetting*" where the model was forgetting the information that it had already learnt in the pre-training rates.
- To counter this, we were advised to tweak the hyperparameters and training strategy of the model, such as learning rate, optimiser used, etc. 
- We decided to try out this following approaches:
  - **Start the fine-tuning steps with a lower learning rate**: Pre-training was done at 0.002, so we decided to do fine-tuning with 2e-5. Note that the code also implemented alleaning learning rate strategy, where learning rate was reduced after few steps. We did not change it as it had given good results at pre-training.
  - **Changing the optimizer from Adam to SGD**: Because the number of samples used for fine-tuning were less, and SGD has been known to generalise better for a smaller sample size, we decided to do this.
  - **Freezing the Encoder of the Tacotron while fine-tuning**: We thought of this because the main purposed of the encoder is to convert the text to a latent space. Since LJ Speech had a better vocabulary either way, we did not feel the need to re-train this component of the model over RAVDESS' much inferior voabulary size.

### :x: Approach 2: Using a smaller learning rate for fine-tuning

In this approach, we repeated [Approach 1](###Approach-1:-Fine-tuning-a-Vanilla-Tacotron model-on-RAVDESS-that-was-pre-trained-on-LJ-Speech)


