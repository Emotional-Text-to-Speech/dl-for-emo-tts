# Deep Learning for Emotional Text-to-speech
A summary on our attempts at using Deep Learning approached for Emotional Text to Speech



# Emotional Text-to-speech with Tacotron and related models

## Motivation

Even though our project was cetered around using HMMs for Speech synthsis, we were encouraged to try out alternate approaches for the same. Hence, we started this separate thread of DL Approaches for Emotional TTS.

### Preliminaries

Before we start out with Deep Learning based approaches for TTS, it is essential to learn about the data resources and the possible models for that have been developed for neural TTS and can be adapted for Emotional TTS. In this section, we briefly talk about the resources we explored in terms of both datasets and existing neural architectures, with a link to more elaborate resources.

#### Datasets

| Dataset | No. of Speakers | Emotions | No. of utterances | No. of unique prompts | Duration | Language | Comments | Pros | Cons|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **[RAVDESS](https://zenodo.org/record/1188976#.Xqw8ntMvPBI)** | 24 (12 female, 12 male) | 8 (calm, neutral, happy, sad, angry, fearful, surprise, and disgust) | 1440 | 2 | ~1 hour | English | This dataset consists of 24 actors (12 female, 12 male), speaking in 8 emotions (calm, neutral, happy, sad, angry, fearful, surprise, and disgust). Each speaker has 4 utterances for neutral emotion and 8 utterances for all other emotions, leading to 60 utterances per speaker | Easily available, smaller dataset. | Very limited utterances, poor vocabulary, same utterance in different voices |
| **[EMOV-DB](https://github.com/numediart/EmoV-DB)** | 5 (3 male, 2 female) | 5 (neutral, amused, angry sleepy, disgust) |  6914 (1568, 1315, 1293, 1720, 1018) | 1150 | ~7 hours | English, French (1 male speaker) |An attempt at a large scale corpus for Emotional speech. The Amused emotion contains non-verbal cues like chuckling, etc. which do not show up in the transcript. Similarly, Sleepiness has yawning sounds. | Only large scale emotional corpus that we found freely available. | Emotions covered are not very standard. The non-verbal cues make synthsis difficult. Also, not all emotions are available for all speakers. |
| **[IEMOCAP](https://sail.usc.edu/iemocap/)** | 10 (5 female, 5 male) | 9 (anger, happiness, excitement, sadness, frustration, fear, surprise, other and neutral state) | 10039 | NA | 12.5 hours | English |This dataset consists of 10 actors (5 male; 5 female) of multi-modal dyadic conversations, which were scripted and enacted by a group of actors. | Variety of utterances, rich vocabulary, multi-modal input, emotion annotations | The access is very restricted and upon being granted access, we got a corpus archive file. |

#### Relevant literature

- **[Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)**
  - An extremely influential paper in the are of Neural Text-to-speech. The idea can be abstracted to a simple encoder-decoder network, that takes as input the ground-truth audio and textual transcript. The reconstruction loss of the generated audio drives the training of the model. This was one of the architectures that we explored in this project.
- **[Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis
](https://arxiv.org/abs/1803.09017)**
  - This work was done by the same team that developed Tacotron.
  - The core idea was to improve the expressiveness of the generated speech, by incorporating "Style Tokens" which was basically an additional embedding layer for the ground-truth audio, which was used to condition the generated audio, so that transfer of "prosodic features" could occur. 
  - We also explored this model, and presented it for a class lecture. [\[slides\]](https://docs.google.com/presentation/d/1aug9OmIrd8nDY4BmyIK_z65D9DKP5cbrjgX4aDBngFU/edit?usp=sharing)
  - However, we did not explore this as extensively as the Tacotron, as it took a lot of time and resources to train.
- **[Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969)**: 
  - This work aimed at more efficient text-to-speech generation by using fully convolutional layers with guided attention.
  - We came across this work while looking for resources for efficient TTS systems that could be fine-tunes with low amount of data.
  
There are many more relevant papers that build up on the Vanilla Tacotron model. However, for the scope of our project, we restricted ourselves to these three papers.



### Approach 1

