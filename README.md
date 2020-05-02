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
| **[EMOV-DB](https://github.com/numediart/EmoV-DB)** | 5 (3 male, 2 female) | 5 (neutral, amused, angry sleepy, disgust) |  6914 (1568, 1315, 1293, 1720, 1018) | 1150 |  | English, French (1 male speaker) |An attempt at a large scale corpus for Emotional speech. The Amused emotion contains non-verbal cues like chuckling, etc. which do not show up in the transcript. Similarly, Sleepiness has yawning sounds. | Only large scale emotional corpus that we found freely available. | Emotions covered are not very standard. The non-verbal cues make synthsis difficult. Also, not all emotions are available for all speakers. |
| **[IEMOCAP](https://sail.usc.edu/iemocap/)** | 10 (5 female, 5 male) | 9 (anger, happiness, excitement, sadness, frustration, fear, surprise, other and neutral state) | 10039 | NA | 12.5 hours | English |This dataset consists of 10 actors (5 male; 5 female) of multi-modal dyadic conversations, which were scripted and enacted by a group of actors. | Variety of utterances, rich vocabulary, multi-modal input, emotion annotations | The access is very restricted and upon being granted access, we got a corpus archive file. |
### Approach 1

