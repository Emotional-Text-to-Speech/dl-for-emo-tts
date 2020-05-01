# Deep Learning for Emotional Text-to-speech
A summary on our attempts at using Deep Learning approached for Emotional Text to Speech



# Emotional Text-to-speech with Tacotron and related models

## Motivation

Even though our project was cetered around using HMMs for Speech synthsis, we were encouraged to try out alternate approaches for the same. Hence, we started this separate thread of DL Approaches for Emotional TTS.

### Preliminaries

Before we start out with Deep Learning based approaches for TTS, it is essential to learn about the data resources and the possible models for that have been developed for neural TTS and can be adapted for Emotional TTS. In this section, we briefly talk about the resources we explored in terms of both datasets and existing neural architectures, with a link to more elaborate resources.

#### Datasets

| Dataset | Comments | Pros | Cons|
| -- | -- | -- | -- |
| **[IEMOCAP](https://sail.usc.edu/iemocap/)** | This dataset consists of 10 actors (5 male; 5 female) of multi-modal dyadic conversations, which were scripted and enacted by a group of actors. | Variety of utterances, rich vocabulary, multi-modal input, emotion annotations | The access is very restricted and upon being granted access, we got a corpus archive file. |
| **[RAVDESS](https://zenodo.org/record/1188976#.Xqw8ntMvPBI)** | This dataset consists of 24 actors (12 female, 12 male), speaking in 8 emotions (calm, neutral, happy, sad, angry, fearful, surprise, and disgust). Each speaker has 4 utterances for neutral emotion and 8 utterances for all other emotions, leading to 60 utterances per speaker | Easily available, smaller dataset. | Very limited utterances, poor vocabulary, same utterance in different voices |
| **[EMOV-DB](https://github.com/numediart/EmoV-DB)** | An attempt at a large scale corpus for Emotional speech. It consists of | | |
### Approach 1

