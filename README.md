# Deep Learning for Emotional Text-to-speech
A summary on our attempts at using Deep Learning approaches for Emotional Text to Speech

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Emotional-Text-to-Speech/dl-for-emo-tts/blob/master/Demo_DL_Based_Emotional_TTS.ipynb) [![DOI](https://zenodo.org/badge/260492782.svg)](https://zenodo.org/badge/latestdoi/260492782)
![demo](assets/dl_demo_2.gif)

---

# Contents
- [Datasets](#datasets)
- [Relevant literature](#relevant-literature)
- [Approach: Tacotron Models](#approach-tacotron-models)
  - [Approach 1: Fine-tuning a Vanilla Tacotron model on RAVDESS pre-trained on LJ Speech](#x-approach-1-fine-tuning-a-vanilla-tacotron-model-on-ravdess-pre-trained-on-lj-speech)
  - [Approach 2: Using a smaller learning rate for fine-tuning](#x-approach-2-using-a-smaller-learning-rate-for-fine-tuning)
  - [Approach 3: Using a smaller learning rate and SGD for fine-tuning](#x-approach-3-using-a-smaller-learning-rate-and-sgd-for-fine-tuning)
  - [Approach 4: Freezing the encoder and postnet](#x-approach-4-freezing-the-encoder-and-postnet)
  - [Approach 5: Freezing the encoder and postnet, and switching back to Adam](#x-approach-5-freezing-the-encoder-and-postnet-and-switching-back-to-adam)
  - [Approach 6: Freezing just the post-net, using Adam with low initial learning rate, training on EMOV-DB](#white_check_mark-approach-6-freezing-just-the-post-net-using-adam-with-low-initial-learning-rate-training-on-emov-db)
- [Approach: DCTTS Models](#approach-dctts-models)
  - [Approach 7: Fine-tuning the Text2Mel module of the DC-TTS model on EMOV-DB pre-trained on LJ Speech](#x-approach-7-fine-tuning-the-text2mel-module-of-the-dc-tts-model-on-emov-db-pre-trained-on-lj-speech)
  - [Approach 8: Fine-tuning only on one speaker with reduced `top_db` and monotonic attention](#white_check_mark-approach-8-fine-tuning-only-on-one-speaker-with-reduced-top_db-and-monotonic-attention)
- [Reproducibility and Code](#reproducibility-and-code)
- [Demonstration](#demonstration)
- [Cite](#cite)
- [Contact](#contact)


---

# Datasets

| Dataset | No. of Speakers | Emotions | No. of utterances | No. of unique prompts | Duration | Language | Comments | Pros | Cons|
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **[RAVDESS](https://zenodo.org/record/1188976#.Xqw8ntMvPBI)** | 24 (12 female, 12 male) | 8 (calm, neutral, happy, sad, angry, fearful, surprise, and disgust) | 1440 | 2 | ~1 hour | English | <ul><li> Each speaker has 4 utterances for neutral emotion and 8 utterances for all other emotions, leading to 60 utterances per speaker</li></ul> | <ul><li>Easily available</li><li>Emotions contained are very easy to interpret</li></ul> | <ul><li>Very limited utterances</li><li>Poor vocabulary</li><li> Same utterance in different voices</li></ul> |
| **[EMOV-DB](https://github.com/numediart/EmoV-DB)** | 5 (3 male, 2 female) | 5 (neutral, amused, angry sleepy, disgust) |  6914 (1568, 1315, 1293, 1720, 1018) | 1150 | ~7 hours | English, French (1 male speaker) |<ul><li>An attempt at a large scale corpus for Emotional speech</li><li> The *Amused* emotion contains non-verbal cues like chuckling, etc. which do not show up in the transcript</li><li> Similarly, *Sleepiness* has yawning sounds.</li></ul> | <ul><li>Only large scale emotional corpus that we found freely available</li></ul> | <ul><li>Emotions covered are not very easy to interpret</li><li>The non-verbal cues make synthsis difficult</li><li> Also, not all emotions are available for all speakers</li></ul> |
| **[LJ Speech](https://keithito.com/LJ-Speech-Dataset/)** | 1 (1 female) | NA (can be considered neutral) | 13100 | 13100 | 	23 hours 55 minutes 17 seconds | English | <ul><li>This is one of the largest corpuses for speech generation, with a rich vocabulary of over ~14k unique words</li><li>The sentences are taken from 7 non-fiction books</li></ul> | <ul><li>Large scale corpus</li><li> Rich vocabulary</li><li> Abbreviations in text are expanded in speech</li></ul> | <ul><li>No emotional annotations are available</li></ul> | 
| **[IEMOCAP](https://sail.usc.edu/iemocap/)** | 10 (5 female, 5 male) | 9 (anger, happiness, excitement, sadness, frustration, fear, surprise, other and neutral state) | 10039 | NA | 12.5 hours | English | <ul><li>This dataset consists of 10 actors (5 male; 5 female) of multi-modal dyadic conversations, which were scripted and enacted by a group of actors</li></ul> | <ul><li>Variety of utterances</li><li> Rich vocabulary</li><li> Multi-modal input</li><li> Easy to interpret emotional annotations</li></ul> | <ul><li>The access is very restricted and upon being granted access, we got a corrupted archive file.</li></ul> |

# Relevant literature

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

# Approach: Tacotron Models

## :x: Approach 1: Fine-tuning a Vanilla Tacotron model on RAVDESS pre-trained on LJ Speech

Our first approach was to train a vanilla Tacotron model from scratch on just one emotion (say, anger) and see if the generated voice has captured the prosodic features of that emotion.

### Motivation

- We did not have acccess to any of the datasets described above except for RAVDESS and LJ Speech and had also never tried any of the Tacotron-flavored models before.
- Hence, we just wanted to play around initially and at least generate the results on LJ Speech, and analyse the quality of speech generated.
- The fine-tuning idea seemed natural after the pre-training was done, as the RAVDESS dataset was extremely limited, there was no point on training on it from scratch, as the vocabulary that the model was exposed to would be extremely low.
- We were hoping that at best, less amount of fine-tuning would lead to transfer of prosodic features to the model and at worst, after fine-tuning for a long interval, would lead to over-fitting on the dataset.

### Observations 

- The alignment of the encoder and decoder states was completely destroyed in the first 1000 iterations of training itself.
- At test time, generated audio was initially empty. On further analysis, we discovered that this was because of the restriction on the decoder's stopping criteria.
- If the all the values in the generated frames were below a certain threshold, the decoder would stop producing the new frames. We observed that in our case, this was happening at the beginning itself.
- To fix this, we removed this condition and instead made the decoder produce sounds for a minimum number of iterations.
- We observed that for lesser iterations of finetuning (1-3k iterations) the audio was produced was complete noise, with no intelligible speech.
- If we fine-tune for long durations (~40k iterations), we observe that the model is able to generate angry speech for the utterances that are in the training set. However, even for utterances outside the training set, it speaks parts of the training set utterances only, indicating that the model has overfitted on this dataset.

### Inference and next steps

- The observations presented above seemed to present a case of "*catastrophic forgetting*" where the model was forgetting the information that it had already learnt in the pre-training rates.
- To counter this, we were advised to tweak the hyperparameters and training strategy of the model, such as learning rate, optimiser used, etc. 
- We decided to try out this following approaches:
  - **Start the fine-tuning steps with a lower learning rate**: Pre-training was done at 0.002, so we decided to do fine-tuning with 2e-5. Note that the code also implemented annealing learning rate strategy, where learning rate was reduced after few steps. We did not change it as it had given good results at pre-training.
  - **Changing the optimizer from Adam to SGD**: Because the number of samples used for fine-tuning were less, and SGD has been known to generalise better for a smaller sample size, we decided to do this.
  - **Freezing the Encoder of the Tacotron while fine-tuning**: We thought of this because the main purpose of the encoder is to convert the text to a latent space. Since LJ Speech had a better vocabulary either way, we did not feel the need to re-train this component of the model over RAVDESS' much inferior voabulary size.
  


## :x: Approach 2: Using a smaller learning rate for fine-tuning

In this approach, we repeated [Approach 1](#x-approach-1-fine-tuning-a-vanilla-tacotron-model-on-ravdess-that-was-pre-trained-on-lj-speech), but this time, while commencing the fine-tuning, we stuck with a smaller learning rate of **2e-5** as compared to the previous learning rate of **2e-3**. We did not make any other changes to the code or hyperparameters.

### Motivation

- Again, uptil this point, we had not discovered any new, larger corpus for Emotional Speech. Hence, we were stuck with RAVDESS again.
- Drawing from the previous experiment, we wanted to see if the effect of the pre-trained Tacotron model, forgetting its weights could be mitigated by reducing the effect of the new gradients that are used for update during fine-tuning. 
- To verify this, we reduced the learning rate of the model from **2e-3** to **2e-5**. 

### Observations

- We did not observe any pattern, or any improvements as such in the initial iterations of fine-tuning (after 1k, 3k and 5k iterations).
- Simply changing the initial learning rate, did not seem to have any effect on the training process.
- The alignment would still get destroyed, and there was not audio generated at test time again.

### Inference and next steps

- We decided that simply toggling the learning rate would not help with the training.
- To help with generalization, we decided to try out replacing the Adam optimiser which was being used as default, with SGD at the time of fine-tuning.



## :x: Approach 3: Using a smaller learning rate and SGD for fine-tuning

In this approach, we repeated [Approach 2](#x-approach-2-using-a-smaller-learning-rate-for-fine-tuning), but this time, while commencing the fine-tuning, we also switch the optimiser from Adam to SGD.

### Motivation

- Some studies have claimed that [SGD performs better than more sophisticated optimisers, such as Adam in later stages of training the model](https://arxiv.org/pdf/1712.07628.pdf).
- We had a smaller dataset too, and we wanted to see if SGD might perform better on this dataset. This was to see if [claims of SGD generalising better on smaller datasets](https://stats.stackexchange.com/questions/313278/no-change-in-accuracy-using-adam-optimizer-when-sgd-works-fine) on different forums, could be replicated on this problem.

### Observations

- Again, we could not observe any clear progression from the alignment plots. 
- The utterances that were seen in the test set could be produced at test time. However, the model was not able to generate any unseen utterances.

### Inference and next steps

- Since even in this case, the model was not able to generalise properly, we felt that this could be due to two reasons:
  1. The model had not even started learning anything in the fine-tuning stage
  2. Even all these measures to prevent "*catastrophic forgetting*" were not helping the model to retain information
- We decided to go with assumption 2 for now. To further aid the network in retaining its gained knowledge, we decided that we would only backpropagate the gradients through the decoder module of the Tacotron.




## :x: Approach 4: Freezing the encoder and postnet

In this approach, we repeated [Approach 3](#x-approach-3-using-a-smaller-learning-rate-and-sgd-for-fine-tuning), with the addition of sending only the decoder's parameters for optimisation.

### Motivation

- Freezing the encoder's parameters would further help the model in retaining what it learnt in the previous layers.
- We hypothesized that since the encoder learns a much richer vocabulary while being trained on LJ Speech, therefore there was no merit in re-training the encoder on the much inferior vocabulary of RAVDESS.
- Similarly, the postnet merely learns a mapping from the Mel-space to a Linear-space, which would not change for new data. Hence, it can be frozen too.

### Observations

- We got the same results as [Approach 3](#observations-2), and the observations were inconclusive.

### Inference and next steps

- Since the observations were inconclusive, we felt that probably the only way to resolve the problem was to get more emotionally annotated data.
- By this time, we also felt that it would make sense to only consider data spoken by a single speaker.
- We started looking for even non-published sources to see if there were any resources on low-resource training of Neural TTS models.



## :x: Approach 5: Freezing the encoder and postnet, and switching back to Adam

For the sake of completeness, we also repeated [Approach 4](#x-approach-4-freezing-the-encoder-and-postnet) but this time with Adam as the optimiser.

### Motivation

- This was mostly done for the sake of completeness
- We had not tried the different changes we had made with Adam, so we felt that it might make sense to go back to it.

### Observations

- The results had virtually no change from Approach 4.

### Inference and next steps

- We had discovered the preprint, "[Exploring Transfer Learning for Low Resource Emotional TTS](https://arxiv.org/abs/1901.04276)", and the for the next few attempts that we made, we focussed on trying to replicate the method given here.
- We also decided to shift our experiments form RAVDESS to EMOV-DB, as the preprint was working with this dataset.
- EMOV-DB also is larger in size as compared to RAVDESS and with a richer vocabulary, albeit the emotional labels are a bit difficult to interpret perceptively.



## :white_check_mark: Approach 6: Freezing just the post-net, using Adam with low initial learning rate, training on EMOV-DB

The experiments inspired from the preprint based on DC-TTS have been described [below](#dc-tts-models). We also thought of applying the strategy of DC-TTS to our Vanilla Tacotron strategy. Additionally, for each emotion, we used only one female speaker's data for every emotion. Details on how data was picked is given in [Approach 8](#white_check_mark-approach-8-fine-tuning-only-on-one-speaker-with-reduced-top_db-and-monotonic-attention).

### Motivation

- Using a single speaker for each emotion separately:
  - We discovered on online forums dedicated to TTS systems, such as the Mozilla discourse channel, the even SOTA Neural TTS systems, perform miserably in case of fine-tuning on multi-speaker datasets. [**\[post\]**](https://discourse.mozilla.org/t/fine-tuning-trained-model-to-new-dataset/43335/5)
- Freezing the postnet only: 
  - In the case of DC-TTS, we were retraining the entire Text2Mel module, which is responsible for mapping the input text to a Mel-spectrogram. 
  - We do not tamper with the SSRN, which learns a mel to linear mapping. 
  - Analogously, in the Tacotron, we fine-tune the Encoder + Decoder, which are responsible for mappint input text to the Mel-spectrogram.
  - We leave the Post net which learns a mapping from the Mel-spectrogram to the Linear-spectrogram.
- Using a lower learning rate:
  - We just wanted to be cautious not to erase the previous weights of the pre-trained model completely
- Using Adam as optimiser:
  - There was no specific reason to do this. Maybe SGD would have worked better here. We have not checked this!

### Observations

- We tried three emotions through this approach, **Disgust**, **Sleepiness** and **Amused** through this approach.
- All of them showed extremely improved results! We could hear intelligible speech with emotions too!
- The alignment plots were also greatly improved and could be seen to improve with increase in fine-tuning steps.

### Inference and next steps

- The idea of freezing only the post-net seemed to work wonders.
- It would be interesting to investigate the effect of changing optimisers and learning rates in this setup. It would help ascertain how much of a role the changed data played in the improved performance.

# Approach: DCTTS Models

## :x: Approach 7: Fine-tuning the Text2Mel module of the DC-TTS model on EMOV-DB pre-trained on LJ Speech

We started off with obtaining a pre-trained DC-TTS model on LJ Speech from this [PyTorch implementation of DC-TTS](https://github.com/tugstugi/pytorch-dc-tts). In this repository, a pre-trained model of DC-TTS was fine-tuned on Mongolian Speech Data, and we started of by exploring if the same process helps transfer emotional cues to the generated speech.

We also simultaneously came across [this work](https://arxiv.org/abs/1901.04276) that explores Transfer Learning methods for low-resource emotional TTS. In their approach, they decided to keep the __SSRN module frozen__ while finetuning, because SSRN does the mapping between MFBs and full spectrogram. Therefore, it should not depend on the speaker identity on speaking style as it is just trained to do the mapping between two audios. The entire __Text2Mel module was fine-tuned__ on a single emotion (Anger) of EMOV-DB.

### Motivation

- We wanted to see if the default settings used for transfer learning on the Mongolian dataset works for emotional data.
- Because so far, Tacotron was not leading to any results, trying out something that is claimed to work felt like a natural next step.

### Observations

- Pre-trained DC-TTS on LJ Speech worked fine with the ```synthesize.py``` script in the given repository.
- For the fine-tuned model, even though the audio generated was not of 0 length, it did not contain any audio. The melspectrograms (mels) and the magnitude spectrograms (mags) were also completely empty.

### Inference and next steps

- We initially suspected that because we were not fine-tuning the SSRN module, it was leading to the blank audios. However, on delving deeper, we found that it was Text2Mel which was not even generating the required output, as the mel-spectrograms generated by it were blank.
- We then explored the finer details of [this work](https://arxiv.org/abs/1901.04276).

## :white_check_mark: Approach 8: Fine-tuning only on one speaker with reduced `top_db` and monotonic attention

In this approach, we repeated the steps in [Approach 1](#x-approach-1-fine-tuning-the-text2mel-module-of-the-dc-tts-model-on-emov-db-pre-trained-on-lj-speech). In accordance with the pre-processing steps described in [the preprint](https://arxiv.org/abs/1901.04276), we made two small changes:

- For silence trimming from the input audio clips, we [set the value of the `top_db` parameter to `20` instead of the default `60`](https://github.com/Emotional-Text-to-Speech/pytorch-dc-tts/blob/aac698d46e6eb8bcab53b591126a1f781045ab17/audio.py#L37). You can learn more about the `top_db` parameter from documentation page of [`librosa.effects.trim`](https://librosa.github.io/librosa/generated/librosa.effects.trim.html).
- For the training procedure, we further set the [`monotonic_attention` parameter to `True`](https://github.com/Emotional-Text-to-Speech/pytorch-dc-tts/blob/aac698d46e6eb8bcab53b591126a1f781045ab17/train-text2mel.py#L113), as recommended in the preprint.

Additionally, we also only used the data for one female speaker per emotion. The details for the files from EMOV-DB used for each speaker are elaborated below:

| Emotion | Speaker |
| -- | -- |
| Anger | jenie | 
| Disgust | bea |
| Sleepiness | bea |
| Amused | bea |

- The data files for these speakers can be downloaded from the EMOV-DB repository (link in **Datasets** table above). We downloaded files from the sorted version of the files from the link given in the repository)

### Motivation

- The change in `top_db` was solely motivated by replicating the preprint's pipeline.
- The monotonic attention was also suggested by the preprint. And it made sense also, as monotonic attention helps induce some semblance of a temporal structure to the model.
- Lastly, using one female speaker was also motivated by the preprint. It does not mention which female speaker was used. So we used the speaker who had a higher number of utterances for each emotion. Using female speaker made sense, as LJ Speech also has a female voice, and we hoped it would reduce the amount of learning that the model had to do.
- The idea for using a single speaker on a single emotion was also discussed on online forums on TTS systems, like the Mozilla discourse channel, where in this [**\[post\]**](https://discourse.mozilla.org/t/fine-tuning-trained-model-to-new-dataset/43335/5) the users have discussed that TTS systems, even the SOTA, do not perform well on multi-speaker low-resource datasets, which we felt also justified taking this approach.

### Observations

- For the first time, we saw good-quality, generalisable speech for an emotion! **Anger** emotion was getting generated quite properly!
- However, with the other emotions, the previous problems still persisted. The generated spectrograms were still blank for all other emotions :cry:

### Inference and next steps

- **Amused** and **Sleepiness** were challenging emotions to learn in the first place. This was because of the presence of non-verbal cues like chuckling, yawning, etc, which are absent from the transcripts. The preprint said the same thing about these emotions.
- For **Disgust**, on plotting the mel-spectrograms of some ground-truth samples, we discovered that on the temporal axis, the perceptual distinction between successive temporal frames was ***lower*** as compared to **Anger**. We believe that this was the reason that the model was not able to generate **Disgust** properly. However, this is just a speculation.

# Reproducibility and Code


- For Tacotron, we worked on [our modified fork](https://github.com/Emotional-Text-to-Speech/tacotron_pytorch) of [r9y9's repository](https://github.com/r9y9/tacotron_pytorch). To reproduce our results, you can use our fork.
- For DC-TTS, we worked on [our modified fork](https://github.com/Emotional-Text-to-Speech/pytorch-dc-tts) of [tugstugi's repository](https://github.com/tugstugi/pytorch-dc-tts). Again, to reproduce our results, you can use our fork.
- Below, for each  approach, we have specified the location of the saved models, the training script to run for an approach, the dataset used, and a link to the slides we made for a detailed presentation of results. 
- For Tacotron-based approaches, the learning rate can be changed by editing the [`initial_learning_rate` parameter in `hparams.py`](https://github.com/Emotional-Text-to-Speech/tacotron_pytorch/blob/15a8352422245405ad6495f20782625e846c9945/hparams.py#L35)
- Note that for the DC-TTS approaches, we have not specified Learning Rate, Optimiser and Training script as they do not change.

| Approach | Dataset | Result Dumps | Optimiser | Learning Rate | Training Script | Slides |
| -- | -- | -- | -- | -- | -- | -- |
| [Approach 1](#x-approach-1-fine-tuning-a-vanilla-tacotron-model-on-ravdess-pre-trained-on-lj-speech) | RAVDESS (angry) | [`approach_1`](https://drive.google.com/drive/folders/16Ljdx-KVlVCwNnN7_8emntt1OjI2qfuz?usp=sharing) | Adam | 2e-3 | [`train.py`](https://github.com/Emotional-Text-to-Speech/tacotron_pytorch/blob/master/train.py) | [\[slides\]](https://docs.google.com/presentation/d/1CuOiThrBodv6HRp5gCdFN8dnUwkVpFYs7DRP1DS2nKk/edit?usp=sharing) |
| [Approach 2](#x-approach-2-using-a-smaller-learning-rate-for-fine-tuning) | RAVDESS (angry) | [`approach_2`](https://drive.google.com/open?id=1VJb5_GjZGWdGvTnRWG2fqXpFht9sMnuC) | Adam | 2e-5 | [`train.py`](https://github.com/Emotional-Text-to-Speech/tacotron_pytorch/blob/master/train.py) | [\[slides\]](https://docs.google.com/presentation/d/1pZSFGBPgVUbMk3gFBctaJTEg7VvYQGp1sxEKh0sQ7OA/edit?usp=sharing) |
| [Approach 3](#x-approach-3-using-a-smaller-learning-rate-and-sgd-for-fine-tuning) | RAVDESS (angry) | [`approach_3`](https://drive.google.com/drive/folders/19r3BXQKjfLWxhJHjv9OfS_U8djAP8VnV?usp=sharing) | SGD | 2e-5 | [`train_sgd.py`](https://github.com/Emotional-Text-to-Speech/tacotron_pytorch/blob/master/train_sgd.py) | [\[slides\]](https://docs.google.com/presentation/d/1Q8WbV8xmzZNUv8k62t2FgyN2ZxS5AWQI7y4mpuwomH0/edit?usp=sharing) |
| [Approach 4](#x-approach-4-freezing-the-encoder-and-postnet) | RAVDESS (angry) | [`approach_4`](https://drive.google.com/open?id=11SxFVEtQDSBIlz549oXs9M27AcztjfJE) | SGD | 2e-5 | [`train_fr_enc_sgd.py`](https://github.com/Emotional-Text-to-Speech/tacotron_pytorch/blob/master/train_fr_enc_sgd.py) | [\[slides\]](https://docs.google.com/presentation/d/1d8VMrd8vAVRE7PcDd3vKdCgFBVf27qDNqPbWMU1QKj4/edit?usp=sharing) |
| [Approach 5](#x-approach-5-freezing-the-encoder-and-postnet-and-switching-back-to-adam) | RAVDESS (angry) | [`approach_5`](https://drive.google.com/open?id=1-2bCkUftdJidkIB5uuD2yYOGm-g9_B8S) | Adam | 2e-5 | [`train_fr_enc_adam.py`](https://github.com/Emotional-Text-to-Speech/tacotron_pytorch/blob/master/train_fr_enc_adam.py) | [\[slides\]](https://docs.google.com/presentation/d/1Y99-84_6S35PRPEKQ1oYmD3kghqDU1D1rUECHwKogNE/edit?usp=sharing) |
| [Approach 6](#white_check_mark-approach-6-freezing-just-the-post-net-using-adam-with-low-initial-learning-rate-training-on-emov-db) | EMOV-DB (each emotion, one speaker) | [`approach_6`](https://drive.google.com/open?id=18VZBbNImoZmN2NrbZgziVoDWF5nXXLqY) | Adam | 2e-5 | [`train_fr_postnet_adam.py `](https://github.com/Emotional-Text-to-Speech/tacotron_pytorch/blob/master/train_fr_postnet_adam.py) | [\[slides\]](https://docs.google.com/presentation/d/1Y99-84_6S35PRPEKQ1oYmD3kghqDU1D1rUECHwKogNE/edit?usp=sharing) |
| [Approach 7](#x-approach-7-fine-tuning-the-text2mel-module-of-the-dc-tts-model-on-emov-db-pre-trained-on-lj-speech) | EMOV-DB (angry) | [`approach_7`](https://drive.google.com/open?id=1HaVlGpiVFKG40vLFlorvtFzkFAHqZSjj) | \- | \- | \- | [\[slides\]](https://docs.google.com/presentation/d/1d8VMrd8vAVRE7PcDd3vKdCgFBVf27qDNqPbWMU1QKj4/edit?usp=sharing) |
| [Approach 8](#white_check_mark-approach-8-fine-tuning-only-on-one-speaker-with-reduced-top_db-and-monotonic-attention) | EMOV-DB (each emotion, one speaker) |  [`approach_8`](https://drive.google.com/open?id=1UIbVj-KjI1YJh6dDZTTJNUYHo4jAKZ1h) | \- | \- | \- | [\[slides\]](https://docs.google.com/presentation/d/1Y99-84_6S35PRPEKQ1oYmD3kghqDU1D1rUECHwKogNE/edit?usp=sharing) |

- The pre-trained model for Tacotron, trained on LJ Speech is available here: [`pretrained_ljspeech_tacotron`](https://drive.google.com/open?id=1fgh_1asVi5fsFo_PMyVGQfNOn7kD1xyE)
- The pre-trained model for DC-TTS, trained on LJ Speech is available here: [`pretrained_ljspeech_dctts`](https://drive.google.com/drive/folders/10nz8_0O4g5vc1K0pEoiP4QTeBzA2s-sl?usp=sharing)

# Demonstration

In order to view a working demonstration of the models, open the file ```Demo_DL_Based_Emotional_TTS.ipynb``` and click on ```Open in Colab```. Follow the steps as mentioned in the Colab Notebook.

Models used in our code are here: [`demo_models`](https://drive.google.com/open?id=1n9RYwClrcWz7jbrTM4gzypCMvW7HQHu4)

# Cite

If you find the models, code or approaches in this repository helpful, please consider citing this repository as follows:

```
@software{aditya_chetan_2020_3876081,
  author       = {Aditya Chetan and
                  Brihi Joshi and
                  Pulkit Madaan and
                  Pranav Jain and
                  Srija Anand and
                  Eshita and
                  Shruti Singh},
  title        = {{An exploration into Deep Learning methods for 
                   Emotional Text-to-Speech}},
  month        = jun,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.3876081},
  url          = {https://doi.org/10.5281/zenodo.3876081}
}
```

---

# Contact

For any errors or help in running the project, please open an issue or write to any of the project members - 

- Pulkit Madaan (pulkit16257 [at] iiitd [dot] ac [dot] in)
- Aditya Chetan (aditya16217 [at] iiitd [dot] ac [dot] in)
- Brihi Joshi (brihi16142 [at] iiitd [dot] ac [dot] in)
- Pranav Jain (pranav16255 [at] iiitd [dot] ac [dot] in)
- Srija Anand (srija17199 [at] iiitd [dot] ac [dot] in)
- Eshita (eshita17149 [at] iiitd [dot] ac [dot] in)
- Shruti Singh (shruti17211 [at] iiitd [dot] ac [dot] in)





