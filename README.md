# Video-Feedback-System

This project's main goal is to develop a feedback system where viewer can instantly get an answer while watching a video from a course video series using machine learning methods. Let's say student watches a video and writes a feedback about the video to the system about their understandings. Using sentiment analysis, system labels the feedback as positive(understood) or negative(not understood). If the label is negative, cosine similairty is calculated between student feedback and video transcriptions. Afterwards, according to cosine similarity calculations a section of the video is returned to the student implying that they should rewatch this part with more attention.

Demo video without any explanation can be found [here](https://drive.google.com/file/d/1rNt5LDIPw395cAenD47yuzZHjBFJYbVY/view?usp=sharing)

## Dataset
In this project, MEF University's EE 204 Signals and System Course's FLipped Learning videos and their transcripts are used as dataset. Due to confidentiality, I can only share percentile results.

## Data processing
Since the data from EE 204 course was limited, IMDb's Large Movie Review Dataset is used to pre-train the BERT Language Model (IMDb_training.ipynb). Afterwards, EE 204 Course data used to train the pre-trained model(Student_data_training.ipynb).

## Keyword Extraction

To match the feedback with the video transcriptions, I've used cosine similarity method and calculated each transcription sentence's cosine similarity with input feedback. Then return the 6 sentences with the highest average cosine similarity. From my observations, matched texts with higher than 0.6 cosine similarity scores are meaningful to the input feedback. 

## Results


After training the BERT Model on IMDb's Large Movie Review Dataset, validation accuracy is found as 0.916 and test accuracy is found as 0.9.

After training Student's Feedback Data on pre-trained model, validation accuracy is found as 0.95 and test accuracy is found as 0.80. Reason behind the huge difference between validation and test accuracy is that, Student's Feedback Data is limited amount and majority of the feedbacks are, similiar to "no",  too short.


For the Keyword Extraction Method, only 20% of the matched video sections had higher than 0.6 cosine similarity, which is due to limited data and majority of them being too short.



