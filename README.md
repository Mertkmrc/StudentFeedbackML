# Video-Feedback-System

This project's main goal is to develop a feedback system where viewer can instantly get an answer while watching a video from a course video series using machine learning methods. For example, after student watched the video gives a feedback about the video, if they understood the content or not. Using sentiment analysis, system labels the feedback as positive(understood) or negative(not understood). If the label is negative, cosine similairty is calculated between student feedback and video transcriptions. Afterwards, according to cosine similarity calculations a section of the video is returned to the student implying that they should rewatch this part with more attention.

## Dataset
In this project, MEF University's EE 204 Signals and System Course's FLipped Learning videos and their transcripts are used as dataset. Due to confidentiality, I can only share percentile results.

## Data processing
Since the data from EE 204 course was limited, IMDb's Large Movie Review Dataset is used to pre-train the model on BERT Language Model (IMDb_training.ipynb). Afterwards, EE 204 Course data used to train the pre-trained model(Student_data_training.ipynb).






