# Utrecht-University-Internship:
The internship project I did with Jessica Heeman at the Utrecht University was titled 'Oculomotor control and eye movements: The effect of reward on express saccades'. During this project, I used Python and Psychopy to create an eye-tracking experiment about the influence of rewards on express saccades.

## What Are Express Saccades?:

Express saccades are very short-latency saccades and have latencies around 80 to 130 ms after stimulus onset. These saccades are visually guided and considered to be the result of advanced preparation of the oculomotor program toward the target. There is a known effect in visual research called the Gap effect, this effect occurs when there is a temporal gap between fixation offset and target onset, and this effect reduces the activity of neurons in Superior Colliculus, releasing the visual attention and allowing the saccadic response system to respond more quickly to the stimuli. This effect reduces saccadic reaction time and increases the percentage of express saccades generated. 

## Experiment Structure:

![image](https://user-images.githubusercontent.com/96383593/170816190-d438b718-c67b-466d-b217-03dd937d7f4f.png)


## Preprocessing and Exclusion Criterion:

We need to define an exclusion criterion for filtering data, before doing any statistical analysis, to get relevant information from the eye-tracking data. Firstly, saccades needed to be initiated after the target appeared, saccade started before the target appearance were excluded because we specifically inform participants to make an eye movement after the target appears and we do not want to include anticipatory saccades. Anticipatory saccades are not visually guided saccades and have a latency of around 80ms. Saccades had to have a minimum amplitude of a 2-degree visual angle and had to start within 2 degrees around the fixation point and 4 degrees around the target, it is common that people undershoot while generating express saccades because of that we also want to include those saccades. Saccades must have a minimum of 2 degrees of amplitude. Amplitude is defined as the distance between the starting point and endpoint of a saccade in visual degrees. Saccades that had a duration of over 75ms were excluded and Saccades that had latencies over 500ms were excluded because these saccades are too slow to be interest in this study. Finally, each trial for each participant was visually inspected, filtered, and excluded if they contained any technical errors or blinks.

Example of Valid Eye Movement:

![image](https://user-images.githubusercontent.com/96383593/170816577-87438910-f1c8-47b5-a0ee-6f1b03514660.png)

Example of Invalid Eye Movement:

![image](https://user-images.githubusercontent.com/96383593/170816590-87aee7e2-d931-4eaa-a57d-f901a8fc6f26.png)

## Results:

There are three conditions in the experiment, the first condition is no reward condition which is the first block of the experiment, the second condition is a low reward condition in which the target appeared on the low reward value side of the screen in the second block and the third condition is high reward condition which target appeared in high reward value side of the screen in the second block. Median Saccadic Reaction Time for all participants in each condition was calculated and used a three-level repeated-measures analysis of variance(ANOVA) with the reward a value as a factor, this analysis shows there is no statistical difference between three conditions with a p-value of 0.945. There is no difference between latency of reaction time of no reward, low reward, and high reward condition. This table shows the Saccadic Reaction Time distribution for all three conditions.

![image](https://user-images.githubusercontent.com/96383593/170818017-8d98eac7-3528-4cfa-834e-1d27f2bc5215.png)

![image](https://user-images.githubusercontent.com/96383593/170818023-f3ada238-86ab-4ea3-ae58-00d94b6458c4.png)

We also applied a three-level repeated-measures analysis of variance(ANOVA) with the reward value as a factor for amplitude. Interestingly there were statistical differences between the three conditions. The post Hoc test showed a statistically significant difference between the No Reward-Low Reward condition and the No Reward-High Reward condition, but the difference is not significant in the Low Reward-High Reward condition.

![image](https://user-images.githubusercontent.com/96383593/170818037-49026cf8-0de5-46e4-b515-b632fabcfa74.png)

![image](https://user-images.githubusercontent.com/96383593/170818040-90cdf0b2-96d5-414c-9c32-72168e669178.png)

![image](https://user-images.githubusercontent.com/96383593/170818044-0079b29b-eaee-4c9d-bf6a-5201ccc75f1f.png)










