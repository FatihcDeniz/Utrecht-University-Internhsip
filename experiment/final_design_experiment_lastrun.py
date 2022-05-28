#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on Sat May 28 09:40:32 2022
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, iohub, hardware
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

"""
This is python document of Psychopy that involves all the
code written by me and psychopy. For running this experiment
make sure that you read Readme document. If any of the conditions
described in that document is not met Psychopy will raise an error.
If everything in the Readme document is setted up you can just click
green run arrow in Psychopy and it will run the experiment for you.

The experiment involves two different blocks, first block is for baseline
condition which we don't give any reward to the participants, in second block
participant receive reward based on their performance. 

In each block experimental procedure is the same, only whether participant will
receive reward or not or which side of the screen will give reward changes. Each
run starts with fixation cross appearing on the screen and it is followed by a cue arrow
that is showing either right or left side of the screen. Before cue target disappears a "beep"
sound played to indicate target will appear. After cue disappeared there is 150ms-200ms
gap and after this gap target appears either right or left side of the screen. After the target 
there is a feedback screen, in baseline condition this feedback screen will show whether participant
made and eye movement to the target or not, if it is on target they receive "right" if it is 
not they will receive "false" as a feedback. In reward condition feedback either is "100 points"
or "10 points" based on condition. There is two different condition in reward block of the experiment,
first condition is that right side gives high reward %80 percent of the time and gives 
low reward %20 percent of them and left side gives low reward %80 percent of the time and gives 
high reward %20 percent of them, second condition is that left side gives high reward %80 percent of 
the time and gives low reward %20 percent of them right side gives low reward %80 percent 
of the time and gives high reward %20 percent of them. At the end of each reward block
participant will see a scoreboard that show their result and fake results of other people,
idea of scoreboard is to encourage people to be more competetive.

Calibration routine in PSychopy will run eye-track calibration and validation, you can change
appearance of targets for calibration and validation, number of points, progress time or animation.

Welcome Screen is the first screen of the experiment gives relevant information about the experiment
and participant can start experiment by starting "space button".

Train is the baseline block.


"""


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'final_design_experiment'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/fatihdeniz/Desktop/Fatih Celalettin Deniz - Internship/Psychopy/final_design_experiment_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1440, 900], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='test', color=[-0.8,-0.8,-0.8], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='pix')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = 'eyetracker.hw.sr_research.eyelink.EyeTracker'
ioConfig = {
    ioDevice: {
        'name': 'tracker',
        'model_name': 'EYELINK 1000 DESKTOP',
        'simulation_mode': False,
        'network_settings': '100.1.1.1',
        'default_native_data_file_name': 'EXPFILE',
        'runtime_settings': {
            'sampling_rate': 1000.0,
            'track_eyes': 'LEFT_EYE',
            'sample_filtering': {
                'sample_filtering': 'FILTER_LEVEL_2',
                'elLiveFiltering': 'FILTER_LEVEL_OFF',
            },
            'vog_settings': {
                'pupil_measure_types': 'PUPIL_DIAMETER',
                'tracking_mode': 'PUPIL_CR_TRACKING',
                'pupil_center_algorithm': 'ELLIPSE_FIT',
            }
        }
    }
}
ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, experiment_code='final_design_experiment', session_code=ioSession, datastore_name=filename, **ioConfig)
eyetracker = ioServer.getDevice('tracker')

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "welcome_screen"
welcome_screenClock = core.Clock()
welcome_text = visual.TextStim(win=win, name='welcome_text',
    text='Welcome ,\n\nMake an eyement AS FAST AS possible to the target appearing right or left  side of the screen.\nIf eye movements on the target you will receive "Correct" as a feedback, if it is not feedback will be  "False".\n\nPress "Space" key to start the experiment.\n ',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=1000.0, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
welcome_key = keyboard.Keyboard()
#ioConfig[ioDevice]['monitor_event_types'] = ['SaccadeStartEvent','SaccadeEndEvent']
#eyetracker.runSetupProcedure()
#Import Required Libraries
# 1 means r_h, 0 means r_l
import random
import time
#Setting mouse to invisible
win.mouseVisible = False
#Target Location
target_loc_right = 600
target_loc_left = -600
#Feedback and cue text
high_reward_text = '100 points'
low_reward_text = '10 points'
no_reward_text = '0 points'
space = 'press space key to continue'
cue_right = '->'
cue_left = '<-'
correct_text = "correct"
false_text = "false"
#Size of target,cue,feedback and invisible target
target_size = 25
invisible_target_size = 125
feedback_text_size = 60
fixation_size = 30
cue_size = 30
helper_size = 0
#Points and texts for scoreboard
count = 0
score_1 = 19980
score_2 = 14840
score_3 = 9350
score_4 = 4550
score_1_text = f'JH: {score_1} points.'
score_2_text = f'FD: {score_2} points.'
score_3_text = f'CD: {score_3} points.'
score_4_text = f'KD:  {score_4} points.'


# Initialize components for Routine "train"
trainClock = core.Clock()
fixation_training = visual.ShapeStim(
    win=win, name='fixation_training', vertices='cross',
    size=fixation_size,
    ori=0.0, pos=(0, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
cue_training = visual.TextStim(win=win, name='cue_training',
    text='',
    font='Open Sans',
    pos=(0, 0), height=cue_size, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
sound_1 = sound.Sound('right.wav', secs=0.1, stereo=True, hamming=True,
    name='sound_1')
sound_1.setVolume(1.0)
gap_training = visual.TextStim(win=win, name='gap_training',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
circle_training_2 = visual.ShapeStim(
    win=win, name='circle_training_2',
    size=invisible_target_size, vertices='circle',
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[-0.8,-0.8,-0.8], fillColor=[-0.8,-0.8,-0.8],
    opacity=None, depth=-4.0, interpolate=True)
circle_training = visual.ShapeStim(
    win=win, name='circle_training',
    size=target_size, vertices='circle',
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)
feedback_training = visual.TextStim(win=win, name='feedback_training',
    text='',
    font='Open Sans',
    pos=(0, 0), height=feedback_text_size, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-6.0);
etRecord = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)
polygon_2 = visual.ShapeStim(
    win=win, name='polygon_2',
    size=helper_size, vertices='circle',
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='blue', fillColor='blue',
    opacity=None, depth=-9.0, interpolate=True)

# Initialize components for Routine "test_start"
test_startClock = core.Clock()
text_7 = visual.TextStim(win=win, name='text_7',
    text='Welcome,\n\nMake an eyement AS FAST AS possible to the target appearing right or left  side of the screen.\n\nTarget appearing on the right side of the screen has more chance to to gove high reward and target appearing on the left side of the screen has more chance to give a low reward. If eye movements are on the target, you will receive a reward. If is not on the target you will not receive a reward.\n\nPress "Space" key to start experiment.\n',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=1000.0, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_3 = keyboard.Keyboard()
eyetracker.sendMessage('welcome_test_1')


# Initialize components for Routine "test_block_1"
test_block_1Clock = core.Clock()
fixation_test_1 = visual.ShapeStim(
    win=win, name='fixation_test_1', vertices='cross',
    size=fixation_size,
    ori=0.0, pos=(0, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
cue_block1 = visual.TextStim(win=win, name='cue_block1',
    text='',
    font='Open Sans',
    pos=(0, 0), height=cue_size, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
sound_2 = sound.Sound('right.wav', secs=0.1, stereo=True, hamming=True,
    name='sound_2')
sound_2.setVolume(1.0)
gap_test_1 = visual.TextStim(win=win, name='gap_test_1',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
circle_test_2 = visual.ShapeStim(
    win=win, name='circle_test_2',
    size=invisible_target_size, vertices='circle',
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[-0.8,-0.8,-0.8], fillColor=[-0.8,-0.8,-0.8],
    opacity=None, depth=-4.0, interpolate=True)
circle_test = visual.ShapeStim(
    win=win, name='circle_test',
    size=target_size, vertices='circle',
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)
feedback_test_1 = visual.TextStim(win=win, name='feedback_test_1',
    text='',
    font='Open Sans',
    pos=(0, 0), height=feedback_text_size, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-6.0);
eyetracker_test = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)
polygon_3 = visual.ShapeStim(
    win=win, name='polygon_3',
    size=helper_size, vertices='circle',
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='blue', fillColor='blue',
    opacity=None, depth=-9.0, interpolate=True)

# Initialize components for Routine "Amount_1"
Amount_1Clock = core.Clock()
text_6 = visual.TextStim(win=win, name='text_6',
    text='',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=1000.0, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_5 = keyboard.Keyboard()
text_10 = visual.TextStim(win=win, name='text_10',
    text='Scoreboard',
    font='Open Sans',
    pos=(0, 350), height=50.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
text_12 = visual.TextStim(win=win, name='text_12',
    text='',
    font='Open Sans',
    pos=(0, -350), height=30.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "Break"
BreakClock = core.Clock()
text_4 = visual.TextStim(win=win, name='text_4',
    text='Welcome,\nMake an eyement AS FAST AS possible to the target appearing right or left  side of the screen.\n\nTarget appearing on the left side of the screen has more chance to to give high reward and target appearing on the right side of the screen has more chance to give a low reward. If eye movements are on the target, you will receive a reward. If is not on the target you will not receive a reward.\nPress "Space" key to start experiment.\n',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=1000.0, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_2 = keyboard.Keyboard()

# Initialize components for Routine "test_block_2"
test_block_2Clock = core.Clock()
fixation_test_block_2 = visual.ShapeStim(
    win=win, name='fixation_test_block_2', vertices='cross',
    size=fixation_size,
    ori=0.0, pos=(0, 0),
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
gap_test_block_2 = visual.TextStim(win=win, name='gap_test_block_2',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
cue_block_2 = visual.TextStim(win=win, name='cue_block_2',
    text='',
    font='Open Sans',
    pos=(0, 0), height=cue_size, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
sound_3 = sound.Sound('right.wav', secs=0.1, stereo=True, hamming=True,
    name='sound_3')
sound_3.setVolume(1.0)
circle_test_block_3 = visual.ShapeStim(
    win=win, name='circle_test_block_3',
    size=invisible_target_size, vertices='circle',
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[-0.8,-0.8,-0.8], fillColor=[-0.8,-0.8,-0.8],
    opacity=None, depth=-4.0, interpolate=True)
circle_test_block_2 = visual.ShapeStim(
    win=win, name='circle_test_block_2',
    size=target_size, vertices='circle',
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)
test_feedback_block_2 = visual.TextStim(win=win, name='test_feedback_block_2',
    text='',
    font='Open Sans',
    pos=(0, 0), height=feedback_text_size, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-6.0);
eyetracker_test_block_2 = hardware.eyetracker.EyetrackerControl(
    server=ioServer,
    tracker=eyetracker
)
polygon_4 = visual.ShapeStim(
    win=win, name='polygon_4',
    size=helper_size, vertices='circle',
    ori=0.0, pos=[0,0],
    lineWidth=1.0,     colorSpace='rgb',  lineColor='blue', fillColor='blue',
    opacity=None, depth=-9.0, interpolate=True)

# Initialize components for Routine "Amount_2"
Amount_2Clock = core.Clock()
text_8 = visual.TextStim(win=win, name='text_8',
    text='',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=1000.0, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_6 = keyboard.Keyboard()
text_11 = visual.TextStim(win=win, name='text_11',
    text='Scoreboard',
    font='Open Sans',
    pos=(0, 350), height=50.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);
text_13 = visual.TextStim(win=win, name='text_13',
    text='',
    font='Open Sans',
    pos=(0, -350), height=30.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "end_screen"
end_screenClock = core.Clock()
text_5 = visual.TextStim(win=win, name='text_5',
    text='Thank you for participating.\n\nYou can leave.',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_4 = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# -------Run Routine 'calibration'-------

# define target for calibration
calibrationTarget = visual.TargetStim(win, 
    name='calibrationTarget',
    radius=25.0, fillColor='white', borderColor='white', lineWidth=1.0,
    innerRadius=25.0, innerFillColor='white', innerBorderColor='white', innerLineWidth=1.0,
    colorSpace='rgb', units=None
)
# define parameters for calibration
calibration = hardware.eyetracker.EyetrackerCalibration(win, 
    eyetracker, calibrationTarget,
    units=None, colorSpace='rgb',
    progressMode='space key', targetDur=1.0, expandScale=1.5,
    targetLayout='NINE_POINTS', randomisePos=True,
    movementAnimation=False, targetDelay=1.0
)
# run calibration
calibration.run()
# clear any keypresses from during calibration so they don't interfere with the experiment
defaultKeyboard.clearEvents()
# the Routine "calibration" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "welcome_screen"-------
continueRoutine = True
# update component parameters for each repeat
welcome_key.keys = []
welcome_key.rt = []
_welcome_key_allKeys = []
#Sending message
eyetracker.sendMessage('Welcome')

# keep track of which components have finished
welcome_screenComponents = [welcome_text, welcome_key]
for thisComponent in welcome_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
welcome_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "welcome_screen"-------
while continueRoutine:
    # get current time
    t = welcome_screenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=welcome_screenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *welcome_text* updates
    if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        welcome_text.frameNStart = frameN  # exact frame index
        welcome_text.tStart = t  # local t and not account for scr refresh
        welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
        welcome_text.setAutoDraw(True)
    
    # *welcome_key* updates
    waitOnFlip = False
    if welcome_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        welcome_key.frameNStart = frameN  # exact frame index
        welcome_key.tStart = t  # local t and not account for scr refresh
        welcome_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(welcome_key, 'tStartRefresh')  # time at next scr refresh
        welcome_key.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(welcome_key.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(welcome_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if welcome_key.status == STARTED and not waitOnFlip:
        theseKeys = welcome_key.getKeys(keyList=['space'], waitRelease=False)
        _welcome_key_allKeys.extend(theseKeys)
        if len(_welcome_key_allKeys):
            welcome_key.keys = _welcome_key_allKeys[-1].name  # just the last key pressed
            welcome_key.rt = _welcome_key_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in welcome_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "welcome_screen"-------
for thisComponent in welcome_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('welcome_text.started', welcome_text.tStartRefresh)
thisExp.addData('welcome_text.stopped', welcome_text.tStopRefresh)
# check responses
if welcome_key.keys in ['', [], None]:  # No response was made
    welcome_key.keys = None
thisExp.addData('welcome_key.keys',welcome_key.keys)
if welcome_key.keys != None:  # we had a response
    thisExp.addData('welcome_key.rt', welcome_key.rt)
thisExp.addData('welcome_key.started', welcome_key.tStartRefresh)
thisExp.addData('welcome_key.stopped', welcome_key.tStopRefresh)
thisExp.nextEntry()
# the Routine "welcome_screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
train_loop = data.TrialHandler(nReps=3.0, method='fullRandom', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('location_reward.xlsx', selection='0:2'),
    seed=None, name='train_loop')
thisExp.addLoop(train_loop)  # add the loop to the experiment
thisTrain_loop = train_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrain_loop.rgb)
if thisTrain_loop != None:
    for paramName in thisTrain_loop:
        exec('{} = thisTrain_loop[paramName]'.format(paramName))

for thisTrain_loop in train_loop:
    currentLoop = train_loop
    # abbreviate parameter names if possible (e.g. rgb = thisTrain_loop.rgb)
    if thisTrain_loop != None:
        for paramName in thisTrain_loop:
            exec('{} = thisTrain_loop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "train"-------
    continueRoutine = True
    # update component parameters for each repeat
    sound_1.setSound('right.wav', secs=0.1, hamming=True)
    sound_1.setVolume(1.0, log=False)
    circle_training_2.setPos(location)
    circle_training.setPos(location)
    #Sending Message to eye tracker
    eyetracker.sendMessage('train')
    #Fixation Duration is between 1.0 to 0.8 seconds.
    number_t = random.uniform(0,0.2)
    fixation_training_t = 1.0 - number_t
    #Cue Duration is between 0.4 to 0.7 seconds.
    cue_t = random.uniform(0.4,0.7)
    cue_training_t = cue_t + fixation_training_t
    #Sounc Cue appear last 0.1 seconds of Cue
    sound_t = cue_training_t - 0.1
    #Gap Duration is 0.2 to 0.25 seconds.
    number_2_t = random.uniform(0.2,0.25)
    gap_training_t=  number_2_t + cue_training_t
    #Target Duration is 1 second.
    poly_training_t = 1+ gap_training_t
    #Feedback Duration is 1 second. 
    feedback_training_t = poly_training_t + 1.0
    
    
    
    startt = time.time()
    switch = 0
    # keep track of which components have finished
    trainComponents = [fixation_training, cue_training, sound_1, gap_training, circle_training_2, circle_training, feedback_training, etRecord, polygon_2]
    for thisComponent in trainComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trainClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "train"-------
    while continueRoutine:
        # get current time
        t = trainClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trainClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_training* updates
        if fixation_training.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_training.frameNStart = frameN  # exact frame index
            fixation_training.tStart = t  # local t and not account for scr refresh
            fixation_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_training, 'tStartRefresh')  # time at next scr refresh
            fixation_training.setAutoDraw(True)
        if fixation_training.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > fixation_training_t-frameTolerance:
                # keep track of stop time/frame for later
                fixation_training.tStop = t  # not accounting for scr refresh
                fixation_training.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fixation_training, 'tStopRefresh')  # time at next scr refresh
                fixation_training.setAutoDraw(False)
        
        # *cue_training* updates
        if cue_training.status == NOT_STARTED and tThisFlip >= fixation_training_t-frameTolerance:
            # keep track of start time/frame for later
            cue_training.frameNStart = frameN  # exact frame index
            cue_training.tStart = t  # local t and not account for scr refresh
            cue_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cue_training, 'tStartRefresh')  # time at next scr refresh
            cue_training.setAutoDraw(True)
        if cue_training.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > cue_training_t-frameTolerance:
                # keep track of stop time/frame for later
                cue_training.tStop = t  # not accounting for scr refresh
                cue_training.frameNStop = frameN  # exact frame index
                win.timeOnFlip(cue_training, 'tStopRefresh')  # time at next scr refresh
                cue_training.setAutoDraw(False)
        if cue_training.status == STARTED:  # only update if drawing
            cue_training.setText(cue_text, log=False)
        # start/stop sound_1
        if sound_1.status == NOT_STARTED and tThisFlip >= sound_t-frameTolerance:
            # keep track of start time/frame for later
            sound_1.frameNStart = frameN  # exact frame index
            sound_1.tStart = t  # local t and not account for scr refresh
            sound_1.tStartRefresh = tThisFlipGlobal  # on global time
            sound_1.play(when=win)  # sync with win flip
        if sound_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sound_1.tStartRefresh + 0.1-frameTolerance:
                # keep track of stop time/frame for later
                sound_1.tStop = t  # not accounting for scr refresh
                sound_1.frameNStop = frameN  # exact frame index
                win.timeOnFlip(sound_1, 'tStopRefresh')  # time at next scr refresh
                sound_1.stop()
        
        # *gap_training* updates
        if gap_training.status == NOT_STARTED and tThisFlip >= cue_training_t-frameTolerance:
            # keep track of start time/frame for later
            gap_training.frameNStart = frameN  # exact frame index
            gap_training.tStart = t  # local t and not account for scr refresh
            gap_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(gap_training, 'tStartRefresh')  # time at next scr refresh
            gap_training.setAutoDraw(True)
        if gap_training.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > gap_training_t-frameTolerance:
                # keep track of stop time/frame for later
                gap_training.tStop = t  # not accounting for scr refresh
                gap_training.frameNStop = frameN  # exact frame index
                win.timeOnFlip(gap_training, 'tStopRefresh')  # time at next scr refresh
                gap_training.setAutoDraw(False)
        
        # *circle_training_2* updates
        if circle_training_2.status == NOT_STARTED and tThisFlip >= gap_training_t-frameTolerance:
            # keep track of start time/frame for later
            circle_training_2.frameNStart = frameN  # exact frame index
            circle_training_2.tStart = t  # local t and not account for scr refresh
            circle_training_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_training_2, 'tStartRefresh')  # time at next scr refresh
            circle_training_2.setAutoDraw(True)
        if circle_training_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > poly_training_t-frameTolerance:
                # keep track of stop time/frame for later
                circle_training_2.tStop = t  # not accounting for scr refresh
                circle_training_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(circle_training_2, 'tStopRefresh')  # time at next scr refresh
                circle_training_2.setAutoDraw(False)
        
        # *circle_training* updates
        if circle_training.status == NOT_STARTED and tThisFlip >= gap_training_t-frameTolerance:
            # keep track of start time/frame for later
            circle_training.frameNStart = frameN  # exact frame index
            circle_training.tStart = t  # local t and not account for scr refresh
            circle_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_training, 'tStartRefresh')  # time at next scr refresh
            circle_training.setAutoDraw(True)
        if circle_training.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > poly_training_t-frameTolerance:
                # keep track of stop time/frame for later
                circle_training.tStop = t  # not accounting for scr refresh
                circle_training.frameNStop = frameN  # exact frame index
                win.timeOnFlip(circle_training, 'tStopRefresh')  # time at next scr refresh
                circle_training.setAutoDraw(False)
        
        # *feedback_training* updates
        if feedback_training.status == NOT_STARTED and tThisFlip >= poly_training_t-frameTolerance:
            # keep track of start time/frame for later
            feedback_training.frameNStart = frameN  # exact frame index
            feedback_training.tStart = t  # local t and not account for scr refresh
            feedback_training.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(feedback_training, 'tStartRefresh')  # time at next scr refresh
            feedback_training.setAutoDraw(True)
        if feedback_training.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > feedback_training_t-frameTolerance:
                # keep track of stop time/frame for later
                feedback_training.tStop = t  # not accounting for scr refresh
                feedback_training.frameNStop = frameN  # exact frame index
                win.timeOnFlip(feedback_training, 'tStopRefresh')  # time at next scr refresh
                feedback_training.setAutoDraw(False)
        if feedback_training.status == STARTED:  # only update if drawing
            feedback_training.setText(text_feedback_training, log=False)
        try:
            if circle_training.pos[0] == target_loc_right:
                cue_text = cue_right
            elif circle_training.pos[0] == target_loc_left:
                cue_text = cue_left
            else:
                cue_text = ' '
            position_training = eyetracker.getLastGazePosition()
            if circle_training.pos[0] == target_loc_right:
                if circle_training_2.contains(position_training):
                    text_feedback_training = correct_text
                else:
                    text_feedback_training = false_text
            if circle_training.pos[0] == target_loc_left:
                if circle_training_2.contains(position_training):
                    text_feedback_training = correct_text
                else:
                    text_feedback_training = false_text
        except:
            text_feedback_training =' '
            position_training = (999,999)
        
        #Sending messages to eyetracker
        if time.time()-startt > fixation_training_t and switch ==0:
            eyetracker.sendMessage(f'fixation_offset: {fixation_training_t}')
            print(f'fixation {fixation_training_t},{time.time()-startt}')
            switch= switch + 1
        if time.time()-startt>cue_training_t and switch ==1:
            eyetracker.sendMessage(f'cue_offset: {cue_t} {cue_training_t}')
            print(f'cue {cue_training_t},{time.time()-startt}')
            switch = switch+1
        if time.time()-startt>gap_training_t and switch ==2:
            eyetracker.sendMessage(f'gap_offset: {number_2_t}  {gap_training_t}')
            print(f' gap {gap_training_t},{time.time()-startt}')
            switch = switch + 1
        if time.time()-startt>poly_training_t and switch ==3:
            eyetracker.sendMessage(f'poly_offset:{poly_training_t} {text_feedback_training}')
            print(f'poly {poly_training_t},{time.time()-startt} {text_feedback_training}')
            switch = switch +1
        # *etRecord* updates
        if etRecord.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            etRecord.frameNStart = frameN  # exact frame index
            etRecord.tStart = t  # local t and not account for scr refresh
            etRecord.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(etRecord, 'tStartRefresh')  # time at next scr refresh
            etRecord.status = STARTED
        if etRecord.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > poly_training_t-frameTolerance:
                # keep track of stop time/frame for later
                etRecord.tStop = t  # not accounting for scr refresh
                etRecord.frameNStop = frameN  # exact frame index
                win.timeOnFlip(etRecord, 'tStopRefresh')  # time at next scr refresh
                etRecord.status = FINISHED
        
        # *polygon_2* updates
        if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon_2.frameNStart = frameN  # exact frame index
            polygon_2.tStart = t  # local t and not account for scr refresh
            polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
            polygon_2.setAutoDraw(True)
        if polygon_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > poly_training_t-frameTolerance:
                # keep track of stop time/frame for later
                polygon_2.tStop = t  # not accounting for scr refresh
                polygon_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(polygon_2, 'tStopRefresh')  # time at next scr refresh
                polygon_2.setAutoDraw(False)
        if polygon_2.status == STARTED:  # only update if drawing
            polygon_2.setPos(position_training, log=False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trainComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "train"-------
    for thisComponent in trainComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    train_loop.addData('fixation_training.started', fixation_training.tStartRefresh)
    train_loop.addData('fixation_training.stopped', fixation_training.tStopRefresh)
    train_loop.addData('cue_training.started', cue_training.tStartRefresh)
    train_loop.addData('cue_training.stopped', cue_training.tStopRefresh)
    sound_1.stop()  # ensure sound has stopped at end of routine
    train_loop.addData('sound_1.started', sound_1.tStartRefresh)
    train_loop.addData('sound_1.stopped', sound_1.tStopRefresh)
    train_loop.addData('gap_training.started', gap_training.tStartRefresh)
    train_loop.addData('gap_training.stopped', gap_training.tStopRefresh)
    train_loop.addData('circle_training_2.started', circle_training_2.tStartRefresh)
    train_loop.addData('circle_training_2.stopped', circle_training_2.tStopRefresh)
    train_loop.addData('circle_training.started', circle_training.tStartRefresh)
    train_loop.addData('circle_training.stopped', circle_training.tStopRefresh)
    train_loop.addData('feedback_training.started', feedback_training.tStartRefresh)
    train_loop.addData('feedback_training.stopped', feedback_training.tStopRefresh)
    eyetracker.sendMessage('train_end')
    # make sure the eyetracker recording stops
    if etRecord.status != FINISHED:
        etRecord.status = FINISHED
    train_loop.addData('polygon_2.started', polygon_2.tStartRefresh)
    train_loop.addData('polygon_2.stopped', polygon_2.tStopRefresh)
    # the Routine "train" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 3.0 repeats of 'train_loop'


# set up handler to look after randomisation of conditions etc
big_loop = data.TrialHandler(nReps=2.0, method='fullRandom', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='big_loop')
thisExp.addLoop(big_loop)  # add the loop to the experiment
thisBig_loop = big_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBig_loop.rgb)
if thisBig_loop != None:
    for paramName in thisBig_loop:
        exec('{} = thisBig_loop[paramName]'.format(paramName))

for thisBig_loop in big_loop:
    currentLoop = big_loop
    # abbreviate parameter names if possible (e.g. rgb = thisBig_loop.rgb)
    if thisBig_loop != None:
        for paramName in thisBig_loop:
            exec('{} = thisBig_loop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "test_start"-------
    continueRoutine = True
    # update component parameters for each repeat
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # keep track of which components have finished
    test_startComponents = [text_7, key_resp_3]
    for thisComponent in test_startComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    test_startClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "test_start"-------
    while continueRoutine:
        # get current time
        t = test_startClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=test_startClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_7* updates
        if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            text_7.setAutoDraw(True)
        
        # *key_resp_3* updates
        waitOnFlip = False
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in test_startComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "test_start"-------
    for thisComponent in test_startComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    big_loop.addData('text_7.started', text_7.tStartRefresh)
    big_loop.addData('text_7.stopped', text_7.tStopRefresh)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    big_loop.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        big_loop.addData('key_resp_3.rt', key_resp_3.rt)
    big_loop.addData('key_resp_3.started', key_resp_3.tStartRefresh)
    big_loop.addData('key_resp_3.stopped', key_resp_3.tStopRefresh)
    # the Routine "test_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    test_block_1_loop = data.TrialHandler(nReps=8.0, method='fullRandom', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('location_reward.xlsx', selection='0:2'),
        seed=None, name='test_block_1_loop')
    thisExp.addLoop(test_block_1_loop)  # add the loop to the experiment
    thisTest_block_1_loop = test_block_1_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTest_block_1_loop.rgb)
    if thisTest_block_1_loop != None:
        for paramName in thisTest_block_1_loop:
            exec('{} = thisTest_block_1_loop[paramName]'.format(paramName))
    
    for thisTest_block_1_loop in test_block_1_loop:
        currentLoop = test_block_1_loop
        # abbreviate parameter names if possible (e.g. rgb = thisTest_block_1_loop.rgb)
        if thisTest_block_1_loop != None:
            for paramName in thisTest_block_1_loop:
                exec('{} = thisTest_block_1_loop[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "test_block_1"-------
        continueRoutine = True
        # update component parameters for each repeat
        sound_2.setSound('right.wav', secs=0.1, hamming=True)
        sound_2.setVolume(1.0, log=False)
        circle_test_2.setPos(location)
        circle_test.setPos(location)
        eyetracker.sendMessage('block1')
        #Generates a random number between 0 and 1 for reward condition
        random_number = random.uniform(0,1)
        #Fixation Duration is between 1.0 to 0.8 seconds.
        number_t_1 = random.uniform(0,0.2)
        fixation_training_b1 = 1.0 - number_t_1
        #Cue Duration is between 0.4 to 0.7 seconds.
        cue_b1 = random.uniform(0.4,0.7)
        cue_training_b1 = cue_b1 + fixation_training_b1
        #Sounc Cue appear last 0.1 seconds of Cue
        sound_b1 = cue_training_b1 - 0.1
        #Gap Duration is 0.2 to 0.25 seconds.
        number_2_t_1 = random.uniform(0.2,0.25)
        gap_training_b1= cue_training_b1 + number_2_t_1
        #Target Duration is 1 second
        poly_training_b1 = 1+ gap_training_b1
        #Feedback Duration is 1 second 
        feedback_training_b1 = poly_training_b1 + 1
        startt_test = time.time()
        switch_test = 0
        
        # keep track of which components have finished
        test_block_1Components = [fixation_test_1, cue_block1, sound_2, gap_test_1, circle_test_2, circle_test, feedback_test_1, eyetracker_test, polygon_3]
        for thisComponent in test_block_1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        test_block_1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "test_block_1"-------
        while continueRoutine:
            # get current time
            t = test_block_1Clock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=test_block_1Clock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_test_1* updates
            if fixation_test_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_test_1.frameNStart = frameN  # exact frame index
                fixation_test_1.tStart = t  # local t and not account for scr refresh
                fixation_test_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_test_1, 'tStartRefresh')  # time at next scr refresh
                fixation_test_1.setAutoDraw(True)
            if fixation_test_1.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > fixation_training_b1-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_test_1.tStop = t  # not accounting for scr refresh
                    fixation_test_1.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fixation_test_1, 'tStopRefresh')  # time at next scr refresh
                    fixation_test_1.setAutoDraw(False)
            
            # *cue_block1* updates
            if cue_block1.status == NOT_STARTED and tThisFlip >= fixation_training_b1-frameTolerance:
                # keep track of start time/frame for later
                cue_block1.frameNStart = frameN  # exact frame index
                cue_block1.tStart = t  # local t and not account for scr refresh
                cue_block1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_block1, 'tStartRefresh')  # time at next scr refresh
                cue_block1.setAutoDraw(True)
            if cue_block1.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > cue_training_b1-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_block1.tStop = t  # not accounting for scr refresh
                    cue_block1.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(cue_block1, 'tStopRefresh')  # time at next scr refresh
                    cue_block1.setAutoDraw(False)
            if cue_block1.status == STARTED:  # only update if drawing
                cue_block1.setText(cue_text_b1, log=False)
            # start/stop sound_2
            if sound_2.status == NOT_STARTED and tThisFlip >= sound_b1-frameTolerance:
                # keep track of start time/frame for later
                sound_2.frameNStart = frameN  # exact frame index
                sound_2.tStart = t  # local t and not account for scr refresh
                sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                sound_2.play(when=win)  # sync with win flip
            if sound_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_2.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_2.tStop = t  # not accounting for scr refresh
                    sound_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_2, 'tStopRefresh')  # time at next scr refresh
                    sound_2.stop()
            
            # *gap_test_1* updates
            if gap_test_1.status == NOT_STARTED and tThisFlip >= fixation_training_b1-frameTolerance:
                # keep track of start time/frame for later
                gap_test_1.frameNStart = frameN  # exact frame index
                gap_test_1.tStart = t  # local t and not account for scr refresh
                gap_test_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gap_test_1, 'tStartRefresh')  # time at next scr refresh
                gap_test_1.setAutoDraw(True)
            if gap_test_1.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > gap_training_b1-frameTolerance:
                    # keep track of stop time/frame for later
                    gap_test_1.tStop = t  # not accounting for scr refresh
                    gap_test_1.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(gap_test_1, 'tStopRefresh')  # time at next scr refresh
                    gap_test_1.setAutoDraw(False)
            
            # *circle_test_2* updates
            if circle_test_2.status == NOT_STARTED and tThisFlip >= gap_training_b1-frameTolerance:
                # keep track of start time/frame for later
                circle_test_2.frameNStart = frameN  # exact frame index
                circle_test_2.tStart = t  # local t and not account for scr refresh
                circle_test_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle_test_2, 'tStartRefresh')  # time at next scr refresh
                circle_test_2.setAutoDraw(True)
            if circle_test_2.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > poly_training_b1-frameTolerance:
                    # keep track of stop time/frame for later
                    circle_test_2.tStop = t  # not accounting for scr refresh
                    circle_test_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(circle_test_2, 'tStopRefresh')  # time at next scr refresh
                    circle_test_2.setAutoDraw(False)
            
            # *circle_test* updates
            if circle_test.status == NOT_STARTED and tThisFlip >= gap_training_b1-frameTolerance:
                # keep track of start time/frame for later
                circle_test.frameNStart = frameN  # exact frame index
                circle_test.tStart = t  # local t and not account for scr refresh
                circle_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle_test, 'tStartRefresh')  # time at next scr refresh
                circle_test.setAutoDraw(True)
            if circle_test.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > poly_training_b1-frameTolerance:
                    # keep track of stop time/frame for later
                    circle_test.tStop = t  # not accounting for scr refresh
                    circle_test.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(circle_test, 'tStopRefresh')  # time at next scr refresh
                    circle_test.setAutoDraw(False)
            
            # *feedback_test_1* updates
            if feedback_test_1.status == NOT_STARTED and tThisFlip >= poly_training_b1-frameTolerance:
                # keep track of start time/frame for later
                feedback_test_1.frameNStart = frameN  # exact frame index
                feedback_test_1.tStart = t  # local t and not account for scr refresh
                feedback_test_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_test_1, 'tStartRefresh')  # time at next scr refresh
                feedback_test_1.setAutoDraw(True)
            if feedback_test_1.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > feedback_training_b1-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_test_1.tStop = t  # not accounting for scr refresh
                    feedback_test_1.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(feedback_test_1, 'tStopRefresh')  # time at next scr refresh
                    feedback_test_1.setAutoDraw(False)
            if feedback_test_1.status == STARTED:  # only update if drawing
                feedback_test_1.setText(text_feedback, log=False)
            try:
                #If target appears right side if the screen cue is "->", if it appears on left cue is ""<-"
                if circle_test.pos[0] == target_loc_right:
                    cue_text_b1 = cue_right
                elif circle_test.pos[0] == target_loc_left:
                    cue_text_b1 = cue_left
                else:
                    cue_text_b1 = ' '
                position = eyetracker.getLastGazePosition() #Gets last eye tracker position every frame.
                if circle_test.pos[0] == target_loc_right: #If target appears right side of screen
                    if circle_test_2.contains(position) and 0.2<random_number<=1: #If invisible target circle contains last eye movement.
                        text_feedback = high_reward_text
                    elif circle_test_2.contains(position) and 0<random_number<=0.2:
                        text_feedback = low_reward_text
                    else:
                        text_feedback = no_reward_text
                if circle_test.pos[0] == target_loc_left:
                    if circle_test_2.contains(position) and 0.2<random_number<=1:
                        text_feedback = low_reward_text
                    elif circle_test_2.contains(position) and 0<random_number<=0.2:
                        text_feedback = high_reward_text
                    else:
                        text_feedback = no_reward_text  
            except:
                text_feedback = no_reward_text
                position = (999,999)
            
            thisExp.addData('point_results',text_feedback)
            thisExp.addData('reward_loc',1)
            
            #Sending messages to eyetracker
            if time.time()-startt_test > fixation_training_b1 and switch_test ==0:
                eyetracker.sendMessage(f'fixation_offset_b1: {fixation_training_b1}')
                print(f'fixation_b1 {fixation_training_b1},{time.time()-startt_test}')
                switch_test= switch_test + 1
            if time.time()-startt_test>cue_training_b1 and switch_test ==1:
                eyetracker.sendMessage(f'cue_offset_b1: {cue_b1} {cue_training_b1}')
                print(f'cue_b1 {cue_training_b1},{time.time()-startt_test}')
                switch_test = switch_test+1
            if time.time()-startt_test>gap_training_b1 and switch_test ==2:
                eyetracker.sendMessage(f'gap_offset_b1: {number_2_t_1} {gap_training_b1} {text_feedback} {circle_test.pos[0]} 1 ')
                print(f' gap_b1 {gap_training_b1},{time.time()-startt_test}')
                switch_test = switch_test + 1
            if time.time()-startt_test>poly_training_b1 and switch_test ==3:
                eyetracker.sendMessage(f'poly_offset_b1:{poly_training_b1}')
                print(f'poly_b1 {poly_training_b1},{time.time()-startt_test} {text_feedback} {circle_test.pos[0]} r_h ')
                switch_test = switch_test +1
            
            
            
            # *eyetracker_test* updates
            if eyetracker_test.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                eyetracker_test.frameNStart = frameN  # exact frame index
                eyetracker_test.tStart = t  # local t and not account for scr refresh
                eyetracker_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(eyetracker_test, 'tStartRefresh')  # time at next scr refresh
                eyetracker_test.status = STARTED
            if eyetracker_test.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > poly_training_b1-frameTolerance:
                    # keep track of stop time/frame for later
                    eyetracker_test.tStop = t  # not accounting for scr refresh
                    eyetracker_test.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(eyetracker_test, 'tStopRefresh')  # time at next scr refresh
                    eyetracker_test.status = FINISHED
            
            # *polygon_3* updates
            if polygon_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_3.frameNStart = frameN  # exact frame index
                polygon_3.tStart = t  # local t and not account for scr refresh
                polygon_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_3, 'tStartRefresh')  # time at next scr refresh
                polygon_3.setAutoDraw(True)
            if polygon_3.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > feedback_training_b1-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_3.tStop = t  # not accounting for scr refresh
                    polygon_3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(polygon_3, 'tStopRefresh')  # time at next scr refresh
                    polygon_3.setAutoDraw(False)
            if polygon_3.status == STARTED:  # only update if drawing
                polygon_3.setPos(position, log=False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in test_block_1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "test_block_1"-------
        for thisComponent in test_block_1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        test_block_1_loop.addData('fixation_test_1.started', fixation_test_1.tStartRefresh)
        test_block_1_loop.addData('fixation_test_1.stopped', fixation_test_1.tStopRefresh)
        test_block_1_loop.addData('cue_block1.started', cue_block1.tStartRefresh)
        test_block_1_loop.addData('cue_block1.stopped', cue_block1.tStopRefresh)
        sound_2.stop()  # ensure sound has stopped at end of routine
        test_block_1_loop.addData('sound_2.started', sound_2.tStartRefresh)
        test_block_1_loop.addData('sound_2.stopped', sound_2.tStopRefresh)
        test_block_1_loop.addData('gap_test_1.started', gap_test_1.tStartRefresh)
        test_block_1_loop.addData('gap_test_1.stopped', gap_test_1.tStopRefresh)
        test_block_1_loop.addData('circle_test_2.started', circle_test_2.tStartRefresh)
        test_block_1_loop.addData('circle_test_2.stopped', circle_test_2.tStopRefresh)
        test_block_1_loop.addData('circle_test.started', circle_test.tStartRefresh)
        test_block_1_loop.addData('circle_test.stopped', circle_test.tStopRefresh)
        test_block_1_loop.addData('feedback_test_1.started', feedback_test_1.tStartRefresh)
        test_block_1_loop.addData('feedback_test_1.stopped', feedback_test_1.tStopRefresh)
        #Give points for each trial
        if text_feedback == high_reward_text:
            count = count + 100
        if text_feedback == low_reward_text:
            count = count + 10
        else:
            count = count + 0 
        count = round(count,2)
        #Create Scoreboard
        amount = f"You earned {count} points"
        if count > score_1:
            location_first = amount
            location_second = score_1_text
            location_third = score_2_text
            location_forth = score_3_text
            location_fifth = score_4_text
        else:
            location_first = score_1_text
        if score_1>count>score_2:
            location_first = score_1_text
            location_second = amount
            location_third = score_2_text
            location_forth = score_3_text
            location_fifth = score_4_text
        else:
            location_second = score_2_text
        if score_2>count>score_3:
            location_first = score_1_text
            location_second = score_2_text
            location_third = amount
            location_forth = score_3_text
            location_fifth = score_4_text
        else:
            location_third = score_3_text
        if score_3>count>score_4:
            location_first = score_1_text
            location_second = score_2_text
            location_third = score_3_text
            location_forth = amount
            location_fifth = score_4_text
        else:
            location_forth = score_4_text
        if score_4>count:
            location_first = score_1_text
            location_second = score_2_text
            location_third = score_3_text
            location_forth = score_4_text
            location_fifth = amount
        
        eyetracker.sendMessage('block1_end')
        # make sure the eyetracker recording stops
        if eyetracker_test.status != FINISHED:
            eyetracker_test.status = FINISHED
        test_block_1_loop.addData('polygon_3.started', polygon_3.tStartRefresh)
        test_block_1_loop.addData('polygon_3.stopped', polygon_3.tStopRefresh)
        # the Routine "test_block_1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 8.0 repeats of 'test_block_1_loop'
    
    
    # ------Prepare to start Routine "Amount_1"-------
    continueRoutine = True
    # update component parameters for each repeat
    text_6.setText('1)'+location_first
+'\n'+'\n'+'2)'+location_second
+'\n'+'\n'+'3)'+ location_third+'\n'+'\n'+'4)'+location_forth+'\n'+'\n'+'5)'+location_fifth


)
    key_resp_5.keys = []
    key_resp_5.rt = []
    _key_resp_5_allKeys = []
    text_12.setText(space)
    eyetracker.sendMessage('scoreboard_1')
    
    # keep track of which components have finished
    Amount_1Components = [text_6, key_resp_5, text_10, text_12]
    for thisComponent in Amount_1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Amount_1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Amount_1"-------
    while continueRoutine:
        # get current time
        t = Amount_1Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Amount_1Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_6* updates
        if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            text_6.setAutoDraw(True)
        
        # *key_resp_5* updates
        waitOnFlip = False
        if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_5.frameNStart = frameN  # exact frame index
            key_resp_5.tStart = t  # local t and not account for scr refresh
            key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
            key_resp_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_5.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_5.getKeys(keyList=['space'], waitRelease=False)
            _key_resp_5_allKeys.extend(theseKeys)
            if len(_key_resp_5_allKeys):
                key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *text_10* updates
        if text_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_10.frameNStart = frameN  # exact frame index
            text_10.tStart = t  # local t and not account for scr refresh
            text_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
            text_10.setAutoDraw(True)
        
        # *text_12* updates
        if text_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_12.frameNStart = frameN  # exact frame index
            text_12.tStart = t  # local t and not account for scr refresh
            text_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
            text_12.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Amount_1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Amount_1"-------
    for thisComponent in Amount_1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    big_loop.addData('text_6.started', text_6.tStartRefresh)
    big_loop.addData('text_6.stopped', text_6.tStopRefresh)
    # check responses
    if key_resp_5.keys in ['', [], None]:  # No response was made
        key_resp_5.keys = None
    big_loop.addData('key_resp_5.keys',key_resp_5.keys)
    if key_resp_5.keys != None:  # we had a response
        big_loop.addData('key_resp_5.rt', key_resp_5.rt)
    big_loop.addData('key_resp_5.started', key_resp_5.tStartRefresh)
    big_loop.addData('key_resp_5.stopped', key_resp_5.tStopRefresh)
    big_loop.addData('text_10.started', text_10.tStartRefresh)
    big_loop.addData('text_10.stopped', text_10.tStopRefresh)
    big_loop.addData('text_12.started', text_12.tStartRefresh)
    big_loop.addData('text_12.stopped', text_12.tStopRefresh)
    # the Routine "Amount_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "Break"-------
    continueRoutine = True
    # update component parameters for each repeat
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    eyetracker.sendMessage('welcome_test_2')
    
    # keep track of which components have finished
    BreakComponents = [text_4, key_resp_2]
    for thisComponent in BreakComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    BreakClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Break"-------
    while continueRoutine:
        # get current time
        t = BreakClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=BreakClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            text_4.setAutoDraw(True)
        
        # *key_resp_2* updates
        waitOnFlip = False
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BreakComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Break"-------
    for thisComponent in BreakComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    big_loop.addData('text_4.started', text_4.tStartRefresh)
    big_loop.addData('text_4.stopped', text_4.tStopRefresh)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    big_loop.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        big_loop.addData('key_resp_2.rt', key_resp_2.rt)
    big_loop.addData('key_resp_2.started', key_resp_2.tStartRefresh)
    big_loop.addData('key_resp_2.stopped', key_resp_2.tStopRefresh)
    # the Routine "Break" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    test_block_2_loop = data.TrialHandler(nReps=8.0, method='fullRandom', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('location_reward.xlsx', selection='0:2'),
        seed=None, name='test_block_2_loop')
    thisExp.addLoop(test_block_2_loop)  # add the loop to the experiment
    thisTest_block_2_loop = test_block_2_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTest_block_2_loop.rgb)
    if thisTest_block_2_loop != None:
        for paramName in thisTest_block_2_loop:
            exec('{} = thisTest_block_2_loop[paramName]'.format(paramName))
    
    for thisTest_block_2_loop in test_block_2_loop:
        currentLoop = test_block_2_loop
        # abbreviate parameter names if possible (e.g. rgb = thisTest_block_2_loop.rgb)
        if thisTest_block_2_loop != None:
            for paramName in thisTest_block_2_loop:
                exec('{} = thisTest_block_2_loop[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "test_block_2"-------
        continueRoutine = True
        # update component parameters for each repeat
        sound_3.setSound('right.wav', secs=0.1, hamming=True)
        sound_3.setVolume(1.0, log=False)
        circle_test_block_3.setPos(location)
        circle_test_block_2.setPos(location)
        #Sending message to eyetracker
        eyetracker.sendMessage('block2')
        #Generates a random number between 0 and 1 for reward condition
        random_number = random.uniform(0,1)
        #Fixation Duration is between 1.0 to 0.8 seconds.
        number_t_2 = random.uniform(0,0.2)
        fixation_training_b2 = 1.0 - number_t_2
        #Cue Duration is between 0.4 to 0.7 seconds.
        cue_b2 = random.uniform(0.4,0.7)
        cue_training_b2 = cue_b2 + fixation_training_b2
        #Sounc Cue appear last 0.1 seconds of Cue
        sound_b2 = cue_training_b2- 0.1
        #Gap Duration is 0.2 to 0.25 seconds.
        number_2_t_2 = random.uniform(0.2,0.25)
        gap_training_b2= cue_training_b2 + number_2_t_2
        #Target Duration is 1 second
        poly_training_b2 = 1+ gap_training_b2
        #Feedback Duration is 1 second 
        feedback_training_b2 = poly_training_b2 + 1.0
        
        startt_test2 = time.time()
        switch_test2 = 0
        # keep track of which components have finished
        test_block_2Components = [fixation_test_block_2, gap_test_block_2, cue_block_2, sound_3, circle_test_block_3, circle_test_block_2, test_feedback_block_2, eyetracker_test_block_2, polygon_4]
        for thisComponent in test_block_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        test_block_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "test_block_2"-------
        while continueRoutine:
            # get current time
            t = test_block_2Clock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=test_block_2Clock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_test_block_2* updates
            if fixation_test_block_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_test_block_2.frameNStart = frameN  # exact frame index
                fixation_test_block_2.tStart = t  # local t and not account for scr refresh
                fixation_test_block_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_test_block_2, 'tStartRefresh')  # time at next scr refresh
                fixation_test_block_2.setAutoDraw(True)
            if fixation_test_block_2.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > fixation_training_b2-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_test_block_2.tStop = t  # not accounting for scr refresh
                    fixation_test_block_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fixation_test_block_2, 'tStopRefresh')  # time at next scr refresh
                    fixation_test_block_2.setAutoDraw(False)
            
            # *gap_test_block_2* updates
            if gap_test_block_2.status == NOT_STARTED and tThisFlip >= fixation_training_b2-frameTolerance:
                # keep track of start time/frame for later
                gap_test_block_2.frameNStart = frameN  # exact frame index
                gap_test_block_2.tStart = t  # local t and not account for scr refresh
                gap_test_block_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gap_test_block_2, 'tStartRefresh')  # time at next scr refresh
                gap_test_block_2.setAutoDraw(True)
            if gap_test_block_2.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > gap_training_b2-frameTolerance:
                    # keep track of stop time/frame for later
                    gap_test_block_2.tStop = t  # not accounting for scr refresh
                    gap_test_block_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(gap_test_block_2, 'tStopRefresh')  # time at next scr refresh
                    gap_test_block_2.setAutoDraw(False)
            
            # *cue_block_2* updates
            if cue_block_2.status == NOT_STARTED and tThisFlip >= fixation_training_b2-frameTolerance:
                # keep track of start time/frame for later
                cue_block_2.frameNStart = frameN  # exact frame index
                cue_block_2.tStart = t  # local t and not account for scr refresh
                cue_block_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_block_2, 'tStartRefresh')  # time at next scr refresh
                cue_block_2.setAutoDraw(True)
            if cue_block_2.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > cue_training_b2-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_block_2.tStop = t  # not accounting for scr refresh
                    cue_block_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(cue_block_2, 'tStopRefresh')  # time at next scr refresh
                    cue_block_2.setAutoDraw(False)
            if cue_block_2.status == STARTED:  # only update if drawing
                cue_block_2.setText(cue_text_b2, log=False)
            # start/stop sound_3
            if sound_3.status == NOT_STARTED and tThisFlip >= sound_b2-frameTolerance:
                # keep track of start time/frame for later
                sound_3.frameNStart = frameN  # exact frame index
                sound_3.tStart = t  # local t and not account for scr refresh
                sound_3.tStartRefresh = tThisFlipGlobal  # on global time
                sound_3.play(when=win)  # sync with win flip
            if sound_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_3.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_3.tStop = t  # not accounting for scr refresh
                    sound_3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(sound_3, 'tStopRefresh')  # time at next scr refresh
                    sound_3.stop()
            
            # *circle_test_block_3* updates
            if circle_test_block_3.status == NOT_STARTED and tThisFlip >= gap_training_b2-frameTolerance:
                # keep track of start time/frame for later
                circle_test_block_3.frameNStart = frameN  # exact frame index
                circle_test_block_3.tStart = t  # local t and not account for scr refresh
                circle_test_block_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle_test_block_3, 'tStartRefresh')  # time at next scr refresh
                circle_test_block_3.setAutoDraw(True)
            if circle_test_block_3.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > poly_training_b2-frameTolerance:
                    # keep track of stop time/frame for later
                    circle_test_block_3.tStop = t  # not accounting for scr refresh
                    circle_test_block_3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(circle_test_block_3, 'tStopRefresh')  # time at next scr refresh
                    circle_test_block_3.setAutoDraw(False)
            
            # *circle_test_block_2* updates
            if circle_test_block_2.status == NOT_STARTED and tThisFlip >= gap_training_b2-frameTolerance:
                # keep track of start time/frame for later
                circle_test_block_2.frameNStart = frameN  # exact frame index
                circle_test_block_2.tStart = t  # local t and not account for scr refresh
                circle_test_block_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle_test_block_2, 'tStartRefresh')  # time at next scr refresh
                circle_test_block_2.setAutoDraw(True)
            if circle_test_block_2.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > poly_training_b2-frameTolerance:
                    # keep track of stop time/frame for later
                    circle_test_block_2.tStop = t  # not accounting for scr refresh
                    circle_test_block_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(circle_test_block_2, 'tStopRefresh')  # time at next scr refresh
                    circle_test_block_2.setAutoDraw(False)
            
            # *test_feedback_block_2* updates
            if test_feedback_block_2.status == NOT_STARTED and tThisFlip >= poly_training_b2-frameTolerance:
                # keep track of start time/frame for later
                test_feedback_block_2.frameNStart = frameN  # exact frame index
                test_feedback_block_2.tStart = t  # local t and not account for scr refresh
                test_feedback_block_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_feedback_block_2, 'tStartRefresh')  # time at next scr refresh
                test_feedback_block_2.setAutoDraw(True)
            if test_feedback_block_2.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > feedback_training_b2-frameTolerance:
                    # keep track of stop time/frame for later
                    test_feedback_block_2.tStop = t  # not accounting for scr refresh
                    test_feedback_block_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(test_feedback_block_2, 'tStopRefresh')  # time at next scr refresh
                    test_feedback_block_2.setAutoDraw(False)
            if test_feedback_block_2.status == STARTED:  # only update if drawing
                test_feedback_block_2.setText(text_feedback_2, log=False)
            try:
                if circle_test_block_2.pos[0] == target_loc_right:
                    cue_text_b2 = cue_right
                elif circle_test_block_2.pos[0] == target_loc_left:
                    cue_text_b2 = cue_left
                else:
                    cue_text_b2 = ' '
                    
                position_b2 =  eyetracker.getLastGazePosition()
                if circle_test_block_2.pos[0] == target_loc_right:
                    if circle_test_block_3.contains(position_b2) and 0.2<random_number<=1:
                        text_feedback_2 = low_reward_text
                    elif circle_test_block_3.contains(position_b2) and 0<random_number<=0.2:
                        text_feedback_2 = high_reward_text
                    else:
                        text_feedback_2 = no_reward_text
                if circle_test_block_2.pos[0] == target_loc_left:
                    if circle_test_block_3.contains(position_b2) and 0.2<random_number<=1:
                        text_feedback_2 = high_reward_text
                    elif circle_test_block_3.contains(position_b2) and 0<random_number<=0.2:
                        text_feedback_2 = low_reward_text
                    else:
                        text_feedback_2 = no_reward_text
            except:
                text_feedback_2  =no_reward_text
                position = (999,999)
                
            thisExp.addData('point_results',text_feedback_2)
            thisExp.addData('reward_loc',0)
            #Sending messages to eyetracker
            if time.time()-startt_test2 > fixation_training_b1 and switch_test2 ==0:
                eyetracker.sendMessage(f'fixation_offset_b2: {fixation_training_b2}')
                print(f'fixation_b2 {fixation_training_b2},{time.time()-startt_test2}')
                switch_test2= switch_test2 + 1
            if time.time()-startt_test2>cue_training_b2 and switch_test2 ==1:
                eyetracker.sendMessage(f'cue_offset_b2: {cue_b2} {cue_training_b2}')
                print(f'cue_b2 {cue_training_b2},{time.time()-startt_test2}')
                switch_test2 = switch_test2+1
            if time.time()-startt_test2>gap_training_b2 and switch_test2 ==2:
                eyetracker.sendMessage(f'gap_offset_b2: {number_2_t_2}  {gap_training_b2} {text_feedback_2} {circle_test_block_2.pos[0]} 0')
                print(f' gap_b2 {gap_training_b2},{time.time()-startt_test2}')
                switch_test2 = switch_test2 + 1
            if time.time()-startt_test2>poly_training_b1 and switch_test2 ==3:
                eyetracker.sendMessage(f'poly_offset_b2:{poly_training_b2}')
                print(f'poly_b2 {poly_training_b2},{time.time()-startt_test2} {text_feedback_2} {circle_test_block_2.pos[0]} r_l')
                switch_test2 = switch_test2 +1
            # *eyetracker_test_block_2* updates
            if eyetracker_test_block_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                eyetracker_test_block_2.frameNStart = frameN  # exact frame index
                eyetracker_test_block_2.tStart = t  # local t and not account for scr refresh
                eyetracker_test_block_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(eyetracker_test_block_2, 'tStartRefresh')  # time at next scr refresh
                eyetracker_test_block_2.status = STARTED
            if eyetracker_test_block_2.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > poly_training_b2-frameTolerance:
                    # keep track of stop time/frame for later
                    eyetracker_test_block_2.tStop = t  # not accounting for scr refresh
                    eyetracker_test_block_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(eyetracker_test_block_2, 'tStopRefresh')  # time at next scr refresh
                    eyetracker_test_block_2.status = FINISHED
            
            # *polygon_4* updates
            if polygon_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_4.frameNStart = frameN  # exact frame index
                polygon_4.tStart = t  # local t and not account for scr refresh
                polygon_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_4, 'tStartRefresh')  # time at next scr refresh
                polygon_4.setAutoDraw(True)
            if polygon_4.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > feedback_training_b2-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_4.tStop = t  # not accounting for scr refresh
                    polygon_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(polygon_4, 'tStopRefresh')  # time at next scr refresh
                    polygon_4.setAutoDraw(False)
            if polygon_4.status == STARTED:  # only update if drawing
                polygon_4.setPos(position, log=False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in test_block_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "test_block_2"-------
        for thisComponent in test_block_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        test_block_2_loop.addData('fixation_test_block_2.started', fixation_test_block_2.tStartRefresh)
        test_block_2_loop.addData('fixation_test_block_2.stopped', fixation_test_block_2.tStopRefresh)
        test_block_2_loop.addData('gap_test_block_2.started', gap_test_block_2.tStartRefresh)
        test_block_2_loop.addData('gap_test_block_2.stopped', gap_test_block_2.tStopRefresh)
        test_block_2_loop.addData('cue_block_2.started', cue_block_2.tStartRefresh)
        test_block_2_loop.addData('cue_block_2.stopped', cue_block_2.tStopRefresh)
        sound_3.stop()  # ensure sound has stopped at end of routine
        test_block_2_loop.addData('sound_3.started', sound_3.tStartRefresh)
        test_block_2_loop.addData('sound_3.stopped', sound_3.tStopRefresh)
        test_block_2_loop.addData('circle_test_block_3.started', circle_test_block_3.tStartRefresh)
        test_block_2_loop.addData('circle_test_block_3.stopped', circle_test_block_3.tStopRefresh)
        test_block_2_loop.addData('circle_test_block_2.started', circle_test_block_2.tStartRefresh)
        test_block_2_loop.addData('circle_test_block_2.stopped', circle_test_block_2.tStopRefresh)
        test_block_2_loop.addData('test_feedback_block_2.started', test_feedback_block_2.tStartRefresh)
        test_block_2_loop.addData('test_feedback_block_2.stopped', test_feedback_block_2.tStopRefresh)
        eyetracker.sendMessage('block2_end')
        
        
        #Give points for each trial
        if text_feedback_2 == high_reward_text:
            count = count + 100
        if text_feedback_2 == low_reward_text:
            count = count + 10
        else:
            count = count + 0 
        count = round(count,2)
        #Create Scoreboard
        amount = f"You earned {count} points"
        if count > score_1:
            location_first = amount
            location_second = score_1_text
            location_third = score_2_text
            location_forth = score_3_text
            location_fifth = score_4_text
        else:
            location_first = score_1_text
        if score_1>count>score_2:
            location_first = score_1_text
            location_second = amount
            location_third = score_2_text
            location_forth = score_3_text
            location_fifth = score_4_text
        else:
            location_second = score_2_text
        if score_2>count>score_3:
            location_first = score_1_text
            location_second = score_2_text
            location_third = amount
            location_forth = score_3_text
            location_fifth = score_4_text
        else:
            location_third = score_3_text
        if score_3>count>score_4:
            location_first = score_1_text
            location_second = score_2_text
            location_third = score_3_text
            location_forth = amount
            location_fifth = score_4_text
        else:
            location_forth = score_4_text
        if score_4>count:
            location_first = score_1_text
            location_second = score_2_text
            location_third = score_3_text
            location_forth = score_4_text
            location_fifth = amount
        
        # make sure the eyetracker recording stops
        if eyetracker_test_block_2.status != FINISHED:
            eyetracker_test_block_2.status = FINISHED
        test_block_2_loop.addData('polygon_4.started', polygon_4.tStartRefresh)
        test_block_2_loop.addData('polygon_4.stopped', polygon_4.tStopRefresh)
        # the Routine "test_block_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 8.0 repeats of 'test_block_2_loop'
    
    
    # ------Prepare to start Routine "Amount_2"-------
    continueRoutine = True
    # update component parameters for each repeat
    text_8.setText('1)'+location_first
+'\n'+'\n'+'2)'+location_second
+'\n'+'\n'+'3)'+ location_third+'\n'+'\n'+'4)'+location_forth+'\n'+'\n'+'5)'+location_fifth

)
    key_resp_6.keys = []
    key_resp_6.rt = []
    _key_resp_6_allKeys = []
    text_13.setText(space)
    eyetracker.sendMessage('scoreboard_2')
    
    # keep track of which components have finished
    Amount_2Components = [text_8, key_resp_6, text_11, text_13]
    for thisComponent in Amount_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Amount_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Amount_2"-------
    while continueRoutine:
        # get current time
        t = Amount_2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Amount_2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_8* updates
        if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_8.frameNStart = frameN  # exact frame index
            text_8.tStart = t  # local t and not account for scr refresh
            text_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
            text_8.setAutoDraw(True)
        
        # *key_resp_6* updates
        waitOnFlip = False
        if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_6.frameNStart = frameN  # exact frame index
            key_resp_6.tStart = t  # local t and not account for scr refresh
            key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
            key_resp_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_6.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_6.getKeys(keyList=['space'], waitRelease=False)
            _key_resp_6_allKeys.extend(theseKeys)
            if len(_key_resp_6_allKeys):
                key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # *text_11* updates
        if text_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_11.frameNStart = frameN  # exact frame index
            text_11.tStart = t  # local t and not account for scr refresh
            text_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_11, 'tStartRefresh')  # time at next scr refresh
            text_11.setAutoDraw(True)
        
        # *text_13* updates
        if text_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_13.frameNStart = frameN  # exact frame index
            text_13.tStart = t  # local t and not account for scr refresh
            text_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_13, 'tStartRefresh')  # time at next scr refresh
            text_13.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Amount_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Amount_2"-------
    for thisComponent in Amount_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    big_loop.addData('text_8.started', text_8.tStartRefresh)
    big_loop.addData('text_8.stopped', text_8.tStopRefresh)
    # check responses
    if key_resp_6.keys in ['', [], None]:  # No response was made
        key_resp_6.keys = None
    big_loop.addData('key_resp_6.keys',key_resp_6.keys)
    if key_resp_6.keys != None:  # we had a response
        big_loop.addData('key_resp_6.rt', key_resp_6.rt)
    big_loop.addData('key_resp_6.started', key_resp_6.tStartRefresh)
    big_loop.addData('key_resp_6.stopped', key_resp_6.tStopRefresh)
    big_loop.addData('text_11.started', text_11.tStartRefresh)
    big_loop.addData('text_11.stopped', text_11.tStopRefresh)
    big_loop.addData('text_13.started', text_13.tStartRefresh)
    big_loop.addData('text_13.stopped', text_13.tStopRefresh)
    # the Routine "Amount_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 2.0 repeats of 'big_loop'


# ------Prepare to start Routine "end_screen"-------
continueRoutine = True
routineTimer.add(10.000000)
# update component parameters for each repeat
key_resp_4.keys = []
key_resp_4.rt = []
_key_resp_4_allKeys = []
# keep track of which components have finished
end_screenComponents = [text_5, key_resp_4]
for thisComponent in end_screenComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
end_screenClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "end_screen"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = end_screenClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=end_screenClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_5* updates
    if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_5.frameNStart = frameN  # exact frame index
        text_5.tStart = t  # local t and not account for scr refresh
        text_5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
        text_5.setAutoDraw(True)
    if text_5.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_5.tStartRefresh + 10-frameTolerance:
            # keep track of stop time/frame for later
            text_5.tStop = t  # not accounting for scr refresh
            text_5.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_5, 'tStopRefresh')  # time at next scr refresh
            text_5.setAutoDraw(False)
    
    # *key_resp_4* updates
    waitOnFlip = False
    if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_4.frameNStart = frameN  # exact frame index
        key_resp_4.tStart = t  # local t and not account for scr refresh
        key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
        key_resp_4.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_4.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > key_resp_4.tStartRefresh + 10-frameTolerance:
            # keep track of stop time/frame for later
            key_resp_4.tStop = t  # not accounting for scr refresh
            key_resp_4.frameNStop = frameN  # exact frame index
            win.timeOnFlip(key_resp_4, 'tStopRefresh')  # time at next scr refresh
            key_resp_4.status = FINISHED
    if key_resp_4.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_4.getKeys(keyList=['space'], waitRelease=False)
        _key_resp_4_allKeys.extend(theseKeys)
        if len(_key_resp_4_allKeys):
            key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
            key_resp_4.rt = _key_resp_4_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in end_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end_screen"-------
for thisComponent in end_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_5.started', text_5.tStartRefresh)
thisExp.addData('text_5.stopped', text_5.tStopRefresh)
# check responses
if key_resp_4.keys in ['', [], None]:  # No response was made
    key_resp_4.keys = None
thisExp.addData('key_resp_4.keys',key_resp_4.keys)
if key_resp_4.keys != None:  # we had a response
    thisExp.addData('key_resp_4.rt', key_resp_4.rt)
thisExp.addData('key_resp_4.started', key_resp_4.tStartRefresh)
thisExp.addData('key_resp_4.stopped', key_resp_4.tStopRefresh)
thisExp.nextEntry()
eyetracker.setRecordingState(False)


# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
