# -*- coding: utf-8 -*-
from pytorch_forecasting.data import NaNLabelEncoder, GroupNormalizer
import tensorboard as tb
import tensorflow as tf
from pytorch_lightning import loggers
from pytorch_lightning.loggers import TensorBoardLogger
import pickle
from google.colab import drive
import dill as pickle
import holidays
from tqdm import tqdm
import gc
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.metrics import RMSE, MAPE
from torchmetrics import TweedieDevianceScore
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
!pip install pytorch_forecasting
!pip install pytorch_lightning
!pip install dill

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
random.seed(30)
np.random.seed(30)
tf.random.set_seed(30)
torch.manual_seed(30)

torch.cuda.manual_seed(30)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

drive.mount('/content/drive')

gl = pd.read_csv(
    "/content/drive/Shareddrives/MinneMUDAC/data/final_gl_Lan.csv")

"""# Data Preparation

"""

tran_df = gl[['Date',
              'NumberofGames',
             'DayofWeek',
              'VisitingTeam',
              'VisitingTeamLeague',
              'VisitingTeamGameNumber',
              'HomeTeam',
              'HomeTeamLeague',
              'HomeTeamGameNumber',
              'BallParkID',
              'Attendance',
              'year',
              'month',
              'week',
              'is_weekend',
              'opening_day',
              'is_public_holiday',
              'holidayName',
              'HomeTeam_City',
              'HomeTeam_State',
              'VisitingTeam_City',
              'VisitingTeam_State',
              'HomeTeam_ws_winner',
              'NBA_Team',
              'NFL_Team',
              'NHL_Team',
              'VisitingTeam_ws_winner',
              'MVP_in_visitingteam',
              'MVP_in_hometeam',
              'Cy_Young_in_visitingteam',
              'Cy_Young_in_hometeam',
              'season_end_rank',
              'season_end_w_l_ratio',
              'season_end_runs_mean',
              'season_end_runs_allowed_mean',
              'opponent_season_end_rank',
              'opponent_season_end_w_l_ratio',
              'opponent_season_end_runs_mean',
              'opponent_season_end_runs_allowed_mean',
              'home_as_cnt',
              'visiting_as_cnt',
              'previous_home_as_cnt',
              'previous_visiting_as_cnt',
              'Capacity',
              'home_pitch10',
              'home_bat10',
              'home_field10',
              'visiting_pitch10',
              'visiting_bat10',
              'visiting_field10']]

# add index of game for each team
tran_df = tran_df.sort_values(["HomeTeam", "Date", "NumberofGames"])
tran_df["date_int"] = tran_df.index+1
tran_df["date_int"] = tran_df.groupby("HomeTeam")["date_int"].rank().apply(int)

# remove tiebreaker game as we won't know whether there is a tiebreaker game in 2023
tran_df = tran_df[(tran_df['HomeTeamGameNumber'] != 163)]

# remove data of 2020 and 2021 due to covid
tran_df_2022 = tran_df[(tran_df['year'] != 2023) & (
    tran_df['year'] != 2020) & (tran_df['year'] != 2021)]
prediction = tran_df[tran_df['year'] == 2023]


def handle_nans(df):
    mean_attendance = tran_df_2022.groupby("HomeTeam")['Attendance'].mean(
    ).reset_index().rename(columns={'Attendance': "mean_attendance"})
    df = df.merge(mean_attendance, on="HomeTeam")
    df['Attendance'] = np.where((df['Attendance'].isna()) | (
        df['Attendance'] == 0), df['mean_attendance'], df['Attendance'])

    for f in df.columns:
        if df[f].dtype == "int64" or df[f].dtype == "float64":
            df[f] = df[f].fillna(-1)
        elif df[f].dtype == "object":
            df[f] = df[f].fillna("na")

    return(df)


tran_df_2022 = handle_nans(tran_df_2022)

tran_df_2022.DayofWeek = tran_df_2022.DayofWeek.astype(str).astype('category')
tran_df_2022.VisitingTeam = tran_df_2022.VisitingTeam.astype(
    str).astype('category')
tran_df_2022.VisitingTeamLeague = tran_df_2022.VisitingTeamLeague.astype(
    str).astype('category')
tran_df_2022.HomeTeam = tran_df_2022.HomeTeam.astype(str).astype('category')
tran_df_2022.HomeTeamLeague = tran_df_2022.HomeTeamLeague.astype(
    str).astype('category')
tran_df_2022.BallParkID = tran_df_2022.BallParkID.astype(
    str).astype('category')
tran_df_2022.year = tran_df_2022.year.astype(str).astype('category')
tran_df_2022.month = tran_df_2022.month.astype(str).astype('category')
tran_df_2022.week = tran_df_2022.week.astype(str).astype('category')
tran_df_2022.is_weekend = tran_df_2022.is_weekend.astype(
    str).astype('category')
tran_df_2022.NBA_Team = tran_df_2022.NBA_Team.astype(str).astype('category')
tran_df_2022.NFL_Team = tran_df_2022.NFL_Team.astype(str).astype('category')
tran_df_2022.NHL_Team = tran_df_2022.NHL_Team.astype(str).astype('category')
tran_df_2022.opening_day = tran_df_2022.opening_day.astype(
    str).astype('category')
tran_df_2022.is_public_holiday = tran_df_2022.is_public_holiday.astype(
    str).astype('category')
tran_df_2022.holidayName = tran_df_2022.holidayName.astype(
    str).astype('category')
tran_df_2022.HomeTeam_City = tran_df_2022.HomeTeam_City.astype(
    str).astype('category')
tran_df_2022.HomeTeam_State = tran_df_2022.HomeTeam_State.astype(
    str).astype('category')
tran_df_2022.VisitingTeam_City = tran_df_2022.VisitingTeam_City.astype(
    str).astype('category')
tran_df_2022.VisitingTeam_State = tran_df_2022.VisitingTeam_State.astype(
    str).astype('category')
tran_df_2022.HomeTeam_ws_winner = tran_df_2022.HomeTeam_ws_winner.apply(
    int).astype(str).astype('category')
tran_df_2022.VisitingTeam_ws_winner = tran_df_2022.VisitingTeam_ws_winner.apply(
    int).astype(str).astype('category')
tran_df_2022.Attendance = tran_df_2022.Attendance.astype(np.float32)
tran_df_2022.MVP_in_hometeam = tran_df_2022.MVP_in_hometeam.apply(
    int).astype(str).astype('category')
tran_df_2022.MVP_in_visitingteam = tran_df_2022.MVP_in_visitingteam.apply(
    int).astype(str).astype('category')
tran_df_2022.Cy_Young_in_hometeam = tran_df_2022.Cy_Young_in_hometeam.apply(
    int).astype(str).astype('category')
tran_df_2022.Cy_Young_in_visitingteam = tran_df_2022.Cy_Young_in_visitingteam.apply(
    int).astype(str).astype('category')

# use data prior to 2019 for training and data of 2022 for out of sample testing
train = tran_df_2022[pd.to_datetime(tran_df_2022['Date']).dt.year < 2019]
test = tran_df_2022[pd.to_datetime(tran_df_2022['Date']).dt.year == 2019]

"""# model building"""

# use 500 games to predict 81 games
max_prediction_length = 81
max_encoder_length = 500

# Let's create a Dataset
training = TimeSeriesDataSet(
    train,
    time_idx="date_int",
    target="Attendance",
    group_ids=["HomeTeam"],
    # keep encoder length long (as it is in the validation set)
    min_encoder_length=max_prediction_length//2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["HomeTeam", "HomeTeamLeague",
                         "HomeTeam_City", "HomeTeam_State"],
    time_varying_known_categoricals=['DayofWeek', 'year', 'VisitingTeam', 'VisitingTeamLeague',
                                     'month', 'week', 'is_weekend',
                                     'opening_day', 'is_public_holiday',
                                     'HomeTeam_ws_winner', 'VisitingTeam_ws_winner', 'MVP_in_visitingteam',
                                     'MVP_in_hometeam', 'Cy_Young_in_visitingteam',
                                     'Cy_Young_in_hometeam'
                                     ],
    static_reals=['Capacity'],
    # variable_groups={"is_holiday": ["is_holiday"]},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["home_pitch10",
                              "home_bat10",
                              "home_field10",
                              "visiting_pitch10",
                              "visiting_bat10",
                              "visiting_field10",
                              'HomeTeam_ws_winner',
                              'VisitingTeam_ws_winner',
                              'Capacity',
                              'season_end_rank',
                              'season_end_w_l_ratio',
                              'season_end_runs_mean',
                              'season_end_runs_allowed_mean',
                              'opponent_season_end_rank',
                              'opponent_season_end_w_l_ratio',
                              'opponent_season_end_runs_mean',
                              'opponent_season_end_runs_allowed_mean',
                              "VisitingTeamGameNumber",
                              "HomeTeamGameNumber"
                              ],
    time_varying_unknown_reals=[
        'Attendance'
    ],
    target_normalizer=GroupNormalizer(
        groups=["HomeTeam"], transformation="softplus"
    ),  # use softplus and normalize by group

    lags={'Attendance': [1, 3, 5, 7, 14]},
    add_encoder_length=True,
    add_relative_time_idx=True,
    add_target_scales=True,
    allow_missing_timesteps=True,
    categorical_encoders={"HomeTeam_ws_winner": NaNLabelEncoder(add_nan=True),
                          "VisitingTeam_ws_winner": NaNLabelEncoder(add_nan=True),
                          'year': NaNLabelEncoder(add_nan=True)
                          }
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(
    training, train, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0)

# hyperparameter tuning


# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    # use Optuna to find ideal learning rate or use in-built learning rate finder
    use_learning_rate_finder=False,
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)

"""Trial 0 finished with value: 981.5067749023438 and parameters: {'gradient_clip_val': 0.47392885281873304, 'hidden_size': 17, 'dropout': 0.12057343228729256, 'hidden_continuous_size': 16, 'attention_head_size': 3, 'learning_rate': 0.012067986205675296}."""

# retrain with best hyperparameters

PATIENCE = 30
MAX_EPOCHS = 200
LEARNING_RATE = 0.012067986205675296
OPTUNA = False
early_stop_callback = EarlyStopping(
    monitor="train_loss", min_delta=0.001, patience=PATIENCE, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
# logging results to a tensorboard
logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    gpus=1,
    enable_model_summary=True,
    gradient_clip_val=0.47392885281873304,
    limit_train_batches=10,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LEARNING_RATE,
    lstm_layers=2,
    hidden_size=17,
    attention_head_size=3,
    dropout=0.12057343228729256,
    hidden_continuous_size=16,
    output_size=1,  # 7 quantiles by default
    loss=MAPE(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4
)

tft.to(DEVICE)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# load the best model according to the validation loss
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# in sample validation
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
np.sqrt(((actuals - predictions)**2).mean())
print('MAPE:', (np.abs(actuals - predictions)/actuals).mean())
print('RMSE:', np.sqrt(((actuals - predictions)**2).mean()))

# out of sample testing


def testing_MAPE(tft):
    # max_date = train.groupby("HomeTeam")["date_int"].max().reset_index()
    # last_data = max_date.merge(train, on = ["HomeTeam","date_int"], how = "left")

    # select last 500 from data (max_encoder_length is 500)
    encoder_data = train.groupby('HomeTeam').tail(500)

    decoder_data = test
# combine encoder and decoder data
    new_prediction_data = pd.concat(
        [encoder_data, decoder_data], ignore_index=True)

    # 'quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]'
    new_raw_predictions, new_x = tft.predict(
        new_prediction_data, mode="raw", return_x=True)
    pred = new_raw_predictions[0].numpy().reshape(2430,)
    pred = np.delete(pred, 180)
    actual = test['Attendance'].array

    return (np.abs(actual - pred)/actual).mean()


testing_MAPE(best_tft)

"""# prediction for 2023"""

prediction = tran_df[tran_df['year'] == 2023]

# mapping team league for home team and visiting team
league_maping = encoder_data[['VisitingTeam',
                              "VisitingTeamLeague"]].drop_duplicates()
prediction = pd.merge(prediction, league_maping, on="VisitingTeam")
prediction = prediction.drop('VisitingTeamLeague_x', axis=1)
prediction = pd.merge(prediction, league_maping,
                      left_on="HomeTeam", right_on="VisitingTeam")
prediction = prediction.drop('VisitingTeam_y', axis=1)
prediction = prediction.drop('HomeTeamLeague', axis=1)
prediction = prediction.rename(columns={
                               "VisitingTeamLeague_x": "VisitingTeamLeague", "VisitingTeamLeague_y": "HomeTeamLeague"})
prediction = prediction.rename(columns={"VisitingTeam_x": "VisitingTeam"})
prediction.DayofWeek = prediction.DayofWeek.astype(str).astype('category')


def handle_2023_nans(df):
    df['Attendance'] = df['Attendance'].fillna(0)
    for f in df.columns:
        if df[f].dtype == "int64" or df[f].dtype == "float64":
            df[f] = df[f].fillna(-1)
        elif df[f].dtype == "object":
            df[f] = df[f].fillna("na")

    return(df)


prediction = handle_2023_nans(prediction)

prediction.DayofWeek = prediction.DayofWeek.astype(str).astype('category')
prediction.VisitingTeam = prediction.VisitingTeam.astype(
    str).astype('category')
prediction.VisitingTeamLeague = prediction.VisitingTeamLeague.astype(
    str).astype('category')
prediction.HomeTeam = prediction.HomeTeam.astype(str).astype('category')
prediction.HomeTeamLeague = prediction.HomeTeamLeague.astype(
    str).astype('category')
prediction.BallParkID = prediction.BallParkID.astype(str).astype('category')
prediction.year = prediction.year.astype(int)
prediction.month = prediction.month.astype(str).astype('category')
prediction.week = prediction.week.astype(str).astype('category')
prediction.is_weekend = prediction.is_weekend.astype(str).astype('category')
prediction.opening_day = prediction.opening_day.astype(str).astype('category')
prediction.is_public_holiday = prediction.is_public_holiday.astype(
    str).astype('category')
prediction.holidayName = prediction.holidayName.astype(str).astype('category')
prediction.HomeTeam_City = prediction.HomeTeam_City.astype(
    str).astype('category')
prediction.HomeTeam_State = prediction.HomeTeam_State.astype(
    str).astype('category')
prediction.VisitingTeam_City = prediction.VisitingTeam_City.astype(
    str).astype('category')
prediction.VisitingTeam_State = prediction.VisitingTeam_State.astype(
    str).astype('category')
prediction.HomeTeam_ws_winner = prediction.HomeTeam_ws_winner.apply(
    int).astype(str).astype('category')
prediction.VisitingTeam_ws_winner = prediction.VisitingTeam_ws_winner.apply(
    int).astype(str).astype('category')
prediction.NBA_Team = prediction.NBA_Team.astype(str).astype('category')
prediction.NFL_Team = prediction.NFL_Team.astype(str).astype('category')
prediction.NHL_Team = prediction.NHL_Team.astype(str).astype('category')
prediction.Attendance = prediction.Attendance.astype(np.float32)
prediction.MVP_in_hometeam = prediction.MVP_in_hometeam.apply(
    int).astype(str).astype('category')
prediction.MVP_in_visitingteam = prediction.MVP_in_visitingteam.apply(
    int).astype(str).astype('category')
prediction.Cy_Young_in_hometeam = prediction.Cy_Young_in_hometeam.apply(
    int).astype(str).astype('category')
prediction.Cy_Young_in_visitingteam = prediction.Cy_Young_in_visitingteam.apply(
    int).astype(str).astype('category')

grouped = train.groupby('HomeTeam')

# Define a function to extract the last 500 numbers from a series


def extract_last_500(series):
    return series.iloc[-500:]


# Apply the function to the numbers_column for each group and concatenate the results
last_500 = pd.concat([grouped.apply(extract_last_500)["date_int"]
                     for team, grouped in grouped])

encoder_data = train.groupby('HomeTeam').tail(500)

decoder_data = prediction
# combine encoder and decoder data
new_prediction_data = pd.concat(
    [encoder_data, decoder_data], ignore_index=True)


# 'quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]'
new_raw_predictions, new_x = best_tft.predict(
    new_prediction_data, mode="raw", return_x=True)
pred = new_raw_predictions[0].numpy().reshape(2430,)

prediction['predicted_attendance'] = pred
