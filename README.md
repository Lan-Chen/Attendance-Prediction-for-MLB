# Attendance-Prediction-for-MLB
Winner of [2023 MinneMUDAC Student Data Science Challenge](https://minneanalytics.org/minnemudac2023/)

Build predictive models for the game-by-game attendance for home games for all MLB teams for the 2023 season and identify factors that tend to influence home-game attendance.

## Team Members
A big thanks and credit to all the team members who made this project possible:

- Congyi Zhang (zhan8373@umn.edu) 
- Jichen Liu (liu02354@umn.edu) 
- Lan Chen (chen7613@umn.edu) 
- Rio Pan (pan00246@umn.edu) 
- Simin Liao (liao0150@umn.edu)

## Deliverables
- [Presentation video](https://www.youtube.com/watch?v=OOTj8_1UaQA)
- [Deck](https://github.com/Lan-Chen/Attendance-Prediction-for-MLB/tree/main/Deck)
- [Code](https://github.com/Lan-Chen/Attendance-Prediction-for-MLB/tree/main/Code)

## Project Overview
Attendance is a crucial factor for the success of MLB teams. Accurately predicting attendance can have significant impacts on both long-term and short-term profitability and operational efficiency. However, the prediction of MLB attendance can be complicated due to multiple factors, leading to inaccurate forecasts and potentially suboptimal business decisions.

To address this issue:
  - We firstly utilize a ***pre-season attendance*** prediction model to help the MLB teams make attendance predictions before the season. The prediction results can help with long-term business planning like game scheduling, staff hiring and season ticket price adjustment.
  - Furthermore, we provide an ***in-season attendance*** model which includes data collected during the new season, such as latest game performance and player list, to dynamically predict attendance and facilitate short-term decsions making.
  - Lastly, to identify ***important factors*** and understand how they affect attendance across teams, we interpret the important factors from feature importance graph.
    
![image](https://media.github.umn.edu/user/19808/files/86bde54e-b670-4cd2-a8a7-1c62eba5fbc0)


### Feature Engineering

We integrate data from multiple sources and build features that can be grouped into 3 buckets:
  - Team performance,
  - player and,
  - calendar

![image](https://media.github.umn.edu/user/19808/files/b677b59e-1d50-4744-a00a-f6990a57fe49)

Based on the features, we deliver the following models and results.


### Temporal Fusion Transfomer(TFT) Model and 2023 Attendance Predictions

**TFT Structure**
![image](https://media.github.umn.edu/user/19808/files/ef4a80cc-7ded-478e-8045-92fa82abf239)

We use Temporal Fusion Transfomer for long-term attendance prediction since the model has advantages as:
* Capable of processing multiple heterogeneous time series data simultaneously.
* Takes into account the impact of all historical data when forecasting time series, resulting in more accurate predictions.
* Achieves high performance across multi-horizon forecasting, providing accurate predictions across different time horizons.
#### Model Outcome
![image](https://media.github.umn.edu/user/19808/files/13d57caf-eeed-4981-a3fb-feede698f185)

### LightGBM Model for in-Season Attendance Prediction
We use LightGBM for its characteristics:
* Capable of efficiently processing a large number of features and automatically generating feature importance, allowing businesses to gain insights into the most important factors affecting their predictions.
* Fast training speed allows for quick iteration and retraining of the model when new data is received, making it a flexible and adaptable tool for short-term time series forecasting
#### Model Outcome
![image](https://media.github.umn.edu/user/19808/files/14e7c66f-0f40-4847-9623-6821d9ea4489)

### Feature Importance, Partial Dependence Plot and intepretation
The deliverables are:
* Feature importance score from the LightGBM model to select important factors
* Partial Dependence Plot(PDP) to show the relationship between attendance and the factors
* Selected teams based on clustering results and detailed analysis towards across teams differences
* Recommendations to optimize game schedule, marketing timing and marketing content
#### Overall Feature Importance
![image](https://media.github.umn.edu/user/19808/files/34a6d299-bf96-487a-bd66-14aa117f86d5)
#### Important Factor #1: Day of Week
![image](https://media.github.umn.edu/user/19808/files/d019d0dd-89e9-488d-8cdc-111ec40fa1f5)
