import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import joblib


def train_polynomial_regression():
    # 데이터 불러오기
    df = pd.read_csv('turnout.csv')

    # 시간 데이터를 숫자로 변환(예: '7시' -> 7)
    df['시간'] = df['시간'].str.replace('시', '').astype(int)

    # PolynomialFeatures 객체 생성 (2차항으로 설정)
    poly = PolynomialFeatures(degree=3)

    # X를 2차항으로 변환
    X = df['시간'].values.reshape(-1, 1)
    X_poly = poly.fit_transform(X)

    # 'poly' 객체를 저장합니다.
    joblib.dump(poly, 'poly.pkl')

    # 각 년도의 데이터로 모델을 학습하고 모델 저장
    for year in ['2017대선', '2018지선', '2020총선']:
        # y(투표율) 구성하기
        y = df[year]

        # 다항 회귀 모델을 초기화합니다.
        model = LinearRegression()

        # 다항 회귀 모델을 학습합니다.
        model.fit(X_poly, y)

        print(f'계수: {model.coef_}')  # 계수 출력
        print(f'절편: {model.intercept_}')  # 절편 출력

        # 모델을 저장합니다.
        joblib.dump(model, f'{year}_model.pkl')


def predict_turnout(input_values):
    # 'poly' 객체와 모델을 불러옵니다.
    poly = joblib.load('poly.pkl')
    models = [joblib.load(f'{year}_model.pkl') for year in ['2017대선', '2018지선', '2020총선']]

    # 투표율 차이를 저장할 2차원 배열 초기화
    turnout_diff = []

    # 각 년도에 대해 모델을 불러와서 예측을 수행합니다.
    for model in models:
        year_diff = []
        last_prediction = 0
        for hour in range(7, 19):  # 7시부터 18시까지
            input_data = np.array([[hour]])
            input_data_poly = poly.transform(input_data)  # 다항 회귀에 맞게 변환
            prediction = model.predict(input_data_poly)
            if hour != 7:  # 첫 시간대가 아니라면 투표율 차이 계산
                diff = prediction[0] - last_prediction
                year_diff.append(diff)
            last_prediction = prediction[0]

        turnout_diff.append(year_diff)

    for i in range(2, len(turnout_diff[0])):
        mean_diff = np.mean([turnout_diff[j][i] for j in range(3)])
        input_values.append(input_values[-1] + mean_diff)

    return input_values
