import pickle

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api

import bandit
import polynomial_regression

# Flask 애플리케이션 및 RESTful API 객체 생성, CORS 적용
app = Flask(__name__)
api = Api(app)
CORS(app)


# RESTful API에서 사용할 리소스 정의
class BanditResource(Resource):
    # HTTP POST 요청을 처리하는 메서드
    def post(self):
    # Bandit 을 생성하는 함수

        # 요청에서 JSON 데이터를 가져오기
        data = request.get_json()

        TYPE = str(data.get('type'))
        ID = str(data.get('id'))
        banner_list = data.get('bannerList', [])

        new_bandit = bandit.DriftingFiniteBernoulliBanditTS(banner_list)

        with open(TYPE + '_bandit_model_' + ID + '.pkl', 'wb') as file:
            pickle.dump(new_bandit, file)

        response_data = {}

        # 응답할 데이터가 없으므로 빈것과 200을 반환
        return response_data, 200

    def patch(self):
        # Bandit 을 업데이트 하는 함수
        data = request.get_json()

        TYPE = str(data.get('type'))
        ID = str(data.get('id'))
        success_arm_ids = data.get('successList', [])
        failure_arm_ids = data.get('failureList', [])

        with open(TYPE + '_bandit_model_' + ID + '.pkl', 'rb') as file:
            loaded_bandit = pickle.load(file)

        loaded_bandit.update_observations(success_arm_ids, failure_arm_ids)

        with open(TYPE + '_bandit_model_' + ID + '.pkl', 'wb') as file:
            pickle.dump(loaded_bandit, file)

        response_data = {}

        return response_data, 200

    def get(self):
    # Bandit을 사용해 관련 정보를 GET 하는 함수
        TYPE = request.args.get('type')
        ID = request.args.get('id')

        with open(TYPE + '_bandit_model_' + ID + '.pkl', 'rb') as file:
            loaded_bandit = pickle.load(file)

        # 적절한 로직으로 응답 데이터 생성
        response_data = {"orderdList": loaded_bandit.pick_action()}

        # 응답 데이터와 HTTP 상태 코드 200을 반환
        return response_data, 200


class PolynomialRegressionResource(Resource):
    def post(self):
        polynomial_regression.train_polynomial_regression()

        response_data = {}
        return response_data, 200

    def get(self):
        data = request.get_json()

        input_values = data.get('inputValues', [])

        response_data = {"predictedTurnout": polynomial_regression.predict_turnout(input_values)}
        return response_data, 200


# 애플리케이션에 리소스를 추가하고 해당 리소스에 대한 엔드포인트 설정
api.add_resource(BanditResource, '/bandit')
api.add_resource(PolynomialRegressionResource, '/polynomial-regression')

# 스크립트가 직접 실행될 때만 서버를 실행
if __name__ == '__main__':
    app.run()
