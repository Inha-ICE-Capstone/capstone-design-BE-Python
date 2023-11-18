# Flask와 Flask-RESTful을 import
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

import bandit
import pickle

# Flask 애플리케이션 및 RESTful API 객체 생성
app = Flask(__name__)
api = Api(app)
CORS(app)

# RESTful API에서 사용할 리소스 정의
class BanditResource(Resource):
    # HTTP POST 요청을 처리하는 메서드
    def post(self):
        # 요청에서 JSON 데이터를 가져오기
        data = request.get_json()

        # JSON 데이터에서 'ballotId' 키를 사용하여 값 추출
        ballot_id = str(data.get('ballotId'))
        banner_list = data.get('bannerList', [])

        new_bandit = bandit.DriftingFiniteBernoulliBanditTS(banner_list)

        with open('bandit_model_' + ballot_id + '.pkl', 'wb') as file:
            pickle.dump(new_bandit, file)

        # 적절한 로직으로 응답 데이터 생성
        response_data = {}

        # 응답 데이터와 HTTP 상태 코드 200을 반환
        return response_data, 200
    def patch(self):
        # 요청에서 JSON 데이터를 가져오기
        data = request.get_json()

        # JSON 데이터에서 'ballotId' 키를 사용하여 값 추출
        ballot_id = str(data.get('ballotId'))
        success_arm_ids = data.get('successList', [])
        failure_arm_ids = data.get('failureList', [])

        with open('bandit_model_' + ballot_id + '.pkl', 'rb') as file:
            loaded_bandit = pickle.load(file)

        loaded_bandit.update_observations(success_arm_ids, failure_arm_ids)

        with open('bandit_model_' + ballot_id + '.pkl', 'wb') as file:
            pickle.dump(loaded_bandit, file)

        # 적절한 로직으로 응답 데이터 생성
        response_data = {}

        # 응답 데이터와 HTTP 상태 코드 200을 반환
        return response_data, 200

    @app.route('/<string:ballotId>')
    def get(ballotId):
        ballot_id = ballotId

        with open('bandit_model_' + ballot_id + '.pkl', 'rb') as file:
            loaded_bandit = pickle.load(file)

        # 적절한 로직으로 응답 데이터 생성
        response_data = {"orderdList": loaded_bandit.pick_action()}

        # 응답 데이터와 HTTP 상태 코드 200을 반환
        return response_data, 200

# 애플리케이션에 리소스를 추가하고 해당 리소스에 대한 엔드포인트 설정
api.add_resource(BanditResource, '/')

# 스크립트가 직접 실행될 때만 서버를 실행
if __name__ == '__main__':
    app.run()