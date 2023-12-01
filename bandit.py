import numpy as np

class DriftingFiniteBernoulliBanditTS:
    """Thompson sampling on finite armed bandit with a drifting effect."""

    # alpha0 beta0는 1, 감마값은 0.01 로 고정하고 arms 를 받아 bandit 생성
    def __init__(self, arms, gamma=0.01, a0=1, b0=1):
        self.arms = arms
        self.gamma = gamma
        self.a0 = a0
        self.b0 = b0
        self.prior_success = np.array([a0] * len(arms))
        self.prior_failure = np.array([b0] * len(arms))
        self.arm_to_index = {arm_id: index for index, arm_id in enumerate(arms)}  # ID를 인덱스로 매핑

    def update_observations(self, success_arm_ids, failure_arm_ids):

        # alpha 파라미터 값 갱신
        for arm_id in success_arm_ids:
            action_index = self.arm_to_index[arm_id]
            for i in range(len(self.arms)):
                if i == action_index:
                    new_value = self.prior_success[i] * (1 - self.gamma) + 1 # 클릭된 샘플은 (1-gamma) 곱한 값에 +1
                    self.prior_success = np.where(np.arange(len(self.prior_success)) == i, new_value, self.prior_success)
                else:
                    new_value = self.prior_success[i] * (1 - self.gamma) # 클릭되지 않은 샘플은 (1-gamma)만 곱함
                    self.prior_success = np.where(np.arange(len(self.prior_success)) == i, new_value, self.prior_success)
                    # (1-gamma) 를 곱하는 이유: 최근의 선택에 가중치를 주기 위함
        # beta 파라미터 값 갱신
        for arm_id in failure_arm_ids:
            action_index = self.arm_to_index[arm_id]
            for i in range(len(self.arms)):
                if i == action_index:
                    new_value = self.prior_failure[i] * (1 - self.gamma) + 1 # 클릭되지 않은 샘플
                    self.prior_failure = np.where(np.arange(len(self.prior_failure)) == i, new_value, self.prior_failure) # 클릭된 경우, 성공을 기록합니다.
                else:
                    new_value = self.prior_failure[i] * (1 - self.gamma) # 클릭된 샘플
                    self.prior_failure = np.where(np.arange(len(self.prior_failure)) == i, new_value, self.prior_failure)

    # 샘플링 후 정렬된 순서인 sorted_arm_ids 반환
    def pick_action(self):
        """Pick an action from available arms based on Thompson sampling with Beta distribution."""
        # 베타 분포를 기반으로 각 암의 추정된 평균값을 샘플링합니다.
        sampled_means = np.random.beta(self.prior_success, self.prior_failure)
        sorted_indices = np.argsort(sampled_means)[::-1]
        sorted_arm_ids = [self.arms[index] for index in sorted_indices]

        return sorted_arm_ids
