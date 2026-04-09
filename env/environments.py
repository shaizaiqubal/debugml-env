from env.models import Observation
import random
#random.seed(42)

def compute_score(state):
    return round(
        0.5 * state.accuracy +
        0.25 * state.precision +
        0.25 * state.recall,
        2
    )

class DebugMLEnv:

    def __init__(self):
        self.cur_state = None
        self.step_count = 0
        self.max_steps = 15
        self.last_action = None
        self.task_name = None

    def reset(self, task_name=None):

        self.step_count = 0
        self.last_action = None
        self.task_name = task_name

        if task_name == "fix_basics":
            scaling = False
            feature_count = 5
            test_split = 0.9
            model_type = "linear"
            accuracy = round(random.uniform(0.5, 0.7), 2)
            precision = round(accuracy - 0.05, 2)
            recall = round(accuracy - 0.03, 2)

        elif task_name == "optimize_features":
            scaling = True
            feature_count = 6
            test_split = 0.2
            model_type = "linear"
            accuracy = round(random.uniform(0.5, 0.7), 2)
            precision = round(accuracy - 0.05, 2)
            recall = round(accuracy - 0.03, 2)

        elif task_name == "full_pipeline_optimization":
            scaling = random.choice([True, False])
            feature_count = random.randint(1, 6)
            test_split = random.choice([0.1, 0.2, 0.4, 0.5, 0.9])
            model_type = random.choice(["linear", "svm", "tree"])
            accuracy = round(random.uniform(0.5, 0.7), 2)
            precision = round(accuracy - 0.05, 2)
            recall = round(accuracy - 0.03, 2)

        elif task_name == "stability_optimization":
                scaling = True
                feature_count = 4
                test_split = 0.2
                model_type = random.choice(["linear", "svm", "tree"])
                accuracy = round(random.uniform(0.75, 0.82), 2)
                precision = round(accuracy - 0.05, 2)
                recall = round(accuracy - 0.03, 2)

        else: 
            scaling = random.choice([True, False])
            feature_count = random.randint(1,6)
            test_split = random.choice([0.1, 0.2, 0.4, 0.5, 0.9])
            model_type = random.choice( ['linear', 'svm', 'tree'])

            accuracy = round(random.uniform(0.4, 0.7), 2)
            precision = round(accuracy - 0.05, 2)
            recall = round(accuracy - 0.03, 2)

        self.cur_state = Observation(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            scaling=scaling,
            feature_count=feature_count,
            test_split=test_split,
            model_type=model_type
        )

        return self.cur_state
    
    def step(self, action):
        
        if self.cur_state is None:
            raise RuntimeError("Call reset() before step()")

        old_score = compute_score(self.cur_state)

        cur_accuracy = self.cur_state.accuracy
        scaling = self.cur_state.scaling
        feature_count = self.cur_state.feature_count
        test_split = self.cur_state.test_split

        action_type = action.type
        penalty = 0  

        if self.last_action == action_type:    # penalize loops
            penalty = -0.02

        self.last_action = action_type

        # ------------------ ACTION LOGIC ------------------

        if action_type == 'add_scaling':
            if not scaling:
                scaling = True

                if self.cur_state.model_type == "linear":
                    delta = random.uniform(0.08, 0.12)
                elif self.cur_state.model_type == "svm":
                    delta = random.uniform(0.05, 0.10)
                else:
                    delta = random.uniform(0.0, 0.03)

                new_accuracy = cur_accuracy + delta

            else:
                new_accuracy = cur_accuracy
                penalty = -0.01   # <-- FIXED

            self.cur_state.scaling = scaling

        # -------------------------------------------------

        elif action_type == 'fix_split':
            if test_split == 0.2:
                new_accuracy = cur_accuracy
                penalty = -0.01   # <-- FIXED
            else:
                self.cur_state.test_split = 0.2
                delta = random.uniform(0.05, 0.10)
                new_accuracy = cur_accuracy + delta

        # -------------------------------------------------

        elif action_type == 'add_feature':
            if feature_count == 6:
                new_accuracy = cur_accuracy
                penalty = -0.01   # <-- FIXED
            else:
                if feature_count < 3:
                    delta = random.uniform(0.03, 0.08)
                elif feature_count <= 5:
                    delta = random.uniform(0.0, 0.02)
                else:
                    delta = -0.05

                feature_count = min(6, feature_count + 1)
                self.cur_state.feature_count = feature_count
                new_accuracy = cur_accuracy + delta

        # -------------------------------------------------

        elif action_type == 'remove_feature':
            if feature_count == 1:
                new_accuracy = cur_accuracy
                penalty = -0.01   # <-- FIXED
            else:
                if feature_count > 5:
                    delta = random.uniform(0.03, 0.07)
                elif feature_count >= 3:
                    delta = 0
                else:
                    delta = -0.05

                feature_count = max(1, feature_count - 1)
                self.cur_state.feature_count = feature_count
                new_accuracy = cur_accuracy + delta

        # -------------------------------------------------

        else:
            penalty = -0.05
            new_accuracy = cur_accuracy

        # ------------------ COMMON UPDATE ------------------

        new_accuracy = round(max(0.0, min(1.0, new_accuracy)), 2)

        self.cur_state.accuracy = new_accuracy
        self.cur_state.precision = round(new_accuracy - 0.05, 2)
        self.cur_state.recall = round(new_accuracy - 0.03, 2)

        # ------------------ REWARD ------------------

        new_score = compute_score(self.cur_state)
        progress = new_score - old_score
        reward = progress + penalty   # <-- CLEAN FORMULA

        # bonus
        if new_accuracy >= 0.9:
            reward += 0.05

        reward = round(reward, 2)
        


        # ------------------ DONE ------------------

        self.step_count += 1

        score = compute_score(self.cur_state)


        if self.task_name == "stability_optimization":
            done = (
                self.step_count >= self.max_steps
                or (score >= 0.80 and abs(progress) < 0.01)   # lower threshold for stability task
            )
        else:
            done = (
                score >= 0.85
                or self.step_count >= self.max_steps
            )
        # ------------------ INFO ------------------

        info = {
            "accuracy": self.cur_state.accuracy,
            "step_count": self.step_count,
            "model_type": self.cur_state.model_type,
            "score": compute_score(self.cur_state),
            "task_score": self.grade_task(self.task_name, self.step_count) 
        }
        #print(f"DEBUG → task={self.task_name}, score={score}, steps={self.step_count}, done={done}")
        return self.cur_state, reward, done, info

    
    def state(self):
        return self.cur_state
    
    def grade_task(self, task_name, steps):
        if self.cur_state is None:
            return 0.01
        
        score = compute_score(self.cur_state)

        if task_name == "fix_basics":
            return max(0.01, min(score / 0.75, 0.99))

        elif task_name == "optimize_features":
            if 3 <= self.cur_state.feature_count <= 5:
                score += 0.05
            return max(0.01, min(score / 0.85, 0.99))
        elif task_name == "full_pipeline_optimization":
            step_penalty = 0.01 * steps
            final_score = score - step_penalty
            return max(0.01, min(final_score, 0.99))
        
        elif task_name == "stability_optimization":
            # penalize unnecessary changes (too many steps)
            step_penalty = 0.015 * steps
            final_score = score - step_penalty
            return max(0.01, min(final_score, 0.99))

        return 0.01