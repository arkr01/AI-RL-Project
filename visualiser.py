import sys
import platform
import logging
import time
import random
import hashlib
from game_env import GameEnv
from solution import RLAgent

# automatic timeout handling will only be performed on Unix
if platform.system() != 'Windows':
    import signal
    WINDOWS = False
else:
    WINDOWS = True

DEBUG_MODE = False  # set to True to disable time limit checks

CRASH = 255
OVERTIME = 254

"""
Visualiser script.

Run this file to visualise the policy generated by your solver for a given input file. You may modify this file if
desired.

The return code produced by visualiser is your agent's score for the testcase (multiplied by 10 and represented as an
integer).

Visualiser seeds random outcomes to produce consistent policy performance between runs - if your code is deterministic
and does not exceed the time limit, visualiser will always produce the a consistent score.

The visualiser will automatically terminate your agent if it runs over 2x the allowed time limit for any step (on Unix
platforms only - not available on Windows). This feature can be disabled for debugging purposes by setting
DEBUG_MODE = True above.

COMP3702 2021 Assignment 3 Support Code

Last updated by njc 25/10/21
"""


class TimeOutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeOutException


def stable_hash(x):
    return hashlib.md5(str(x).encode('utf-8')).hexdigest()


def main(arglist):
    if len(arglist) != 1:
        print("Running this file tests executes your code and evaluates the performance of the generated policy for the"
              " given input file.")
        print("Usage: simulator.py [input_filename]")
        return

    input_file = arglist[0]
    if '/' in input_file:
        separator = '/'
    else:
        separator = '\\'
    init_seed = stable_hash(input_file.split(separator)[-1])
    random.seed(init_seed)
    env = GameEnv(input_file)

    # initialise RL Agent
    if not WINDOWS and not DEBUG_MODE:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)
    try:
        agent = RLAgent(env)
        if not WINDOWS and not DEBUG_MODE:
            signal.alarm(0)
    except TimeOutException as e:
        logging.exception(e)
        print("/!\\ Terminated due to running over 2x time limit in solver.__init__()")
        sys.exit(OVERTIME)
    except Exception as e:
        logging.exception(e)
        print("/!\\ Terminated due to exception generated in solver.__init__()")
        sys.exit(CRASH)

    # run training
    if not WINDOWS and not DEBUG_MODE:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(env.training_time + 1))
    try:
        t0 = time.time()
        agent.run_training()
        t_training = time.time() - t0
        if not WINDOWS and not DEBUG_MODE:
            signal.alarm(0)
    except TimeOutException as e:
        logging.exception(e)
        print("/!\\ Terminated due to running over 2x time limit in run_training()")
        sys.exit(OVERTIME)
    except Exception as e:
        logging.exception(e)
        print("/!\\ Terminated due to exception generated in run_training()")
        sys.exit(CRASH)
    training_reward = env.get_total_reward()

    # simulate episode
    t_eval_max = 0
    terminal = False
    reward = None
    eval_reward = 0
    persistent_state = env.get_init_state()
    print(f'    row: {persistent_state.row}, col: {persistent_state.col}, '
          f'gem status: {persistent_state.gem_status}')
    env.render(persistent_state)
    time.sleep(1)
    visit_count = {persistent_state.deepcopy(): 1}
    while not terminal and eval_reward > (env.eval_reward_tgt * 2):
        # query agent to select an action
        if not WINDOWS and not DEBUG_MODE:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(1))
        try:
            t0 = time.time()
            action = agent.select_action(persistent_state)
            t_online = time.time() - t0
            if not WINDOWS and not DEBUG_MODE:
                signal.alarm(0)
            if t_online > t_eval_max:
                t_eval_max = t_online
        except TimeOutException as e:
            logging.exception(e)
            print("/!\\ Terminated due to running over 2x time limit in select_action()")
            sys.exit(OVERTIME)
        except Exception as e:
            logging.exception(e)
            print("/!\\ Terminated due to exception generated in select_action()")
            sys.exit(CRASH)

        if action not in GameEnv.ACTIONS:
            print("/!\\ Unrecognised action selected by select_action()")
            sys.exit(CRASH)

        # simulate outcome of action
        seed = (init_seed + stable_hash(str((persistent_state.row, persistent_state.col, persistent_state.gem_status)))
                + stable_hash(visit_count[persistent_state]))
        valid, reward, persistent_state, terminal = env.perform_action(persistent_state, action, seed=seed)
        if not valid:
            print("/!\\ Invalid action selected by select_action()")
            sys.exit(CRASH)
        # render new state
        print(f'    row: {persistent_state.row}, col: {persistent_state.col}, '
              f'gem status: {persistent_state.gem_status}')
        env.render(persistent_state)
        time.sleep(1)
        # updated visited state count (for de-randomisation)
        ps = persistent_state.deepcopy()
        if ps in visit_count.keys():
            visit_count[ps] += 1
        else:
            visit_count[ps] = 1
        # update episode reward
        eval_reward += reward

    print(f"Level completed with a total rewards:\n\ttraining: {round(training_reward, 1)}"
          f"\n\tevaluation: {round(eval_reward, 1)}")

    # compute score for episode
    # run time deductions
    td_training = max(t_training - env.training_time - 1, 0) / env.training_time
    if td_training > 0:
        print(f'Exceeded offline time limit by {round(td_training * 100)}%')
    td_eval = max(t_eval_max - 1, 0)
    if td_eval > 0:
        print(f'Exceeded online time limit by {round(td_eval * 100)}%')
    td = min(td_training + td_eval, 1.0)
    # policy performance deductions
    pd_training = max(env.training_reward_tgt - training_reward, 0) / abs(env.training_reward_tgt)
    if pd_training > 1e-3:
        print(f'Below training reward target by {round(pd_training * 100)}%')
    pd_eval = max(env.eval_reward_tgt - eval_reward, 0) / abs(env.eval_reward_tgt)
    if pd_eval > 1e-3:
        print(f'Below evaluation reward target by {round(pd_eval * 100)}%')
    pd = min(pd_training + pd_eval, 1.0)
    # check that recorded training reward is feasible (detect case where a copy of game_env is used)
    if training_reward > (env.training_reward_tgt / 10):
        print('Simulator does not appear to have been used - make sure you use RLAgent.game_env.perform_action() in '
              'your learning process (to make sure training reward is measured accurately).')
        pd = 1.0
    # total deduction
    total_deduction = min(td + pd, 1.0)
    score = round(8.0 * (1.0 - total_deduction), 1)
    print(f'Score for this testcase is {score} / 8.0')
    ret_code = int(round(score * 10))
    # return code is score (out of 8.0) multiplied by 10
    sys.exit(ret_code)


if __name__ == '__main__':
    main(sys.argv[1:])

