import time


def profile(tag='profile', warmup=0, repeat=1):
    def _profile(func):
        def profile_wrapper(*args, **kwargs):
            print('[{}] function {} is warming up for {} time(s)'.format(
                tag, func.__name__, warmup))
            for _ in range(warmup):
                rtn = func(*args, **kwargs)
            print('[{}] function {} is calling for {} time(s)'.format(
                tag, func.__name__, repeat))
            in_time = time.time()
            for _ in range(repeat):
                rtn = func(*args, **kwargs)
            out_time = time.time()
            print('[{}] Average call time for function {} is {}'.format(
                tag, func.__name__, (out_time - in_time)/repeat))
            return rtn
        return profile_wrapper
    return _profile