from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import librosa

def parallel_process(array, function, n_jobs=4, use_kwargs=False, front_num=1):
    """
        A parallel version of the map function with a progress bar.
        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    else:
        front = []
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
            #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def check_sources_not_equal_to_mix(jams_path):
    mix, sr = librosa.load(jams_path.replace('.json', '.wav'), sr=None)
    jam = jams.load(jams_path)
    data = jam.annotations[0]['data']['value']
    sources = [mix]
    for d in data:
        if d['role'] == 'foreground':
            sources.append(librosa.load(d['saved_source_file'], sr=None)[0])
    if len(set([x.shape[0] for x in sources])) > 1:
        tqdm.write('Not all equal length!')
        return True
    return False