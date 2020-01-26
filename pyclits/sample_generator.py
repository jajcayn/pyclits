import roessler_system

def samples_from_arrays(data ,**kwargs):
    if "history" in kwargs:
        history = kwargs["history"]
    else:
        history = 5

    sampled_dataset = []

    shape_of_array = data.shape
    for item in range(shape_of_array[1]-history):
        sample = []
        for hist in range(history):
            for dim in range(shape_of_array[0]):
                sample.append(data[dim, item + hist])
        sampled_dataset.append(sample)

    return sampled_dataset



if __name__ == "__main__":
    kwargs = {}
    kwargs["tStop"] = 50
    sol = roessler_system.roessler_oscillator(**kwargs)
    print(sol)

    samples = samples_from_arrays(sol.y)
    print(len(samples), sol.y.shape)