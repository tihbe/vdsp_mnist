import numpy as np
from mnist import MNIST
from tqdm import tqdm
import matplotlib.pyplot as plt
from fire import Fire
from sklearn.model_selection import train_test_split
from collections import namedtuple

try:
    from numba import njit
except ModuleNotFoundError:
    njit = lambda i: i

    [print("-" * 51) for _ in range(3)]
    print("Numba not installed, the script will be much slower")
    [print("-" * 51) for _ in range(3)]

NetworkState = namedtuple("NetworkState", ("mem_pot_input", "mem_pot_output", "weights"))
NetworkParameters = namedtuple(
    "NetworkParameters",
    (
        "duration_per_sample",
        "duration_between_samples",
        "input_leak_cst",
        "input_bias",
        "output_leak_cst",
        "refractory_period",
        "threshold",
        "lateral_inhibition_period",
    ),
)
VDSPParameters = namedtuple("VDSPParameters", ("vdsp_lr",))


@njit
def vdsp(w, vmem, lr=1):
    cond_pot = vmem < 0
    cond_dep = vmem > 0

    evmem = np.exp(np.abs(vmem)) - 1

    return lr * ((1 - w) * evmem * cond_pot - w * evmem * cond_dep)


@njit
def run_one_sample(
    X,
    network_state: NetworkState,
    network_parameters: NetworkParameters,
    vdsp_parameters: VDSPParameters,
):
    mem_pot_input, mem_pot_output, weights = network_state

    params = network_parameters
    refractory_neurons = np.zeros(mem_pot_output.shape[0], dtype=np.int64)
    recorded_output_spikes = np.zeros((mem_pot_output.shape[0], params.duration_per_sample))
    for t in range(params.duration_per_sample):
        refractory_neurons = np.maximum(0, refractory_neurons - 1)
        non_refrac_neurons = refractory_neurons == 0
        mem_pot_input = mem_pot_input * params.input_leak_cst + params.input_bias + X
        input_spikes = mem_pot_input > 1
        mem_pot_input[input_spikes] = -1  # reset

        mem_pot_output[non_refrac_neurons] = mem_pot_output[
            non_refrac_neurons
        ] * params.output_leak_cst + input_spikes.astype(np.float64) @ (weights[:, non_refrac_neurons])
        output_spikes = mem_pot_output > params.threshold

        if np.any(output_spikes):
            spiking_neuron = np.argmax(mem_pot_output)  # This neuron is spiking
            recorded_output_spikes[spiking_neuron, t] = 1
            dw = vdsp(weights[:, spiking_neuron], mem_pot_input, vdsp_parameters.vdsp_lr)  # VDSP
            weights[:, spiking_neuron] += dw  # Apply plasticity
            mem_pot_output[spiking_neuron] = -1  # reset mem pot
            refractory_neurons[spiking_neuron] = params.refractory_period  # Set refrac for spiking neuron
            non_spiking_neurons = np.ones(mem_pot_output.shape[0], dtype=np.bool_)
            non_spiking_neurons[spiking_neuron] = False
            mem_pot_output[non_spiking_neurons] = 0
            refractory_neurons[non_spiking_neurons] = params.lateral_inhibition_period

    # Leakage in between samples
    mem_pot_input = mem_pot_input * np.power(params.input_leak_cst, params.duration_between_samples)
    mem_pot_output = mem_pot_output * np.power(params.output_leak_cst, params.duration_between_samples)
    refractory_neurons = np.maximum(0, refractory_neurons - params.duration_between_samples)

    new_state = NetworkState(mem_pot_input, mem_pot_output, weights)
    return (new_state, recorded_output_spikes)


def main(
    seed=0x1B,
    n_output_neurons=10,
    duration_per_sample=350,  # ms
    duration_between_samples=200,  # ms
    input_leak_cst=np.exp(-1 / 30),  # ms
    output_leak_cst=np.exp(-1 / 60),  # ms
    output_threshold=10,
    input_bias=0.032,
    refractory_period=5,  # ms
    lateral_inhibition_period=10,  # ms
    input_scale=0.00675,
    nb_epochs=1,
    vdsp_lr=0.001,
    with_validation=False,
    with_plots=True,
):
    print("Arguments:", locals())
    np.random.seed(seed)
    mndata = MNIST()
    images, labels = mndata.load_training()
    X_train, y_train = np.asarray(images), np.asarray(labels)
    X_train = X_train / 255 * input_scale

    if with_validation:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33)
    else:
        images, labels = mndata.load_testing()
        X_test, y_test = np.asarray(images), np.asarray(labels)
        X_test = X_test / 255 * input_scale

    # Random shuffling
    random_indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[random_indices], y_train[random_indices]

    mem_pot_input = np.zeros(784)
    mem_pot_output = np.zeros(n_output_neurons)
    weights = np.random.uniform(0, 1, size=(784, n_output_neurons))

    network_state = NetworkState(mem_pot_input, mem_pot_output, weights)
    network_parameters = NetworkParameters(
        duration_per_sample,
        duration_between_samples,
        input_leak_cst,
        input_bias,
        output_leak_cst,
        refractory_period,
        output_threshold,
        lateral_inhibition_period,
    )

    vdsp_parameters = VDSPParameters(vdsp_lr)
    vdsp_parameters_without_learning = VDSPParameters(0)

    for epoch in range(nb_epochs):
        for i, (X, y) in enumerate(zip(tqdm(X_train), y_train)):

            (network_state, recorded_output_spikes) = run_one_sample(
                X,
                network_state,
                network_parameters,
                vdsp_parameters,
            )

            if with_plots and (i + 1) % 5000 == 0:
                fig, axs = plt.subplots(2, n_output_neurons // 2, tight_layout=True)
                fig.suptitle(f"Receptive fields of output neurons at iteration {i+1}")
                axs = axs.flatten()
                for neuron in range(n_output_neurons):
                    if i > len(y_train) // 2:
                        axs[neuron].get_yaxis().set_visible(False)
                        axs[neuron].set_xticks([])
                    else:
                        axs[neuron].set_axis_off()
                    axs[neuron].imshow(weights[:, neuron].reshape(28, 28))

                fig.savefig(
                    f"receptive_fields_epoch_{epoch:02}_iteration_{i+1:05}.png",
                    bbox_inches="tight",
                )

    ### Compute spike counts for the training set without learning
    spike_counts = np.zeros((10, n_output_neurons))  # For every class, keep track of spike count per neuron
    for i, (X, y) in enumerate(zip(tqdm(X_train), y_train)):
        (network_state, recorded_output_spikes,) = run_one_sample(
            X,
            network_state,
            network_parameters,
            vdsp_parameters_without_learning,
        )

        spike_counts[y] += np.sum(recorded_output_spikes, axis=1)

    # Associate a label for every neuron based on its highest spike count per class
    labels = np.argmax(spike_counts, axis=0)
    nb_correct_classification_method_2 = 0
    for i, (X, y) in enumerate(zip(tqdm(X_test), y_test)):
        (network_state, recorded_output_spikes,) = run_one_sample(
            X,
            network_state,
            network_parameters,
            vdsp_parameters_without_learning,
        )

        ## Sum the spikes for all labeled neurons
        sum_of_spikes_for_sample = np.zeros(10)
        sum_of_output_spikes = np.sum(recorded_output_spikes, axis=1)
        np.add.at(sum_of_spikes_for_sample, labels, sum_of_output_spikes)
        output_label = np.argmax(sum_of_spikes_for_sample)
        nb_correct_classification_method_2 += y == output_label

    print(f"Final accuracy: {nb_correct_classification_method_2 / len(y_test):.5f}")
    return nb_correct_classification_method_2 / len(y_test)


if __name__ == "__main__":
    Fire(main)
