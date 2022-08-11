# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Import modules

# %%
import matplotlib.pyplot as plt

from collections import namedtuple
import time

import os

import torch
import numpy as np
from sklearn.decomposition import PCA

import ase.io

import resource
import multiprocess as mp

from rascal.representations import SphericalInvariants

print("Resource time limit:", resource.getrlimit(resource.RLIMIT_CPU))

torch.set_num_threads(64)
torch.set_default_dtype(torch.float64)

# %% [markdown]
# ### Enable GPU Accelaration

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# %% [markdown]
# ### Setup models

# %%
class Coupling(torch.nn.Module):
    # Gets SOAP power spectrum in alpha space and returns a reduced power spectrum in J space
    def __init__(self, train_info, d_j):
        super().__init__()

        self.train_info = train_info

        # Initialise the U matrix
        self.d_j = d_j
        self.Kmatrix = self.initialisation()
        self.Umatrix = torch.nn.Parameter(
            self.PCA(self.Kmatrix)
        )  # is this paramtere declaration necessary

    def forward(self, features, info):
        # Gets SOAP features in alpha and returns SOAP features in J
        features = self.packAlphato3D(features, info)
        P = self.transitionMatrix(self.Umatrix, info)
        features = self.unpackJto2D(features, P, info)
        return features

    def PCA(self, Kmatrix):
        # Perform PCA to reduce alpha space to pseudo-element (J) space
        pca = PCA(n_components=self.d_j)
        Umatrix = pca.fit_transform(Kmatrix)
        return torch.nn.Parameter(torch.tensor(Umatrix))

    def packAlphato3D(self, features, info):
        # Reshape the densified SOAP matrix from 2D to 3D
        # where the third axis now corresponds to alpha, alpha' pairs
        features_alpha_2d = features
        features_alpha_3d = features_alpha_2d.reshape(
            info.environments, len(info.alphaPairs), info.soap
        )

        # swap axes to get alpha alpha' to be the last
        # first axis: environments
        # second axis: n n' l combinations
        # third axis: alpha alpha' pairs
        features_alpha_3d = features_alpha_3d.swapaxes(1, 2)
        return features_alpha_3d

    def transitionMatrix(self, Umatrix, info):
        # Compute transition matrix
        # TODO: GPU optimization by turning this into a matrix operation
        P = torch.zeros((len(info.alphaPairs), len(info.jPairs)))
        for alphaPair in enumerate(info.alphaPairs):
            # translate atomic numbers to indices
            alpha1 = info.elementsToIndices.index(alphaPair[1][0])
            alpha2 = info.elementsToIndices.index(alphaPair[1][1])
            for jPair in enumerate(info.jPairs):
                # reduce by one to turn into a matrix index
                j1 = jPair[1][0] - 1
                j2 = jPair[1][1] - 1
                P[alphaPair[0], jPair[0]] = Umatrix[alpha1, j1] * Umatrix[alpha2, j2]
        return P

    def unpackJto2D(self, features, P, info):
        # Perform the transition from alpha to J and unpack into 2D
        features_alpha_3d = torch.tensor(features)
        features_J_3d = features_alpha_3d @ P

        features_J_2d = features_J_3d.reshape(
            info.environments, info.soap * len(info.jPairs)
        )
        return features_J_2d

    def initialisation(self):
        # Set up the data architecture and load the constants
        constants = self.train_info.constants
        Kmatrix = np.zeros((self.train_info.elements, self.train_info.elements))
        for i, atomicNum1 in enumerate(self.train_info.elementsToIndices):
            index1 = list(constants.number).index(str(atomicNum1))
            for j, atomicNum2 in enumerate(self.train_info.elementsToIndices):
                index2 = list(constants.number).index(str(atomicNum2))
                epsilons = (
                    float(constants.electronegativity[index1]),
                    float(constants.electronegativity[index2]),
                )
                radii = (
                    float(constants.radius[index1]),
                    float(constants.radius[index2]),
                )
                Kmatrix[i][j] = self.init_coupling_parameter(
                    epsilons[0], epsilons[1], 1, radii[0], radii[1], 1
                )
        return Kmatrix

    def init_coupling_parameter(
        self, epsilon1, epsilon2, sigmaEpsilon, radius1, radius2, sigmaRadius
    ):
        # Returns the initialization coupling parameters for a pair of elements
        exponent1 = -((epsilon1 - epsilon2) ** 2) / (2 * sigmaEpsilon ** 2)
        exponent2 = -((radius1 - radius2) ** 2) / (2 * sigmaRadius ** 2)
        return np.exp(exponent1 + exponent2)


# %%
class Energy(torch.nn.Module):
    # Gets the reduced SOAP power spectrum and returns the energy
    def __init__(self, train_data, zeta, d_j):
        super().__init__()
        self.zeta = zeta

        # Store train values
        self.train_features = train_data.descriptor.values

        self.train_info = train_data.info

        self.train_labels = torch.tensor(train_data.structureEnergies)
        self.std_label = train_data.structureEnergies.std()
        self.mean_label = train_data.structureEnergies.mean()
        self.train_labels = (self.train_labels - self.mean_label) / self.std_label

        self.coupling = Coupling(self.train_info, d_j)

    def forward(self, features, info):
        # Train the weights to solve for energy
        if self.training == True:
            train_features = self.coupling(self.train_features, self.train_info)
            train_features = self.sumFeatures(train_features, self.train_info)
            train_kernel = self.kernel(train_features, train_features)
            self.weights = torch.linalg.solve(train_kernel, self.train_labels)

            # Store train features as a reference for validation
            self.train_features_coupled = train_features

            # Detach tensors that should not optimize coupling
            self.weights = self.weights.detach()
            self.train_features_coupled = self.train_features_coupled.detach()

        # Test to optimize coupling parameter
        features = self.coupling(features, info)
        features = self.sumFeatures(features, info)
        kernel = self.kernel(features, self.train_features_coupled)
        output = ((kernel @ self.weights) * self.std_label) + self.mean_label
        return output

    def kernel(self, features, refFeatures):
        # Compute the kernel
        return torch.pow(features @ refFeatures.T, self.zeta)

    def sumFeatures(self, features, info):
        # Summation over environments in the SOAP features vector
        summedFeatures = torch.zeros(info.structures, (info.soap * len(info.jPairs)))
        # summedFeatures = torch.zeros(info.structures, (info.soap * len(info.alphaPairs)))
        q = np.arange(0, info.environments, step=info.environmentsPerStructure)
        for envIndex, startEnv in enumerate(q):
            stopEnv = startEnv + info.environmentsPerStructure
            summedFeatures[envIndex, :] = torch.sum(features[startEnv:stopEnv], 0)
        return summedFeatures


# %%
class AlchemicalModel(torch.nn.Module):
    def __init__(self, train_data, zeta, d_j):
        super().__init__()

        self.energy = Energy(train_data, zeta, d_j)

    def forward(self, data):
        # Gets the data and returns the energy
        return self.energy(data.descriptor.values, data.info)


# %% [markdown]
# ### Used elements setup

# %%
# Retrieve information about elements and define them as constants
Constants = namedtuple("Constants", ["symbol", "number", "electronegativity", "radius"])
elementData = np.genfromtxt("data/element_data.txt", skip_header=True, dtype="str")
constants = Constants(
    elementData[:, 0], elementData[:, 1], elementData[:, 2], elementData[:, 3]
)

# %% [markdown]
# ### Parameters

# %%
# Define hyper-parameters for librascal
max_angular = 9
max_radial = 12
atomicNumbers = []
for atomNumber in constants.number:
    atomicNumbers.append(int(atomNumber))

HYPER_PARAMETERS = {
    "soap_type": "PowerSpectrum",
    "interaction_cutoff": 5.0,
    "max_angular": max_angular,
    "max_radial": max_radial,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",  # not sure about this
    "cutoff_smooth_width": 0.5,
    "normalize": False,
    "radial_basis": "GTO",
    "compute_gradients": False,
    "expansion_by_species_method": "user defined",
    "global_species": atomicNumbers,
}

# %% [markdown]
# ### Prepare SOAP & energy input data

# %%
def read_data(sampleSize):
    # Read the input data - structures (.xyz) and energies (.dat)
    # Returns a list where each entry is a list of the structure and the energy of that structure
    structures = ase.io.read("data/elpasolites_10590.xyz", "0:{}".format(sampleSize))
    energies = np.genfromtxt("data/elpasolites_10590_evpa.dat", max_rows=sampleSize)
    structuresAndEnergies = []
    for i, el in enumerate(structures):
        structuresAndEnergies.append([el, energies[i]])
    return np.array(structuresAndEnergies, dtype=list)


def SOAP(hypers, structures, nproc):
    # Create the SOAP power spectrum vectors from structures (using librascal)
    # TODO: make nproc adjust itself automatically?
    # TODO: OR make nproc the largest possible and then do slicing automatically
    Descriptor = namedtuple("Descriptor", ["values", "samples"])
    numStructures = len(structures)
    numElements = len(hypers["global_species"])
    numSOAPentries = (
        ((numElements * (numElements - 1) // 2) + numElements)
        * hypers["max_radial"] ** 2
        * (hypers["max_angular"] + 1)
    )

    startMP = time.time()
    soap2 = SphericalInvariants(**hypers)
    manager = mp.Manager()
    return_dict = manager.dict()

    feat_shape = (numStructures * 10, numSOAPentries)
    samples_shape = (numStructures * 10, 3)

    mp_features = mp.Array("d", feat_shape[0] * feat_shape[1])
    mp_samples = mp.Array("q", samples_shape[0] * samples_shape[1])

    # TODO: assert that the divison of bla bla
    # TODO: clean up the code and put comments
    ntot = len(structures)

    def get_features(ret_dict, i):
        begin = (ntot // nproc) * i
        end = (ntot // nproc) * (i + 1)

        managers = soap2.transform(structures[begin:end])
        feat2 = managers.get_features(soap2)
        rep = managers.get_representation_info()

        for q in range(len(rep)):
            rep[q][0] = int(begin + q // 10)
            rep[q][1] = int((begin * 10) + q)

        begin = begin * 10
        end = end * 10
        features = np.frombuffer(mp_features.get_obj()).reshape(feat_shape)
        samples = np.frombuffer(mp_samples.get_obj()).reshape(samples_shape)

        features[begin:end] = feat2
        samples[begin:end] = rep

    jobs = []
    for i in range(nproc):
        p = mp.Process(
            target=get_features,
            args=(
                return_dict,
                i,
            ),
        )
        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()

    features = np.frombuffer(mp_features.get_obj()).reshape(feat_shape)
    samples = np.frombuffer(mp_samples.get_obj()).reshape(samples_shape)

    return Descriptor(features, samples)


def SOAP_SingleThread(hypers, structures):
    # Check that multiprocessing does not break anything
    Descriptor = namedtuple("Descriptor", ["values", "samples"])
    soap = SphericalInvariants(**hypers)
    manager = soap.transform(structures)
    features = manager.get_features(soap)
    samples = manager.get_representation_info()
    return Descriptor(features, samples)


def dataInfo(descriptor, atomicNumbers, d_j):
    # Returns distribution information of the descriptor
    info = []
    info.append(
        int(len((descriptor.samples[:, 1])) / len(np.unique(descriptor.samples[:, 0])))
    )  # number of environments per structure - TODO: optimize this somehow
    assert np.unique(descriptor.samples[:, 2]).size == len(
        atomicNumbers
    ), "too small dataset that does not contain all elements"
    info.append(
        list(np.unique(descriptor.samples[:, 2]).astype("int32"))
    )  # list linking elements to indices - TODO: optimize this by acessing atomicNumbers
    info.append(
        len(info[-1])
    )  # number of elements - TODO: optimize this by counting up hypers
    info += [
        *calculatePairs(descriptor, atomicNumbers, d_j)
    ]  # list of J pairs and list of alpha pairs
    info.append(len(np.unique(descriptor.samples[:, 0])))  # number of structures
    info.append(
        max_radial * max_radial * (max_angular + 1)
    )  # number of SOAP entries per Alpha / J pair
    info.append(descriptor.values.shape[0])  # number of environments
    return info


def calculatePairs(descriptor, atomicNumbers, d_j):
    def combinations(q):
        output = []
        for i in q:
            for j in q:
                if i > j:
                    continue
                output.append((i, j))
        return output

    # Create all possible J pairs
    possibleJ = list(range(1, d_j + 1))
    Jpairs = combinations(possibleJ)

    # Create all possible alpha pairs
    possibleA = atomicNumbers
    Apairs = combinations(possibleA)

    return Jpairs, Apairs


# %%
# Create shuffled sets of train (6k), test (2k) and validate (2k) datasets
Data = namedtuple(
    "Data", ["structureEnergies", "descriptor", "info"]
)  # define data type
DataInfo = namedtuple(
    "DataInfo",
    [
        "environmentsPerStructure",
        "elementsToIndices",
        "elements",
        "jPairs",
        "alphaPairs",
        "structures",
        "soap",
        "environments",
        "constants",
    ],
)
Constants = namedtuple("Constants", ["symbol", "number", "electronegativity", "radius"])


def dataset(trainSize, testSize, validateSize, d_j):
    sampleSize = (
        trainSize + testSize + validateSize
    )  # number of structures used from the dataset

    data = read_data(sampleSize)

    np.random.shuffle(data)

    # train dataset
    startDataset = time.time()
    structures = []
    energies = []
    for i in range(trainSize):
        structures.append(data[i][0])
        energies.append(data[i][1])
    energies = np.array(energies)
    train_descriptor = SOAP(HYPER_PARAMETERS, structures, 50)

    train_info = dataInfo(train_descriptor, atomicNumbers, d_j)
    # Add constants information
    train_info = DataInfo(*train_info, constants)
    # Asign datasets
    train_data = Data(energies, train_descriptor, train_info)
    print("Train dataset created after: ", time.time() - startDataset)

    # Clear memory
    del structures
    del energies

    # test dataset
    startDataset = time.time()
    structures = []
    energies = []
    for i in range(trainSize, trainSize + testSize):
        structures.append(data[i][0])
        energies.append(data[i][1])
    energies = np.array(energies)
    test_descriptor = SOAP(HYPER_PARAMETERS, structures, 50)

    test_info = dataInfo(test_descriptor, atomicNumbers, d_j)
    test_info = DataInfo(*test_info, constants)
    test_data = Data(energies, test_descriptor, test_info)
    print("Test dataset created after: ", time.time() - startDataset)

    # Clear memory
    del structures
    del energies

    # validate dataset
    startDataset = time.time()
    structures = []
    energies = []
    for i in range(trainSize + testSize, sampleSize):
        structures.append(data[i][0])
        energies.append(data[i][1])
    validate_energies = np.array(energies)
    validate_descriptor = SOAP(HYPER_PARAMETERS, structures, 50)

    validate_info = dataInfo(validate_descriptor, atomicNumbers, d_j)
    validate_info = DataInfo(*validate_info, constants)
    validate_data = Data(energies, validate_descriptor, validate_info)
    print("Validate dataset created after: ", time.time() - startDataset)

    # Clear memory
    del structures
    del energies

    # # Save into a file
    # if not os.path.exists("soap"):
    #     os.makedirs("soap")
    # np.save("soap/train_descriptor_values.npy", train_data.descriptor.values)
    # np.save("soap/train_descriptor_samples.npy", train_data.descriptor.samples)
    # np.save("soap/test_descriptor_values.npy", test_data.descriptor.values)
    # np.save("soap/test_descriptor_samples.npy", test_data.descriptor.samples)
    # np.save("soap/validate_descriptor_values.npy", validate_data.descriptor.values)
    # np.save("soap/validate_descriptor_samples.npy", validate_data.descriptor.samples)

    return train_data, test_data, validate_data


# %% [markdown]
# ### Model and Optimizer setup

# %%
def validate(model, config):
    model.train(mode=False)  # turn off training / testing
    s = time.time()
    actualEnergy = torch.tensor(config["validate_data"].structureEnergies)
    predictedEnergy = model(config["validate_data"])
    MAE_loss_function = torch.nn.L1Loss()
    MAE_loss = MAE_loss_function(predictedEnergy, actualEnergy)  # compute loss
    print("MAE loss is: ", MAE_loss.item(), flush=True)
    model.train(mode=True)
    print("validation took:", time.time() - s)
    return MAE_loss.item()


def optimizationRun(config):
    # Initiate the model with the training dataset
    start = time.time()
    model = AlchemicalModel(config["train_data"], config["zeta"], config["d_j"]).to(
        device
    )
    print("Model initiated. It took {0}".format(time.time() - start))

    # Setup an optimizer, a loss function and a scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learningRate"], momentum=config["momentum"]
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    loss_function = torch.nn.MSELoss()

    # TRAINING AND TESTING
    actualEnergy = torch.tensor(config["test_data"].structureEnergies)
    model.train()  # set model to training / testing

    optimizationLoss = []
    uParameters = []
    MAElosses = []
    # clamping = []

    print("Optimization started.")
    print(
        "SGD Optimizer, Initial Learning Rate: {0}, Momentum: {1}".format(
            config["learningRate"], config["momentum"]
        )
    )

    for epoch in range(config["epochs"]):
        start = time.time()
        optimizer.zero_grad()  # reset gradients
        predictedEnergy = model(config["test_data"])
        loss = loss_function(predictedEnergy, actualEnergy)  # compute loss
        optimizationLoss.append([epoch, loss.item()])
        loss.backward()  # backward propagate the loss

        # # Clamping
        # uGrad = list(model.parameters())[0].grad
        # uMean = uGrad.mean()
        # uStd = uGrad.std()
        # for i in enumerate(uGrad):
        #     for j in enumerate(i[1]):
        #         if abs(j[1]) > (uMean + 2 * uStd):
        #             uGradOrig = torch.clone(uGrad)
        #             uGrad[i[0], j[0]] = j[1].sign() * (uMean + 3 * uStd)
        #             print("!!!!!!!!!!CLAMPED!!!!!!!!!")
        #             clamping.append([epoch, uGradOrig, uGrad])

        optimizer.step()
        if epoch % 1 == 0:  # output log throughout
            print("----", epoch, loss.item(), "----")
        if epoch % 10 == 0:  # output log throughout
            u = np.copy(list(model.parameters())[0].detach().numpy())
            uParameters.append(u)
            MAElosses.append(validate(model, config))
        print("last epoch took {0}".format(time.time() - start))

    torch.save(model.state_dict(), "output-SGD-100/model-100.pt")

    return optimizationLoss, uParameters, MAElosses


# %% [markdown]
# # Main

# %%
def main():
    # Configuration setup
    config = {}
    config["trainSize"] = 4000  # number of structures used for training
    config["testSize"] = 2000  # number of structures used for testing
    config["validateSize"] = 2000  # number of structures used for validation
    config["epochs"] = 500  # number of optimization iterations

    config["zeta"] = 1
    config["d_j"] = 2

    config["learningRate"] = 0.1
    config["momentum"] = 0.9

    # Prepare the datasets
    start = time.time()
    print("Started creating the dataset.")
    config["train_data"], config["test_data"], config["validate_data"] = dataset(
        config["trainSize"], config["testSize"], config["validateSize"], config["d_j"]
    )
    print("Dataset created. It took {0}".format(time.time() - start))

    optimization_losses, uParameters, MAElosses = optimizationRun(config)

    np.savetxt("output-SGD-100/opt-losses.csv", optimization_losses, delimiter="\t")

    np.save("output-SGD-100/u.npy", np.array(uParameters))

    np.savetxt("output-SGD-100/MAE-losses.csv", MAElosses)

    # np.save("output-SGD-100/clamping.npy", np.array(clamping))


if __name__ == "__main__":
    main()


# %%


# %%


# %%
