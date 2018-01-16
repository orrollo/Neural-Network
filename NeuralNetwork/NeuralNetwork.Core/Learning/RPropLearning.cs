using System;
using System.Collections.Generic;
using NeuralNetwork.Core.NetworkModels;

namespace NeuralNetwork.Core.Learning
{
    public class RPropLearning : BaseLearning
    {
        public Network Net { get; protected set; }

        public RPropLearning(Network net)
        {
            Net = net;
            UseMultiThreading = false;
        }

        public void Train(List<DataSet> dataSets)
        {
            Dictionary<Neuron, double> neuronErrors = new Dictionary<Neuron, double>(), prevErrors = null;
            Dictionary<object, double> weightGradients = new Dictionary<object, double>(), prevGradients = null;

            // calc first errors and gradients
            CalcErrorsAndGradients(dataSets, neuronErrors, weightGradients);
            // make random changes to weights



        }

        private void CalcErrorsAndGradients(List<DataSet> dataSets, Dictionary<Neuron, double> neuronErrors, Dictionary<object, double> weightGradients)
        {
            foreach (var dataSet in dataSets)
            {
                Net.ForwardPropagate(dataSet.Values);
                // calc errors for all neurons and gradients for all weights
                var i = 0;
                Net.OutputLayer.ForEach(neuron =>
                {
                    neuron.CalculateGradient(dataSet.Targets[i++]);
                    if (!neuronErrors.ContainsKey(neuron)) neuronErrors[neuron] = 0;
                    neuronErrors[neuron] += neuron.Gradient;
                });
                for (int idx = Net.HiddenLayers.Count - 1; idx >= 0; idx--)
                {
                    var neurons = Net.HiddenLayers[idx];
                    neurons.ForEach(neuron =>
                    {
                        neuron.CalculateGradient();
                        if (!neuronErrors.ContainsKey(neuron)) neuronErrors[neuron] = 0;
                        neuronErrors[neuron] += neuron.Gradient;
                    });
                }
            }
            // for each neuron calc weights gradients
            foreach (var neuron in neuronErrors.Keys)
            {
                weightGradients[neuron] = neuron.Gradient/neuron.Bias;
                foreach (var synapse in neuron.InputSynapses) weightGradients[synapse] = neuron.Gradient/synapse.Weight;
            }
        }

        protected override void UpdateNeuronWeights(Neuron neuron)
        {
            throw new NotImplementedException();
        }
    }
}