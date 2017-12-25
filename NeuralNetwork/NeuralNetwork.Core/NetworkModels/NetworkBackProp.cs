using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.NetworkModels
{
    public class NetworkBackProp : Network
    {
        public double LearnRate { get; set; }
        public double Momentum { get; set; }

        public NetworkBackProp(Type hiddenActivationType = null, Type outputActivationType = null)
            : base(hiddenActivationType, outputActivationType)
        {
            LearnRate = 0;
            Momentum = 0;
        }

        public NetworkBackProp(int inputSize, int[] hiddenSizes, int outputSize, double? learnRate = null, double? momentum = null,
            Type hiddenActivationType = null, Type outputActivationType = null) : base(inputSize, hiddenSizes, outputSize, hiddenActivationType, outputActivationType)
        {
            LearnRate = learnRate ?? .4;
            Momentum = momentum ?? .9;
        }

        public void Train(List<DataSet> dataSets, int numEpochs)
        {
            for (var i = 0; i < numEpochs; i++)
            {
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(dataSet.Values);
                    BackPropagate(dataSet.Targets);
                }
            }
        }

        public void Train(List<DataSet> dataSets, double minimumError)
        {
            var error = 1.0;
            var numEpochs = 0;

            while (error > minimumError && numEpochs < int.MaxValue)
            {
                var errors = new List<double>();
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(dataSet.Values);
                    BackPropagate(dataSet.Targets);
                    errors.Add(CalculateError(dataSet.Targets));
                }
                error = errors.Average();
                numEpochs++;
            }
        }

        private void BackPropagate(params double[] targets)
        {
            var i = 0;
            OutputLayer.ForEach(a => a.CalculateGradient(targets[i++]));
            HiddenLayers.Reverse();
            HiddenLayers.ForEach(a => a.ForEach(b => b.CalculateGradient()));
            HiddenLayers.ForEach(a => a.ForEach(b => b.UpdateWeights(LearnRate, Momentum)));
            HiddenLayers.Reverse();
            OutputLayer.ForEach(a => a.UpdateWeights(LearnRate, Momentum));
        }
    }
}