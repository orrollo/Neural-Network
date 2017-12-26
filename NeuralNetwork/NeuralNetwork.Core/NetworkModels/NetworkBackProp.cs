using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Core.TrainParams;

namespace NeuralNetwork.Core.NetworkModels
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

        public override void Train(List<DataSet> dataSets, TrainParams.TrainParams trainParams)
        {
            var bpParams = trainParams as BackPropTrainParams;
            if (bpParams == null) throw new ArgumentException("params must be <BackPropTrainParams> object");
            if (bpParams.Training == TrainingType.Epoch) 
                bpParams.ResultEpochs = TrainByEpochs(dataSets, bpParams.NumEpochs);
            else if (bpParams.Training == TrainingType.MinimumError)
                bpParams.ResultEpochs = TrainByError(dataSets, bpParams.MinimumError);
            else
                throw new ArgumentException("unknown training type");
        }

        public int TrainByEpochs(List<DataSet> dataSets, int numEpochs)
        {
            for (var i = 0; i < numEpochs; i++)
            {
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(dataSet.Values);
                    BackPropagate(dataSet.Targets);
                }
            }
            return numEpochs;
        }

        public int TrainByError(List<DataSet> dataSets, double minimumError)
        {
            var error = 1.0;
            var numEpochs = 0;
            var rnd = new Random();
            var src = new List<DataSet>(dataSets);

            //var errors = new List<double>();
            var count = src.Count;
            while (error > minimumError && numEpochs < int.MaxValue)
            {
                if ((numEpochs % count) == 0)
                {
                    for (int i = 0; i < src.Count; i++)
                    {
                        int j = rnd.Next(src.Count);
                        if (i == j) continue;
                        var set = src[i];
                        src[i] = src[j];
                        src[j] = set;
                    }
                }
                foreach (var dataSet in src)
                {
                    ForwardPropagate(dataSet.Values);
                    BackPropagate(dataSet.Targets);
                }
                error = 0.0;
                foreach (var dataSet in src)
                {
                    ForwardPropagate(dataSet.Values);
                    error += CalculateError(dataSet.Targets);
                }
                error /= count;
                numEpochs++;
            }
            return numEpochs;
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