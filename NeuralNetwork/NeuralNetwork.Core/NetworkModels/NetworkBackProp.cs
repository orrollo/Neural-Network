using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetwork.Core.Params;

namespace NeuralNetwork.Core.NetworkModels
{
    public class NetworkRProp : Network
    {
        public NetworkRProp(Type hiddenActivationType = null, Type outputActivationType = null)
            : base(hiddenActivationType, outputActivationType)
        {
        }

        public NetworkRProp(int inputSize, int[] hiddenSizes, int outputSize, Type hiddenActivationType = null, Type outputActivationType = null) 
            : base(inputSize, hiddenSizes, outputSize, hiddenActivationType, outputActivationType)
        {
        }

        public override void Train(List<DataSet> dataSets, TrainParams trainParams)
        {
            var rpParams = trainParams as RPropTrainParams;
            if (rpParams == null) throw new ArgumentException("params must be <RPropTrainParams> object");
            //
            Dictionary<object, double> dE = new Dictionary<object, double>(),
                dEold = new Dictionary<object, double>(),
                delta = new Dictionary<object, double>(),
                dW = new Dictionary<object, double>();

            double koefPos = 1.2, koefNeg = 0.5;
            double deltaMax = 50;
            double deltaMin = 1e-6;
            double deltaInit = 0.1;

            for (int epoch = 0; epoch < rpParams.NumEpochs; epoch++)
            {
                // exchange data objects
                var t = dE;
                dE = dEold;
                dEold = t;
                dE.Clear();
                //
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(dataSet.Values);
                    // calc gradients
                    CalculateGradients(dataSet.Targets, rpParams);
                    // collect for synapses and biases
                    HiddenLayers.ForEach(layer => ProcessLayer(layer, dE));
                    ProcessLayer(OutputLayer, dE);
                }
                // process weights
                foreach (var pair in dE)
                {
                    var key = pair.Key;

                    if (!delta.ContainsKey(key)) delta[key] = deltaInit; // change to train parameter

                    var prevDiff = dEold.ContainsKey(key) ? dEold[key] : 0.0;
                    var newDiff = pair.Value;
                    //
                    var diffChange = prevDiff * newDiff;
                    var chg = 0.0;
                    if (diffChange > 0)
                    {
                        delta[key] = Math.Min(deltaMax, delta[key] * koefPos);
                        dW[key] = (newDiff > 0 ? -1 : 1) * delta[key];
                        chg = dW[key];
                    }
                    else if (diffChange < 0)
                    {
                        delta[key] = Math.Max(deltaMin, delta[key] * koefNeg);
                        chg = -dW[key];
                    }
                    else
                    {
                        dW[key] = (newDiff > 0 ? -1 : 1) * delta[key];
                        chg = dW[key];
                    }
                    //
                    var neuron = key as Neuron;
                    var synapse = key as Synapse;
                    if (neuron != null)
                    {
                        neuron.BiasDelta = chg;
                        neuron.Bias += chg;
                    }
                    else if (synapse != null)
                    {
                        synapse.WeightDelta = chg;
                        synapse.Weight += chg;
                    }
                }
            }

        }

        private static void ProcessLayer(List<Neuron> layer, Dictionary<object, double> d)
        {
            layer.ForEach(neuron =>
            {
                if (!d.ContainsKey(neuron)) d[neuron] = 0.0;
                neuron.InputSynapses.ForEach(synapse =>
                {
                    if (!d.ContainsKey(synapse)) d[synapse] = 0.0;
                    d[neuron] += synapse.OutputNeuron.Gradient;
                    d[synapse] += synapse.OutputNeuron.Gradient*synapse.InputNeuron.Value;
                });
            });
        }

        private void CalculateGradients(double[] targets, RPropTrainParams rpParams)
        {
            var i = 0;
            OutputLayer.ForEach(a => a.CalculateGradient(targets[i++]));
            HiddenLayers.Reverse();
            if (rpParams.UseMultiThreading)
                HiddenLayers.ForEach(a => Parallel.ForEach(a, rpParams.ParallelOptionsOptions, b => b.CalculateGradient()));
            else
                HiddenLayers.ForEach(a => a.ForEach(b => b.CalculateGradient()));
            HiddenLayers.Reverse();
        }
    }

    public class NetworkBackProp : Network
    {
        public NetworkBackProp(Type hiddenActivationType = null, Type outputActivationType = null)
            : base(hiddenActivationType, outputActivationType)
        {
        }

        public NetworkBackProp(int inputSize, int[] hiddenSizes, int outputSize, Type hiddenActivationType = null, Type outputActivationType = null) 
            : base(inputSize, hiddenSizes, outputSize, hiddenActivationType, outputActivationType)
        {
        }

        public override void Train(List<DataSet> dataSets, TrainParams trainParams)
        {
            var bpParams = trainParams as BackPropTrainParams;
            if (bpParams == null) throw new ArgumentException("params must be <BackPropTrainParams> object");
            if (bpParams.Training == TrainingType.Epoch) 
                TrainByEpochs(dataSets, bpParams);
            else if (bpParams.Training == TrainingType.MinimumError)
                TrainByError(dataSets, bpParams);
            else
                throw new ArgumentException("unknown training type");
        }

        public void TrainByEpochs(List<DataSet> dataSets, BackPropTrainParams bpp)
        {
            for (var i = 0; i < bpp.NumEpochs; i++)
            {
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(dataSet.Values);
                    BackPropagate(bpp, dataSet.Targets);
                }
            }
            bpp.ResultEpochs = bpp.NumEpochs;
        }

        public void TrainByError(List<DataSet> dataSets, BackPropTrainParams bpp)
        {
            var error = 1.0;
            var numEpochs = 0;
            var rnd = new Random();
            var src = new List<DataSet>(dataSets);

            var count = src.Count;
            while (error > bpp.MinimumError && numEpochs < int.MaxValue)
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
                    BackPropagate(bpp, dataSet.Targets);
                }
                error = CalcErrorForData(src);
                numEpochs++;
            }
            bpp.ResultEpochs = numEpochs;
        }

        private double CalcErrorForData(List<DataSet> src)
        {
            var error = 0.0;
            foreach (var dataSet in src)
            {
                ForwardPropagate(dataSet.Values);
                error += CalculateError(dataSet.Targets);
            }
            error /= src.Count;
            return error;
        }

        private void BackPropagate(BackPropTrainParams bpp, params double[] targets)
        {
            var i = 0;
            double learnRate = bpp.LearnRate, momentum = bpp.Momentum;

            OutputLayer.ForEach(a => a.CalculateGradient(targets[i++]));
            
            HiddenLayers.Reverse();
            if (bpp.UseMultiThreading)
            {
                HiddenLayers.ForEach(a => Parallel.ForEach(a, bpp.ParallelOptionsOptions, b => b.CalculateGradient()));
                HiddenLayers.ForEach(a => Parallel.ForEach(a, bpp.ParallelOptionsOptions, b => b.UpdateWeights(learnRate, momentum)));
            }
            else
            {
                HiddenLayers.ForEach(a => a.ForEach(b => b.CalculateGradient()));
                HiddenLayers.ForEach(a => a.ForEach(b => b.UpdateWeights(learnRate, momentum)));
            }
            HiddenLayers.Reverse();

            if (bpp.UseMultiThreading)
            {
                Parallel.ForEach(OutputLayer, bpp.ParallelOptionsOptions, a => a.UpdateWeights(learnRate, momentum));
            }
            else
            {
                OutputLayer.ForEach(a => a.UpdateWeights(learnRate, momentum));
            }
        }
    }
}