using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NeuralNetwork.Core.NetworkModels;

namespace NeuralNetwork.Core.Learning
{
    public class BackPropLearning : BaseLearning
    {
        public Network Net { get; protected set; }

        public BackPropLearning(Network net)
        {
            Net = net;
        }

        private int _numEpochs = int.MaxValue;
        public int NumEpochs
        {
            get { return _numEpochs; }
            set { _numEpochs = value; }
        }

        public double MinimumError { get; set; }
        public int ResultEpochs { get; set; }

        public double LearnRate { get; set; }
        public double Momentum { get; set; }

        public void TrainByEpochs(List<DataSet> dataSets)
        {
            var src = new List<DataSet>(dataSets); 
            for (var i = 0; i < NumEpochs; i++)
            {
                ShuffleData(src);
                foreach (var dataSet in src)
                {
                    Net.ForwardPropagate(dataSet.Values);
                    BackPropagate(dataSet.Targets);
                }
            }
            ResultEpochs = NumEpochs;
        }

        public void TrainByError(List<DataSet> dataSets, int resetEpochsNumber = -1)
        {
            var error = 1.0;
            var numEpochs = 0;
            var src = new List<DataSet>(dataSets);

            var count = src.Count;
            while (error > MinimumError && numEpochs < int.MaxValue)
            {
                if (resetEpochsNumber != -1 && numEpochs % resetEpochsNumber == 0) ResetLearning();
                if ((numEpochs % count) == 0) ShuffleData(src);
                foreach (var dataSet in src)
                {
                    Net.ForwardPropagate(dataSet.Values);
                    BackPropagate(dataSet.Targets);
                }
                error = CalcErrorForData(src);
                numEpochs++;
            }
            ResultEpochs = numEpochs;
        }

        private void ResetLearning()
        {
            Net.ResetNetwork();
            biasDeltas.Clear();
            weightDeltas.Clear();
        }

        private double CalcErrorForData(List<DataSet> src)
        {
            var error = 0.0;
            foreach (var dataSet in src)
            {
                Net.ForwardPropagate(dataSet.Values);
                error += Net.CalculateError(dataSet.Targets);
            }
            error /= src.Count;
            return error;
        }

        protected Dictionary<Neuron,double> biasDeltas = new Dictionary<Neuron, double>();
        protected Dictionary<Synapse,double> weightDeltas = new Dictionary<Synapse, double>();

        protected override void UpdateNeuronWeights(Neuron neuron)
        {
            if (!biasDeltas.ContainsKey(neuron)) biasDeltas[neuron] = 0.0;
            foreach (var synapse in neuron.InputSynapses) if (!weightDeltas.ContainsKey(synapse)) weightDeltas[synapse] = 0.0;

            var biasDelta = LearnRate*neuron.Gradient;
            neuron.Bias += biasDelta + Momentum*biasDeltas[neuron];
            biasDeltas[neuron] = biasDelta;

            foreach (var synapse in neuron.InputSynapses)
            {
                var weightDelta = LearnRate * neuron.Gradient * synapse.InputNeuron.Value;
                synapse.Weight += weightDelta + Momentum * weightDeltas[synapse];
                weightDeltas[synapse] = weightDelta;
            }
        }

        private void BackPropagate(params double[] targets)
        {
            var i = 0;
            double learnRate = LearnRate, momentum = Momentum;

            Net.OutputLayer.ForEach(neuron => neuron.CalculateGradient(targets[i++]));

            Net.HiddenLayers.Reverse();
            if (UseMultiThreading)
            {
                Net.HiddenLayers.ForEach(neurons => Parallel.ForEach(neurons, ParallelOptionsOptions, neuron => neuron.CalculateGradient()));
                Net.HiddenLayers.ForEach(neurons => Parallel.ForEach(neurons, ParallelOptionsOptions, UpdateNeuronWeights));
            }
            else
            {
                Net.HiddenLayers.ForEach(neurons => neurons.ForEach(neuron => neuron.CalculateGradient()));
                Net.HiddenLayers.ForEach(neurons => neurons.ForEach(UpdateNeuronWeights));
            }
            Net.HiddenLayers.Reverse();

            if (UseMultiThreading)
            {
                Parallel.ForEach(Net.OutputLayer, ParallelOptionsOptions, UpdateNeuronWeights);
            }
            else
            {
                Net.OutputLayer.ForEach(UpdateNeuronWeights);
            }
        }
    }
}