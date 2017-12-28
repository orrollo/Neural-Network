using System;
using System.Threading.Tasks;
using NeuralNetwork.Core.NetworkModels;

namespace NeuralNetwork.Core.Learning
{
    public abstract class BaseLearning
    {
        private ParallelOptions _parallelOptionsOptions = new ParallelOptions()
        {
            MaxDegreeOfParallelism = MaxParallelCount
        };

        private bool _useMultiThreading = true;

        /// <summary>
        /// allow to use multithreadings, when it's possible
        /// </summary>
        public bool UseMultiThreading
        {
            get { return _useMultiThreading; }
            set { _useMultiThreading = value; }
        }

        /// <summary>
        /// parallel thread options, usually not need to change
        /// </summary>
        public ParallelOptions ParallelOptionsOptions
        {
            get { return _parallelOptionsOptions; }
            set { _parallelOptionsOptions = value; }
        }

        public static int MaxParallelCount
        {
            get { return Environment.ProcessorCount; }
        }

        protected abstract void UpdateNeuronWeights(Neuron neuron);
    }
}