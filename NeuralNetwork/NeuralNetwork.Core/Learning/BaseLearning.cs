using System;
using System.Collections.Generic;
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
        protected Random _rnd = new Random();

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

        protected void ShuffleData(List<DataSet> src)
        {
            for (int i = 0; i < src.Count; i++)
            {
                int j = _rnd.Next(src.Count);
                if (i == j) continue;
                var set = src[i];
                src[i] = src[j];
                src[j] = set;
            }
        }
    }
}